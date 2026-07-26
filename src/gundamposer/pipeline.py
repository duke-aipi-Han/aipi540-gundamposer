"""Stable Diffusion and OpenPose ControlNet generation with optional LoRA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image
import torch

from gundamposer.preprocessing import to_oriented_rgb
from gundamposer.prompts import NEGATIVE_PROMPT, build_prompt, normalize_prompt_text


BASE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_MODEL_ID = "lllyasviel/control_v11p_sd15_openpose"
MAX_SEED = 2**31 - 1
LORA_ADAPTER_NAME = "gundamposer"
DEFAULT_LORA_STRENGTH = 0.8


class GenerationError(ValueError):
    """Raised when generation inputs or outputs violate the fixed contract."""


@dataclass(frozen=True)
class GenerationSettings:
    width: int = 384
    height: int = 512
    num_inference_steps: int = 24
    guidance_scale: float = 7.0
    controlnet_conditioning_scale: float = 1.2


@dataclass(frozen=True)
class GenerationMetadata:
    seed: int
    prompt: str
    lora_strength: float
    controlnet_strength: float
    generation_time_seconds: float
    device: str


@dataclass(frozen=True)
class GenerationResult:
    image: Image.Image
    metadata: GenerationMetadata


def resolve_device(requested: str | None = None) -> str:
    """Choose CUDA, MPS, or CPU in that order unless explicitly requested."""

    if requested is not None:
        if requested not in {"cuda", "mps", "cpu"}:
            raise GenerationError("device must be one of: cuda, mps, cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise GenerationError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise GenerationError("MPS was requested but is not available")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GundamPoserPipeline:
    """One-image baseline generation with fixed, testable settings."""

    def __init__(
        self,
        pipeline: Any,
        *,
        device: str,
        settings: GenerationSettings | None = None,
        lora_loaded: bool = False,
    ) -> None:
        self._pipeline = pipeline
        self.device = device
        self.settings = settings or GenerationSettings()
        self.lora_loaded = lora_loaded

    @classmethod
    def load(
        cls,
        *,
        base_model_id: str = BASE_MODEL_ID,
        controlnet_model_id: str = CONTROLNET_MODEL_ID,
        device: str | None = None,
        cache_dir: str | None = None,
        settings: GenerationSettings | None = None,
        lora_path: str | Path | None = None,
    ) -> "GundamPoserPipeline":
        """Load public baseline models once and move them to the chosen device."""

        from diffusers import (
            ControlNetModel,
            DPMSolverMultistepScheduler,
            StableDiffusionControlNetPipeline,
        )

        resolved_device = resolve_device(device)
        dtype = torch.float16 if resolved_device == "cuda" else torch.float32
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            cache_dir=cache_dir,
            token=False,
        )
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
            cache_dir=cache_dir,
            token=False,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        if resolved_device == "mps":
            pipeline.enable_attention_slicing()
        lora_loaded = lora_path is not None
        if lora_path is not None:
            resolved_lora_path = Path(lora_path).expanduser().resolve()
            if not resolved_lora_path.is_file():
                raise GenerationError(f"LoRA adapter does not exist: {resolved_lora_path}")
            pipeline.load_lora_weights(
                str(resolved_lora_path),
                adapter_name=LORA_ADAPTER_NAME,
            )
        pipeline.to(resolved_device)
        return cls(
            pipeline,
            device=resolved_device,
            settings=settings,
            lora_loaded=lora_loaded,
        )

    def generate(
        self,
        pose_image: Image.Image,
        scene_prompt: str,
        seed: int,
        *,
        lora_strength: float = 0.0,
        prompt_override: str | None = None,
    ) -> GenerationResult:
        """Generate exactly one image from a body-pose map and scene text."""

        if not isinstance(pose_image, Image.Image):
            raise GenerationError("A detected pose map is required before generation.")
        if pose_image.size != (self.settings.width, self.settings.height):
            raise GenerationError(
                "The pose map must be "
                f"{self.settings.width}x{self.settings.height} pixels."
            )
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= MAX_SEED
        ):
            raise GenerationError(f"seed must be an integer from 0 to {MAX_SEED}")
        if (
            isinstance(lora_strength, bool)
            or not isinstance(lora_strength, (int, float))
            or not 0 <= lora_strength <= 2
        ):
            raise GenerationError("lora_strength must be a number from 0 to 2")
        normalized_lora_strength = float(lora_strength)
        if normalized_lora_strength > 0 and not self.lora_loaded:
            raise GenerationError(
                "A trained LoRA adapter is required for trained inference."
            )

        if self.lora_loaded:
            if normalized_lora_strength == 0:
                self._pipeline.disable_lora()
            else:
                self._pipeline.enable_lora()
                self._pipeline.set_adapters(
                    LORA_ADAPTER_NAME,
                    adapter_weights=normalized_lora_strength,
                )

        if prompt_override is None:
            prompt = build_prompt(scene_prompt)
        else:
            if not isinstance(prompt_override, str):
                raise GenerationError("prompt must be a string")
            prompt = normalize_prompt_text(prompt_override)
            if not prompt:
                raise GenerationError("prompt cannot be empty")
        control_image = to_oriented_rgb(pose_image)
        generator_device = "cuda" if self.device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        started = perf_counter()
        with torch.inference_mode():
            output = self._pipeline(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=control_image,
                width=self.settings.width,
                height=self.settings.height,
                num_images_per_prompt=1,
                num_inference_steps=self.settings.num_inference_steps,
                guidance_scale=self.settings.guidance_scale,
                controlnet_conditioning_scale=(
                    self.settings.controlnet_conditioning_scale
                ),
                control_guidance_start=0.0,
                control_guidance_end=1.0,
                generator=generator,
            )
        elapsed = perf_counter() - started
        if len(output.images) != 1:
            raise GenerationError("The pipeline must return exactly one image.")
        safety_flags = getattr(output, "nsfw_content_detected", None)
        if safety_flags and any(safety_flags):
            raise GenerationError(
                "The safety checker blocked this output. Try another prompt or seed."
            )

        return GenerationResult(
            image=output.images[0],
            metadata=GenerationMetadata(
                seed=seed,
                prompt=prompt,
                lora_strength=normalized_lora_strength,
                controlnet_strength=self.settings.controlnet_conditioning_scale,
                generation_time_seconds=elapsed,
                device=self.device,
            ),
        )
