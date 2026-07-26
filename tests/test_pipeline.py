from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image
import pytest
import torch

from gundamposer.pipeline import (
    BASE_MODEL_ID,
    CONTROLNET_MODEL_ID,
    DEFAULT_LORA_REPO_ID,
    DEFAULT_LORA_WEIGHT_NAME,
    GenerationError,
    GenerationSettings,
    GundamPoserPipeline,
    LORA_ADAPTER_NAME,
    MAX_SEED,
    resolve_device,
)
from gundamposer.prompts import NEGATIVE_PROMPT


class FakeDiffusionPipeline:
    def __init__(
        self,
        images: list[Image.Image] | None = None,
        *,
        safety_flags: list[bool] | None = None,
    ) -> None:
        self.images = images or [Image.new("RGB", (384, 512), "blue")]
        self.safety_flags = safety_flags
        self.arguments: dict[str, object] | None = None
        self.lora_enabled = False
        self.adapter_calls: list[tuple[str, float]] = []

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.arguments = kwargs
        return SimpleNamespace(
            images=self.images,
            nsfw_content_detected=self.safety_flags,
        )

    def disable_lora(self) -> None:
        self.lora_enabled = False

    def enable_lora(self) -> None:
        self.lora_enabled = True

    def set_adapters(self, name: str, *, adapter_weights: float) -> None:
        self.adapter_calls.append((name, adapter_weights))


def test_generate_wires_fixed_baseline_settings() -> None:
    fake = FakeDiffusionPipeline()
    pipeline = GundamPoserPipeline(fake, device="cpu")
    pose_image = Image.new("RGB", (384, 512), "black")

    result = pipeline.generate(pose_image, "industrial hangar", 123)

    assert fake.arguments is not None
    assert fake.arguments["prompt"].count("hwmecha") == 1
    assert "industrial hangar" in str(fake.arguments["prompt"])
    assert fake.arguments["negative_prompt"] == NEGATIVE_PROMPT
    assert fake.arguments["image"] is not pose_image
    assert fake.arguments["image"].size == (384, 512)  # type: ignore[union-attr]
    assert fake.arguments["width"] == 384
    assert fake.arguments["height"] == 512
    assert fake.arguments["num_images_per_prompt"] == 1
    assert fake.arguments["num_inference_steps"] == 24
    assert fake.arguments["guidance_scale"] == 7.0
    assert fake.arguments["controlnet_conditioning_scale"] == 1.2
    assert fake.arguments["control_guidance_start"] == 0.0
    assert fake.arguments["control_guidance_end"] == 1.0
    generator = fake.arguments["generator"]
    assert isinstance(generator, torch.Generator)
    assert generator.initial_seed() == 123
    assert result.image.size == (384, 512)
    assert result.metadata.seed == 123
    assert result.metadata.lora_strength == 0.0
    assert result.metadata.controlnet_strength == 1.2
    assert result.metadata.device == "cpu"
    assert result.metadata.generation_time_seconds >= 0


def test_generate_accepts_custom_fixed_settings() -> None:
    settings = GenerationSettings(
        num_inference_steps=12,
        guidance_scale=5.5,
        controlnet_conditioning_scale=0.7,
    )
    fake = FakeDiffusionPipeline()

    GundamPoserPipeline(fake, device="cpu", settings=settings).generate(
        Image.new("RGB", (384, 512)),
        "forest battlefield",
        1,
    )

    assert fake.arguments is not None
    assert fake.arguments["num_inference_steps"] == 12
    assert fake.arguments["guidance_scale"] == 5.5
    assert fake.arguments["controlnet_conditioning_scale"] == 0.7


def test_generate_uses_editable_prompt_override() -> None:
    fake = FakeDiffusionPipeline()
    pipeline = GundamPoserPipeline(fake, device="cpu")

    result = pipeline.generate(
        Image.new("RGB", (384, 512)),
        "neutral studio",
        42,
        prompt_override="  custom hwmecha pose,   blue armor  ",
    )

    assert fake.arguments is not None
    assert fake.arguments["prompt"] == "custom hwmecha pose, blue armor"
    assert result.metadata.prompt == "custom hwmecha pose, blue armor"


@pytest.mark.parametrize("prompt", ["", " \n "])
def test_generate_rejects_empty_explicit_prompt(prompt: str) -> None:
    with pytest.raises(GenerationError, match="prompt cannot be empty"):
        GundamPoserPipeline(FakeDiffusionPipeline(), device="cpu").generate(
            Image.new("RGB", (384, 512)),
            "neutral studio",
            42,
            prompt_override=prompt,
        )


def test_generate_switches_between_baseline_and_trained_adapter() -> None:
    fake = FakeDiffusionPipeline()
    pipeline = GundamPoserPipeline(fake, device="cpu", lora_loaded=True)
    pose = Image.new("RGB", (384, 512))

    baseline = pipeline.generate(pose, "studio", 1)
    trained = pipeline.generate(pose, "studio", 1, lora_strength=0.8)

    assert baseline.metadata.lora_strength == 0.0
    assert trained.metadata.lora_strength == 0.8
    assert fake.lora_enabled is True
    assert fake.adapter_calls == [(LORA_ADAPTER_NAME, 0.8)]


def test_generate_requires_loaded_adapter_for_positive_strength() -> None:
    with pytest.raises(GenerationError, match="trained LoRA"):
        GundamPoserPipeline(FakeDiffusionPipeline(), device="cpu").generate(
            Image.new("RGB", (384, 512)),
            "studio",
            1,
            lora_strength=0.8,
        )


def test_generate_rejects_wrong_pose_dimensions() -> None:
    with pytest.raises(GenerationError, match="384x512"):
        GundamPoserPipeline(FakeDiffusionPipeline(), device="cpu").generate(
            Image.new("RGB", (512, 512)),
            "neutral studio",
            42,
        )


@pytest.mark.parametrize("seed", [-1, MAX_SEED + 1, 1.5, True])
def test_generate_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(GenerationError, match="seed"):
        GundamPoserPipeline(FakeDiffusionPipeline(), device="cpu").generate(
            Image.new("RGB", (384, 512)),
            "neutral studio",
            seed,  # type: ignore[arg-type]
        )


def test_generate_rejects_multiple_images() -> None:
    fake = FakeDiffusionPipeline(
        [
            Image.new("RGB", (384, 512)),
            Image.new("RGB", (384, 512)),
        ]
    )
    with pytest.raises(GenerationError, match="exactly one"):
        GundamPoserPipeline(fake, device="cpu").generate(
            Image.new("RGB", (384, 512)),
            "neutral studio",
            42,
        )


def test_generate_reports_safety_checker_block() -> None:
    fake = FakeDiffusionPipeline(safety_flags=[True])
    with pytest.raises(GenerationError, match="safety checker"):
        GundamPoserPipeline(fake, device="cpu").generate(
            Image.new("RGB", (384, 512)),
            "neutral studio",
            42,
        )


def test_load_wires_public_models_scheduler_and_cpu() -> None:
    controlnet = object()
    loaded_pipeline = MagicMock()
    loaded_pipeline.scheduler.config = {"name": "original"}
    scheduler = object()

    with (
        patch(
            "diffusers.ControlNetModel.from_pretrained",
            return_value=controlnet,
        ) as load_controlnet,
        patch(
            "diffusers.StableDiffusionControlNetPipeline.from_pretrained",
            return_value=loaded_pipeline,
        ) as load_pipeline,
        patch(
            "diffusers.DPMSolverMultistepScheduler.from_config",
            return_value=scheduler,
        ) as load_scheduler,
    ):
        result = GundamPoserPipeline.load(device="cpu")

    load_controlnet.assert_called_once_with(
        CONTROLNET_MODEL_ID,
        torch_dtype=torch.float32,
        use_safetensors=True,
        cache_dir=None,
        token=False,
    )
    load_pipeline.assert_called_once_with(
        BASE_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=torch.float32,
        use_safetensors=True,
        cache_dir=None,
        token=False,
    )
    load_scheduler.assert_called_once_with(
        {"name": "original"},
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    assert loaded_pipeline.scheduler is scheduler
    loaded_pipeline.to.assert_called_once_with("cpu")
    assert result.device == "cpu"


def test_load_uses_float32_and_attention_slicing_on_mps() -> None:
    controlnet = object()
    loaded_pipeline = MagicMock()
    loaded_pipeline.scheduler.config = {}

    with (
        patch("torch.backends.mps.is_available", return_value=True),
        patch(
            "diffusers.ControlNetModel.from_pretrained",
            return_value=controlnet,
        ) as load_controlnet,
        patch(
            "diffusers.StableDiffusionControlNetPipeline.from_pretrained",
            return_value=loaded_pipeline,
        ) as load_pipeline,
        patch(
            "diffusers.DPMSolverMultistepScheduler.from_config",
            return_value=object(),
        ),
    ):
        result = GundamPoserPipeline.load(device="mps")

    assert load_controlnet.call_args.kwargs["torch_dtype"] == torch.float32
    assert load_pipeline.call_args.kwargs["torch_dtype"] == torch.float32
    loaded_pipeline.enable_attention_slicing.assert_called_once_with()
    loaded_pipeline.to.assert_called_once_with("mps")
    assert result.device == "mps"


def test_load_adds_named_local_lora_adapter(tmp_path: Path) -> None:
    path = tmp_path / "adapter.safetensors"
    path.write_bytes(b"adapter")
    loaded_pipeline = MagicMock()
    loaded_pipeline.scheduler.config = {}
    with (
        patch("diffusers.ControlNetModel.from_pretrained", return_value=object()),
        patch(
            "diffusers.StableDiffusionControlNetPipeline.from_pretrained",
            return_value=loaded_pipeline,
        ),
        patch(
            "diffusers.DPMSolverMultistepScheduler.from_config",
            return_value=object(),
        ),
    ):
        result = GundamPoserPipeline.load(device="cpu", lora_path=path)

    loaded_pipeline.load_lora_weights.assert_called_once_with(
        str(path.resolve()),
        adapter_name=LORA_ADAPTER_NAME,
    )
    assert result.lora_loaded is True


def test_load_adds_named_private_hub_lora_adapter() -> None:
    loaded_pipeline = MagicMock()
    loaded_pipeline.scheduler.config = {}
    with (
        patch("diffusers.ControlNetModel.from_pretrained", return_value=object()),
        patch(
            "diffusers.StableDiffusionControlNetPipeline.from_pretrained",
            return_value=loaded_pipeline,
        ),
        patch(
            "diffusers.DPMSolverMultistepScheduler.from_config",
            return_value=object(),
        ),
    ):
        result = GundamPoserPipeline.load(
            device="cpu",
            lora_repo_id=DEFAULT_LORA_REPO_ID,
            lora_token=True,
        )

    loaded_pipeline.load_lora_weights.assert_called_once_with(
        DEFAULT_LORA_REPO_ID,
        weight_name=DEFAULT_LORA_WEIGHT_NAME,
        adapter_name=LORA_ADAPTER_NAME,
        token=True,
    )
    assert result.lora_loaded is True


def test_load_rejects_two_lora_sources(tmp_path: Path) -> None:
    with pytest.raises(GenerationError, match="either a local LoRA"):
        GundamPoserPipeline.load(
            device="cpu",
            lora_path=tmp_path / "adapter.safetensors",
            lora_repo_id=DEFAULT_LORA_REPO_ID,
        )


def test_resolve_device_rejects_unavailable_cuda() -> None:
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(GenerationError, match="CUDA"):
            resolve_device("cuda")


def test_resolve_device_prefers_mps_when_cuda_is_unavailable() -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        assert resolve_device() == "mps"


def test_resolve_device_rejects_unknown_device() -> None:
    with pytest.raises(GenerationError, match="cuda, mps, cpu"):
        resolve_device("tpu")
