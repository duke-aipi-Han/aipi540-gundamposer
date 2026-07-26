"""Interactive pose extraction and baseline-versus-LoRA comparison."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gradio as gr
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image

from gundamposer.pose import (
    PoseExtractionError,
    PoseExtractor,
    PosePreviewResult,
)
from gundamposer.pipeline import (
    GenerationError,
    GenerationResult,
    GundamPoserPipeline,
    DEFAULT_LORA_STRENGTH,
    MAX_SEED,
    resolve_device,
)
from gundamposer.prompts import SCENE_PRESETS, build_prompt, build_scene_prompt


POSE_EXAMPLE_ROOT = Path(__file__).resolve().parent / "assets" / "pose-examples"
POSE_EXAMPLES = MappingProxyType(
    {
        "Running stride": POSE_EXAMPLE_ROOT / "running.jpg",
        "Action balance": POSE_EXAMPLE_ROOT / "action-balance.jpg",
        "Celebration": POSE_EXAMPLE_ROOT / "celebration.jpg",
        "Martial arts stance": POSE_EXAMPLE_ROOT / "martial-arts.jpg",
    }
)

APP_CSS = """
.app-shell { max-width: 1120px; margin: 0 auto; }
.generate-button { min-height: 3.25rem; font-size: 1.05rem; }
@media (max-width: 700px) {
  .gradio-container { padding-left: 0.65rem !important; padding-right: 0.65rem !important; }
  .responsive-row { flex-direction: column !important; gap: 0.75rem !important; }
  .responsive-row > div { width: 100% !important; min-width: 100% !important; }
  .responsive-image { min-height: 360px !important; }
}
"""


class PosePreviewer(Protocol):
    def preview(self, image: Image.Image, *, opacity: float) -> PosePreviewResult:
        """Return an aligned pose preview for one image."""


class BaselineGenerator(Protocol):
    def generate(
        self,
        pose_image: Image.Image,
        scene_prompt: str,
        seed: int,
        *,
        lora_strength: float = 0.0,
        prompt_override: str | None = None,
    ) -> GenerationResult:
        """Generate one image from a pose map."""


@lru_cache(maxsize=1)
def get_pose_extractor() -> PoseExtractor:
    """Load and reuse the CPU body-pose model."""

    return PoseExtractor.from_pretrained()


@lru_cache(maxsize=1)
def get_generation_pipeline() -> GundamPoserPipeline:
    """Load and reuse one diffusion pipeline with an optional trained adapter."""

    requested_device = os.getenv("GUNDAMPOSER_DEVICE") or None
    configured_lora = os.getenv("GUNDAMPOSER_LORA_PATH")
    default_lora = Path("outputs/gundamposer_lora.safetensors")
    lora_path: str | Path | None = configured_lora
    if lora_path is None and default_lora.is_file():
        lora_path = default_lora
    load_options: dict[str, object] = {
        "device": requested_device,
        "lora_path": lora_path,
    }
    configured_base_model = os.getenv("GUNDAMPOSER_BASE_MODEL")
    configured_controlnet = os.getenv("GUNDAMPOSER_CONTROLNET_MODEL")
    if configured_base_model:
        load_options["base_model_id"] = configured_base_model
    if configured_controlnet:
        load_options["controlnet_model_id"] = configured_controlnet
    return GundamPoserPipeline.load(
        **load_options,
    )


def load_pose_example(selection: str | None) -> Image.Image | None:
    """Load a bundled pose example into the same input used by uploads."""

    if selection is None:
        return None
    path = POSE_EXAMPLES.get(selection)
    if path is None:
        raise ValueError(f"Unknown built-in pose: {selection}")
    if not path.is_file():
        raise ValueError(f"Built-in pose is unavailable: {selection}")
    with Image.open(path) as image:
        return image.convert("RGB")


def full_prompt_for_scene(preset: str) -> str:
    """Create the editable default prompt for a scene preset."""

    return build_prompt(build_scene_prompt(preset))


def create_pose_preview(
    image: Image.Image | None,
    *,
    previewer: PosePreviewer | None = None,
    opacity: float = 0.75,
) -> tuple[Image.Image, Image.Image, str]:
    """Create an overlay, an aligned pose map, and a short status message."""

    if image is None:
        raise ValueError("Upload a photo or take one with the camera first.")

    active_previewer = previewer or get_pose_extractor()
    result = active_previewer.preview(image, opacity=opacity)
    status = (
        f"Detected {result.detected_body_keypoints} body keypoints. "
        f"Preview size: {result.detector_input_size[0]}×"
        f"{result.detector_input_size[1]}."
    )
    return result.overlay_image, result.pose_image, status


def create_baseline_generation(
    pose_image: Image.Image | None,
    scene_preset: str,
    scene_description: str | None,
    seed: int | float,
    *,
    generator: BaselineGenerator | None = None,
    lora_strength: float = 0.0,
    prompt_override: str | None = None,
) -> tuple[Image.Image, str]:
    """Generate one baseline image using only the extracted pose map."""

    if pose_image is None:
        raise ValueError("Detect a pose before generating an image.")
    if isinstance(seed, bool) or not isinstance(seed, (int, float)):
        raise ValueError("Seed must be a non-negative integer.")
    normalized_seed = int(seed)
    if normalized_seed != seed or not 0 <= normalized_seed <= MAX_SEED:
        raise ValueError(f"Seed must be an integer from 0 to {MAX_SEED}.")

    normalized_description = "" if scene_description is None else scene_description
    scene_prompt = build_scene_prompt(scene_preset, normalized_description)
    active_generator = generator or get_generation_pipeline()
    result = active_generator.generate(
        pose_image,
        scene_prompt,
        normalized_seed,
        lora_strength=lora_strength,
        prompt_override=prompt_override,
    )
    metadata = result.metadata
    status = (
        f"Seed: `{metadata.seed}` · Device: `{metadata.device}` · "
        f"ControlNet: `{metadata.controlnet_strength:.2f}` · "
        f"LoRA: `{metadata.lora_strength:.2f}` · "
        f"Time: `{metadata.generation_time_seconds:.1f}s`\n\n"
        f"Prompt: {metadata.prompt}"
    )
    return result.image, status


def create_pose_guided_generation(
    image: Image.Image | None,
    scene_preset: str,
    scene_description: str | None,
    seed: int | float,
    *,
    previewer: PosePreviewer | None = None,
    generator: BaselineGenerator | None = None,
    prompt_override: str | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, str]:
    """Detect one pose and generate matched baseline and trained outputs."""

    overlay, pose_image, pose_status = create_pose_preview(
        image,
        previewer=previewer,
    )
    baseline, baseline_status = create_baseline_generation(
        pose_image,
        scene_preset,
        scene_description,
        seed,
        generator=generator,
        prompt_override=prompt_override,
    )
    trained, trained_status = create_baseline_generation(
        pose_image,
        scene_preset,
        scene_description,
        seed,
        generator=generator,
        lora_strength=DEFAULT_LORA_STRENGTH,
        prompt_override=prompt_override,
    )
    return (
        overlay,
        pose_image,
        baseline,
        trained,
        f"{pose_status}\n\n**Baseline**\n\n{baseline_status}\n\n"
        f"**Trained LoRA**\n\n{trained_status}",
    )


def _handle_pose_guided_generation(
    image: Image.Image | None,
    scene_preset: str,
    prompt_override: str | None,
    seed: int | float,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, str]:
    try:
        return create_pose_guided_generation(
            image,
            scene_preset,
            "",
            seed,
            prompt_override=prompt_override,
        )
    except HfHubHTTPError as error:
        raise gr.Error(
            "Could not download a public model. Check the network connection and "
            "try again."
        ) from error
    except (GenerationError, PoseExtractionError, TypeError, ValueError) as error:
        raise gr.Error(str(error)) from error
    except RuntimeError as error:
        raise gr.Error(f"Generation failed: {error}") from error


def build_app() -> gr.Blocks:
    requested_device = os.getenv("GUNDAMPOSER_DEVICE") or None
    generation_device = resolve_device(requested_device)
    with gr.Blocks(
        title="GundamPoser Comparison",
        elem_classes=["app-shell"],
    ) as demo:
        gr.HTML(f"<style>{APP_CSS}</style>")
        gr.Markdown(
            "# Pose-Guided Baseline vs Trained LoRA\n"
            "Choose a built-in pose or provide a full-body photo of one person, "
            "then compare the baseline with the trained result."
        )
        gr.Markdown(f"Generation backend: **{generation_device.upper()}**")
        configured_lora = os.getenv("GUNDAMPOSER_LORA_PATH")
        default_lora = Path("outputs/gundamposer_lora.safetensors")
        displayed_lora = Path(configured_lora) if configured_lora else default_lora
        adapter_status = "available" if displayed_lora.is_file() else "not found"
        gr.Markdown(
            f"Trained adapter: `{displayed_lora}` (**{adapter_status}**)"
        )
        gr.Markdown("## Choose a pose")
        pose_example = gr.Dropdown(
            choices=list(POSE_EXAMPLES),
            value=None,
            label="Built-in pose (optional)",
            info="Selecting an example fills the photo input below.",
        )
        source_image = gr.Image(
            label="Upload or take a one-person full-body photo",
            sources=["upload", "webcam"],
            type="pil",
            image_mode="RGB",
            height=480,
            elem_classes=["responsive-image"],
        )
        pose_example.change(
            fn=load_pose_example,
            inputs=pose_example,
            outputs=source_image,
            show_progress="hidden",
        )
        with gr.Accordion("Generation options", open=False):
            scene_preset = gr.Dropdown(
                choices=list(SCENE_PRESETS),
                value="Neutral studio",
                label="Scene preset",
            )
            full_prompt = gr.Textbox(
                value=full_prompt_for_scene("Neutral studio"),
                label="Full prompt",
                info="Edit freely. Keep 'hwmecha' for the trained adapter style.",
                lines=5,
            )
            scene_preset.change(
                fn=full_prompt_for_scene,
                inputs=scene_preset,
                outputs=full_prompt,
                show_progress="hidden",
            )
            seed = gr.Number(
                value=42,
                precision=0,
                minimum=0,
                maximum=MAX_SEED,
                label="Seed",
            )
        generate_button = gr.Button(
            "Generate",
            variant="primary",
            elem_classes=["generate-button"],
        )
        gr.Markdown("## Generated results")
        with gr.Row(equal_height=True, elem_classes=["responsive-row"]):
            baseline_image = gr.Image(
                label="Baseline (LoRA off)",
                type="pil",
                format="png",
                interactive=False,
                height=480,
                elem_classes=["responsive-image"],
            )
            trained_image = gr.Image(
                label=f"Trained LoRA ({DEFAULT_LORA_STRENGTH:.1f})",
                type="pil",
                format="png",
                interactive=False,
                height=480,
                elem_classes=["responsive-image"],
            )
        generation_status = gr.Markdown()
        with gr.Accordion("Extracted pose details", open=False):
            with gr.Row(equal_height=True, elem_classes=["responsive-row"]):
                overlay_image = gr.Image(
                    label="Pose overlay",
                    type="pil",
                    interactive=False,
                    height=480,
                    elem_classes=["responsive-image"],
                )
                pose_image = gr.Image(
                    label="Pose map",
                    type="pil",
                    format="png",
                    interactive=False,
                    height=480,
                    elem_classes=["responsive-image"],
                )
        generate_button.click(
            fn=_handle_pose_guided_generation,
            inputs=[source_image, scene_preset, full_prompt, seed],
            outputs=[
                overlay_image,
                pose_image,
                baseline_image,
                trained_image,
                generation_status,
            ],
            concurrency_limit=1,
        )
        gr.Markdown(
            "The photo is processed in memory for this preview and is not "
            "intentionally retained. Generation receives only the extracted pose map, "
            "not the source photo. Only upload images you have permission to use. "
            "The first generation downloads the public diffusion models when they "
            "are not cached. Baseline and trained images use the same pose, prompt, "
            "seed, and pipeline; only the LoRA adapter state differs."
        )
        with gr.Accordion("Built-in photo credits", open=False):
            gr.Markdown(
                "The bundled examples are public-domain or CC0 images from "
                "Wikimedia Commons. Full source and license details are included "
                "with the assets."
            )
    return demo


demo = build_app()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
