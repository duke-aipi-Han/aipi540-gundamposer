"""Interactive pose preview and baseline image generation."""

from __future__ import annotations

from functools import lru_cache
import os
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
    MAX_SEED,
    resolve_device,
)
from gundamposer.prompts import SCENE_PRESETS, build_scene_prompt


class PosePreviewer(Protocol):
    def preview(self, image: Image.Image, *, opacity: float) -> PosePreviewResult:
        """Return an aligned pose preview for one image."""


class BaselineGenerator(Protocol):
    def generate(
        self,
        pose_image: Image.Image,
        scene_prompt: str,
        seed: int,
    ) -> GenerationResult:
        """Generate one image from a pose map."""


@lru_cache(maxsize=1)
def get_pose_extractor() -> PoseExtractor:
    """Load and reuse the CPU body-pose model."""

    return PoseExtractor.from_pretrained()


@lru_cache(maxsize=1)
def get_generation_pipeline() -> GundamPoserPipeline:
    """Load and reuse the baseline diffusion pipeline."""

    requested_device = os.getenv("GUNDAMPOSER_DEVICE") or None
    return GundamPoserPipeline.load(device=requested_device)


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
    scene_description: str,
    seed: int | float,
    *,
    generator: BaselineGenerator | None = None,
) -> tuple[Image.Image, str]:
    """Generate one baseline image using only the extracted pose map."""

    if pose_image is None:
        raise ValueError("Detect a pose before generating an image.")
    if isinstance(seed, bool) or not isinstance(seed, (int, float)):
        raise ValueError("Seed must be a non-negative integer.")
    normalized_seed = int(seed)
    if normalized_seed != seed or not 0 <= normalized_seed <= MAX_SEED:
        raise ValueError(f"Seed must be an integer from 0 to {MAX_SEED}.")

    scene_prompt = build_scene_prompt(scene_preset, scene_description)
    active_generator = generator or get_generation_pipeline()
    result = active_generator.generate(
        pose_image,
        scene_prompt,
        normalized_seed,
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
    scene_description: str,
    seed: int | float,
    *,
    previewer: PosePreviewer | None = None,
    generator: BaselineGenerator | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image, str]:
    """Detect one pose and generate from its control map in a single action."""

    overlay, pose_image, pose_status = create_pose_preview(
        image,
        previewer=previewer,
    )
    generated, generation_status = create_baseline_generation(
        pose_image,
        scene_preset,
        scene_description,
        seed,
        generator=generator,
    )
    return (
        overlay,
        pose_image,
        generated,
        f"{pose_status}\n\n{generation_status}",
    )


def _handle_pose_guided_generation(
    image: Image.Image | None,
    scene_preset: str,
    scene_description: str,
    seed: int | float,
) -> tuple[Image.Image, Image.Image, Image.Image, str]:
    try:
        return create_pose_guided_generation(
            image,
            scene_preset,
            scene_description,
            seed,
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
    with gr.Blocks(title="GundamPoser Baseline") as demo:
        gr.Markdown(
            "# Pose-Guided Baseline Generation\n"
            "Choose a full-body photo of one person, adjust the scene if desired, "
            "then generate the pose preview and image in one action."
        )
        gr.Markdown(f"Generation backend: **{generation_device.upper()}**")
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 1. Choose a photo")
                source_image = gr.Image(
                    label="One-person full-body photo",
                    sources=["upload", "webcam"],
                    type="pil",
                    image_mode="RGB",
                    height=512,
                )
            with gr.Column(scale=1):
                gr.Markdown("## 2. Set the scene")
                scene_preset = gr.Dropdown(
                    choices=list(SCENE_PRESETS),
                    value="Neutral studio",
                    label="Scene preset",
                )
                scene_description = gr.Textbox(
                    label="Optional scene details",
                    placeholder="for example: dramatic lighting, blue armor",
                    lines=3,
                )
                seed = gr.Number(
                    value=42,
                    precision=0,
                    minimum=0,
                    maximum=MAX_SEED,
                    label="Seed",
                )
                generate_button = gr.Button(
                    "Generate pose-guided image",
                    variant="primary",
                )
        gr.Markdown("## 3. Review the result")
        with gr.Row(equal_height=True):
            overlay_image = gr.Image(
                label="Pose overlay",
                type="pil",
                interactive=False,
                height=512,
            )
            pose_image = gr.Image(
                label="Pose map",
                type="pil",
                format="png",
                interactive=False,
                height=512,
            )
            generated_image = gr.Image(
                label="Generated image",
                type="pil",
                format="png",
                interactive=False,
                height=512,
            )
        generation_status = gr.Markdown()
        generate_button.click(
            fn=_handle_pose_guided_generation,
            inputs=[source_image, scene_preset, scene_description, seed],
            outputs=[overlay_image, pose_image, generated_image, generation_status],
            concurrency_limit=1,
        )
        gr.Markdown(
            "The photo is processed in memory for this preview and is not "
            "intentionally retained. Generation receives only the extracted pose map, "
            "not the source photo. Only upload images you have permission to use. "
            "The first generation downloads the public diffusion models and CPU "
            "generation can be slow."
        )
    return demo


demo = build_app()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
