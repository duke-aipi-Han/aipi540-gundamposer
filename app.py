"""Interactive body-pose preview."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import gradio as gr
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image

from gundamposer.pose import (
    PoseExtractionError,
    PoseExtractor,
    PosePreviewResult,
)


class PosePreviewer(Protocol):
    def preview(self, image: Image.Image, *, opacity: float) -> PosePreviewResult:
        """Return an aligned pose preview for one image."""


@lru_cache(maxsize=1)
def get_pose_extractor() -> PoseExtractor:
    """Load and reuse the CPU body-pose model."""

    return PoseExtractor.from_pretrained()


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


def _handle_preview(
    image: Image.Image | None,
) -> tuple[Image.Image, Image.Image, str]:
    try:
        return create_pose_preview(image)
    except HfHubHTTPError as error:
        raise gr.Error(
            "Could not download the public pose model. Check the network connection "
            "and try again."
        ) from error
    except (PoseExtractionError, TypeError, ValueError) as error:
        raise gr.Error(str(error)) from error


def build_app() -> gr.Blocks:
    with gr.Blocks(title="GundamPoser Pose Preview") as demo:
        gr.Markdown(
            "# Body Pose Preview\n"
            "Upload a one-person photo or use your camera to preview the detected "
            "body skeleton. Full-body photos work best."
        )
        with gr.Row():
            source_image = gr.Image(
                label="One-person photo",
                sources=["upload", "webcam"],
                type="pil",
                image_mode="RGB",
            )
            overlay_image = gr.Image(
                label="Pose overlay",
                type="pil",
                interactive=False,
            )
        pose_image = gr.Image(
            label="Detected pose map",
            type="pil",
            interactive=False,
            height=512,
        )
        status = gr.Markdown()
        preview_button = gr.Button("Detect pose", variant="primary")
        preview_button.click(
            fn=_handle_preview,
            inputs=source_image,
            outputs=[overlay_image, pose_image, status],
            concurrency_limit=1,
        )
        gr.Markdown(
            "The photo is processed in memory for this preview and is not "
            "intentionally retained. Only upload images you have permission to use."
        )
    return demo


demo = build_app()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
