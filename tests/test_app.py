from __future__ import annotations

from PIL import Image
import pytest

from app import build_app, create_pose_preview
from gundamposer.pose import PosePreviewResult


class FakePreviewer:
    def __init__(self) -> None:
        self.received: Image.Image | None = None
        self.opacity: float | None = None

    def preview(self, image: Image.Image, *, opacity: float) -> PosePreviewResult:
        self.received = image
        self.opacity = opacity
        return PosePreviewResult(
            overlay_image=Image.new("RGB", (100, 200), "red"),
            pose_image=Image.new("RGB", (100, 200), "black"),
            person_count=1,
            detected_body_keypoints=14,
            detector_input_size=(100, 200),
        )


def test_create_pose_preview_returns_images_and_status() -> None:
    source = Image.new("RGB", (100, 200), "white")
    previewer = FakePreviewer()

    overlay, pose, status = create_pose_preview(
        source,
        previewer=previewer,
        opacity=0.6,
    )

    assert previewer.received is source
    assert previewer.opacity == 0.6
    assert overlay.size == (100, 200)
    assert pose.size == (100, 200)
    assert status == "Detected 14 body keypoints. Preview size: 100×200."


def test_create_pose_preview_requires_an_image() -> None:
    with pytest.raises(ValueError, match="Upload a photo"):
        create_pose_preview(None, previewer=FakePreviewer())


def test_app_exposes_upload_and_webcam_sources() -> None:
    config = build_app().get_config_file()
    image_components = [
        component
        for component in config["components"]
        if component["type"] == "image"
    ]

    assert image_components[0]["props"]["sources"] == ["upload", "webcam"]
    assert {component["props"]["label"] for component in image_components} == {
        "One-person photo",
        "Pose overlay",
        "Detected pose map",
    }

