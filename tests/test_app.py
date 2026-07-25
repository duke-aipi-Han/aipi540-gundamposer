from __future__ import annotations

from PIL import Image
import pytest
from unittest.mock import patch

from app import (
    build_app,
    create_baseline_generation,
    create_pose_guided_generation,
    create_pose_preview,
    get_generation_pipeline,
)
from gundamposer.pipeline import (
    GenerationMetadata,
    GenerationResult,
)
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
            pose_image=Image.new("RGB", (384, 512), "black"),
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
    assert pose.size == (384, 512)
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

    source = next(
        component
        for component in image_components
        if component["props"]["label"] == "One-person full-body photo"
    )
    assert source["props"]["sources"] == ["upload", "webcam"]
    assert {component["props"]["label"] for component in image_components} == {
        "One-person full-body photo",
        "Pose overlay",
        "Generated image",
        "Pose map",
    }


class FakeGenerator:
    def __init__(self) -> None:
        self.pose_image: Image.Image | None = None
        self.scene_prompt: str | None = None
        self.seed: int | None = None

    def generate(
        self,
        pose_image: Image.Image,
        scene_prompt: str,
        seed: int,
    ) -> GenerationResult:
        self.pose_image = pose_image
        self.scene_prompt = scene_prompt
        self.seed = seed
        return GenerationResult(
            image=Image.new("RGB", (384, 512), "blue"),
            metadata=GenerationMetadata(
                seed=seed,
                prompt="final prompt",
                lora_strength=0.0,
                controlnet_strength=0.8,
                generation_time_seconds=2.25,
                device="cpu",
            ),
        )


def test_create_baseline_generation_uses_only_pose_and_controls() -> None:
    pose_image = Image.new("RGB", (384, 512), "black")
    generator = FakeGenerator()

    image, status = create_baseline_generation(
        pose_image,
        "Industrial hangar",
        "dramatic lighting",
        123,
        generator=generator,
    )

    assert generator.pose_image is pose_image
    assert generator.scene_prompt == "industrial hangar, dramatic lighting"
    assert generator.seed == 123
    assert image.size == (384, 512)
    assert "Seed: `123`" in status
    assert "ControlNet: `0.80`" in status
    assert "LoRA: `0.00`" in status
    assert "Prompt: final prompt" in status


def test_single_action_detects_pose_then_generates_from_pose_map() -> None:
    source = Image.new("RGB", (100, 200), "white")
    previewer = FakePreviewer()
    generator = FakeGenerator()

    overlay, pose, generated, status = create_pose_guided_generation(
        source,
        "Industrial hangar",
        "dramatic lighting",
        123,
        previewer=previewer,
        generator=generator,
    )

    assert previewer.received is source
    assert generator.pose_image is pose
    assert overlay.size == (100, 200)
    assert pose.size == (384, 512)
    assert generated.size == (384, 512)
    assert "Detected 14 body keypoints" in status
    assert "Seed: `123`" in status


def test_app_has_one_generation_action() -> None:
    config = build_app().get_config_file()
    buttons = [
        component
        for component in config["components"]
        if component["type"] == "button"
    ]

    assert [button["props"]["value"] for button in buttons] == [
        "Generate pose-guided image"
    ]


def test_create_baseline_generation_requires_pose() -> None:
    with pytest.raises(ValueError, match="Detect a pose"):
        create_baseline_generation(
            None,
            "Neutral studio",
            "",
            42,
            generator=FakeGenerator(),
        )


@pytest.mark.parametrize("seed", [-1, 2**31, 1.5, True, "42"])
def test_create_baseline_generation_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="Seed"):
        create_baseline_generation(
            Image.new("RGB", (384, 512)),
            "Neutral studio",
            "",
            seed,  # type: ignore[arg-type]
            generator=FakeGenerator(),
        )


def test_generation_pose_and_output_use_lossless_png() -> None:
    config = build_app().get_config_file()
    image_components = {
        component["props"]["label"]: component["props"]
        for component in config["components"]
        if component["type"] == "image"
    }

    assert image_components["Generated image"]["format"] == "png"
    assert image_components["Pose map"]["format"] == "png"


def test_generation_pipeline_honors_device_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUNDAMPOSER_DEVICE", "mps")
    get_generation_pipeline.cache_clear()
    expected = object()

    with patch(
        "app.GundamPoserPipeline.load",
        return_value=expected,
    ) as load:
        result = get_generation_pipeline()

    load.assert_called_once_with(device="mps")
    assert result is expected
    get_generation_pipeline.cache_clear()
