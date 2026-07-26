from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest
from unittest.mock import patch

from app import (
    APP_CSS,
    POSE_EXAMPLES,
    build_app,
    create_baseline_generation,
    create_pose_guided_generation,
    create_pose_preview,
    full_prompt_for_scene,
    get_generation_pipeline,
    load_pose_example,
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
        if component["props"]["label"]
        == "Upload or take a one-person full-body photo"
    )
    assert source["props"]["sources"] == ["upload", "webcam"]
    assert {component["props"]["label"] for component in image_components} == {
        "Upload or take a one-person full-body photo",
        "Pose overlay",
        "Pose map",
        "Baseline (LoRA off)",
        "Trained LoRA (0.8)",
    }


@pytest.mark.parametrize("selection", list(POSE_EXAMPLES))
def test_load_pose_example_returns_an_rgb_photo(selection: str) -> None:
    image = load_pose_example(selection)

    assert image is not None
    assert image.mode == "RGB"
    assert image.width > 0
    assert image.height > 0


def test_load_pose_example_rejects_unknown_selection() -> None:
    with pytest.raises(ValueError, match="Unknown built-in pose"):
        load_pose_example("Unknown")


class FakeGenerator:
    def __init__(self) -> None:
        self.pose_image: Image.Image | None = None
        self.scene_prompt: str | None = None
        self.seed: int | None = None
        self.lora_strengths: list[float] = []
        self.prompt_overrides: list[str | None] = []

    def generate(
        self,
        pose_image: Image.Image,
        scene_prompt: str,
        seed: int,
        *,
        lora_strength: float = 0.0,
        prompt_override: str | None = None,
    ) -> GenerationResult:
        self.pose_image = pose_image
        self.scene_prompt = scene_prompt
        self.seed = seed
        self.lora_strengths.append(lora_strength)
        self.prompt_overrides.append(prompt_override)
        return GenerationResult(
            image=Image.new(
                "RGB",
                (384, 512),
                "green" if lora_strength else "blue",
            ),
            metadata=GenerationMetadata(
                seed=seed,
                prompt="final prompt",
                lora_strength=lora_strength,
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


def test_untouched_optional_description_uses_only_the_scene_preset() -> None:
    pose_image = Image.new("RGB", (384, 512), "black")
    generator = FakeGenerator()

    create_baseline_generation(
        pose_image,
        "Neutral studio",
        None,
        42,
        generator=generator,
    )

    assert generator.scene_prompt == "neutral studio background"


def test_full_prompt_override_is_used_for_both_comparison_images() -> None:
    source = Image.new("RGB", (100, 200), "white")
    generator = FakeGenerator()
    custom_prompt = "custom hwmecha warrior on the moon"

    create_pose_guided_generation(
        source,
        "Neutral studio",
        "",
        42,
        previewer=FakePreviewer(),
        generator=generator,
        prompt_override=custom_prompt,
    )

    assert generator.prompt_overrides == [custom_prompt, custom_prompt]


def test_scene_preset_builds_an_editable_full_prompt() -> None:
    prompt = full_prompt_for_scene("Industrial hangar")

    assert "hwmecha" in prompt
    assert "industrial hangar" in prompt


def test_single_action_detects_pose_then_generates_from_pose_map() -> None:
    source = Image.new("RGB", (100, 200), "white")
    previewer = FakePreviewer()
    generator = FakeGenerator()

    overlay, pose, baseline, trained, status = create_pose_guided_generation(
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
    assert baseline.size == (384, 512)
    assert trained.size == (384, 512)
    assert generator.lora_strengths == [0.0, 0.8]
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
        "Generate"
    ]


def test_generated_results_precede_collapsed_pose_details() -> None:
    components = build_app().get_config_file()["components"]
    labels = [component.get("props", {}).get("label") for component in components]

    assert labels.index("Baseline (LoRA off)") < labels.index("Pose overlay")
    accordions = {
        component["props"]["label"]: component["props"]
        for component in components
        if component["type"] == "accordion"
    }
    assert accordions["Generation options"]["open"] is False
    assert accordions["Extracted pose details"]["open"] is False


def test_mobile_layout_stacks_responsive_rows() -> None:
    assert "@media (max-width: 700px)" in APP_CSS
    assert ".responsive-row { flex-direction: column !important" in APP_CSS
    assert "min-width: 100% !important" in APP_CSS


def test_app_lists_all_bundled_pose_examples() -> None:
    components = build_app().get_config_file()["components"]
    dropdown = next(
        component
        for component in components
        if component.get("props", {}).get("label") == "Built-in pose (optional)"
    )

    assert [choice[0] for choice in dropdown["props"]["choices"]] == list(
        POSE_EXAMPLES
    )


def test_generation_options_expose_the_full_prompt() -> None:
    components = build_app().get_config_file()["components"]
    prompt = next(
        component
        for component in components
        if component.get("props", {}).get("label") == "Full prompt"
    )

    assert "hwmecha" in prompt["props"]["value"]
    assert prompt["props"]["lines"] == 5


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

    assert image_components["Baseline (LoRA off)"]["format"] == "png"
    assert image_components["Trained LoRA (0.8)"]["format"] == "png"
    assert image_components["Pose map"]["format"] == "png"


def test_generation_pipeline_honors_device_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GUNDAMPOSER_DEVICE", "mps")
    get_generation_pipeline.cache_clear()
    expected = object()

    with patch(
        "app.GundamPoserPipeline.load",
        return_value=expected,
    ) as load:
        result = get_generation_pipeline()

    load.assert_called_once_with(device="mps", lora_path=None)
    assert result is expected
    get_generation_pipeline.cache_clear()


def test_generation_pipeline_auto_loads_exported_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GUNDAMPOSER_LORA_PATH", raising=False)
    adapter = Path("outputs/gundamposer_lora.safetensors")
    adapter.parent.mkdir()
    adapter.write_bytes(b"selected adapter")
    get_generation_pipeline.cache_clear()

    with patch("app.GundamPoserPipeline.load", return_value=object()) as load:
        get_generation_pipeline()

    load.assert_called_once_with(device=None, lora_path=adapter)
    get_generation_pipeline.cache_clear()
