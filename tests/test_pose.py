from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from gundamposer.pose import (
    BodyKeypoint,
    BodyPose,
    ControlNetAuxOpenPoseAdapter,
    InvalidPoseError,
    MultiplePeopleDetectedError,
    NoPersonDetectedError,
    PoseExtractionResult,
    PoseExtractor,
    normalize_body_pose,
    overlay_pose_map,
)


def make_pose(
    *points: tuple[int, float, float],
    keypoint_count: int = 18,
) -> BodyPose:
    keypoints: list[BodyKeypoint | None] = [None] * keypoint_count
    for index, x, y in points:
        keypoints[index] = BodyKeypoint(x=x, y=y, id=index)
    return BodyPose(
        keypoints=tuple(keypoints),
        total_score=float(len(points)),
        total_parts=len(points),
    )


class FakeDetector:
    def __init__(self, poses: list[BodyPose]) -> None:
        self.poses = poses
        self.received: np.ndarray | None = None

    def detect(self, image: np.ndarray) -> list[BodyPose]:
        self.received = image.copy()
        return self.poses


def test_extractor_returns_centered_384_by_512_pose_map() -> None:
    detector = FakeDetector(
        [make_pose((0, 0.5, 0.1), (1, 0.5, 0.3), (2, 0.2, 0.4), (5, 0.8, 0.4))]
    )
    extractor = PoseExtractor(detector)

    result = extractor.extract(Image.new("RGB", (400, 800), "white"))

    assert result.pose_image.mode == "RGB"
    assert result.pose_image.size == (384, 512)
    assert result.person_count == 1
    assert result.detected_body_keypoints == 4
    assert result.detector_input_size == (384, 768)
    assert np.asarray(result.pose_image).any()


def test_detector_input_is_oriented_rgb_and_limited_to_768() -> None:
    detector = FakeDetector([make_pose((0, 0.5, 0.5))])
    source = Image.new("RGBA", (1600, 800), (1, 2, 3, 128))

    PoseExtractor(detector).extract(source)

    assert detector.received is not None
    assert detector.received.shape == (384, 768, 3)
    assert detector.received.dtype == np.uint8


def test_exif_orientation_is_applied_before_detection() -> None:
    detector = FakeDetector([make_pose((0, 0.5, 0.5))])
    source = Image.new("RGB", (200, 400), "white")
    source.getexif()[274] = 6

    result = PoseExtractor(detector).extract(source)

    assert detector.received is not None
    assert detector.received.shape == (200, 400, 3)
    assert result.detector_input_size == (400, 200)


def test_no_person_is_rejected() -> None:
    with pytest.raises(NoPersonDetectedError, match="No person"):
        PoseExtractor(FakeDetector([])).extract(Image.new("RGB", (100, 100)))


def test_multiple_people_are_rejected() -> None:
    pose = make_pose((0, 0.5, 0.5))
    with pytest.raises(MultiplePeopleDetectedError, match="Multiple people"):
        PoseExtractor(FakeDetector([pose, pose])).extract(
            Image.new("RGB", (100, 100))
        )


def test_detected_pose_without_usable_body_points_is_rejected() -> None:
    empty_pose = BodyPose(keypoints=(None,) * 18)
    with pytest.raises(InvalidPoseError, match="no body keypoints"):
        PoseExtractor(FakeDetector([empty_pose])).extract(
            Image.new("RGB", (100, 100))
        )


def test_sparse_single_keypoint_is_centered() -> None:
    result = PoseExtractor(FakeDetector([make_pose((0, 0.1, 0.9))])).extract(
        Image.new("RGB", (300, 600))
    )
    array = np.asarray(result.pose_image)
    rows, columns = np.nonzero(array.any(axis=2))

    assert columns.mean() == pytest.approx(192, abs=1)
    assert rows.mean() == pytest.approx(256, abs=1)


def test_pose_normalization_preserves_pixel_space_aspect_ratio() -> None:
    pose = make_pose((0, 0.25, 0.25), (1, 0.75, 0.75))

    normalized = normalize_body_pose(pose, source_size=(400, 800))
    first = normalized.keypoints[0]
    second = normalized.keypoints[1]
    assert first is not None
    assert second is not None

    delta_x = abs(second.x - first.x) * 384
    delta_y = abs(second.y - first.y) * 512
    assert delta_x / delta_y == pytest.approx(0.5)
    assert (first.x + second.x) / 2 == pytest.approx(0.5)
    assert (first.y + second.y) / 2 == pytest.approx(0.5)


def test_pose_normalization_keeps_a_margin() -> None:
    pose = make_pose((0, 0.0, 0.0), (1, 1.0, 1.0))

    normalized = normalize_body_pose(
        pose,
        source_size=(384, 512),
        margin_fraction=0.1,
    )
    points = [point for point in normalized.keypoints if point is not None]

    tolerance = 1e-9
    assert min(point.x for point in points) >= 0.1 - tolerance
    assert max(point.x for point in points) <= 0.9 + tolerance
    assert min(point.y for point in points) >= 0.1 - tolerance
    assert max(point.y for point in points) <= 0.9 + tolerance


def test_result_does_not_expose_source_image() -> None:
    result_fields = {field.name for field in fields(PoseExtractionResult)}
    assert result_fields == {
        "pose_image",
        "person_count",
        "detected_body_keypoints",
        "detector_input_size",
    }


def test_preview_overlays_pose_without_dimming_background() -> None:
    detector = FakeDetector([make_pose((0, 0.25, 0.25), (1, 0.75, 0.75))])
    result = PoseExtractor(detector).preview(
        Image.new("RGB", (200, 400), "white"),
        opacity=1.0,
    )
    overlay = np.asarray(result.overlay_image)

    assert result.overlay_image.size == (200, 400)
    assert result.pose_image.size == (384, 512)
    assert np.any(overlay != 255)
    assert np.any(np.all(overlay == 255, axis=2))
    assert np.asarray(result.pose_image).any()


def test_overlay_rejects_mismatched_dimensions() -> None:
    with pytest.raises(ValueError, match="matching dimensions"):
        overlay_pose_map(
            Image.new("RGB", (100, 100)),
            Image.new("RGB", (50, 50)),
        )


@pytest.mark.parametrize("opacity", [-0.1, 1.1, True])
def test_overlay_rejects_invalid_opacity(opacity: object) -> None:
    with pytest.raises(ValueError, match="opacity"):
        overlay_pose_map(
            Image.new("RGB", (10, 10)),
            Image.new("RGB", (10, 10)),
            opacity=opacity,  # type: ignore[arg-type]
        )


class RawDetector:
    def __init__(self, raw_poses: list[object]) -> None:
        self.raw_poses = raw_poses
        self.include_hand: bool | None = None
        self.include_face: bool | None = None

    def detect_poses(
        self,
        image: np.ndarray,
        *,
        include_hand: bool,
        include_face: bool,
    ) -> list[object]:
        self.include_hand = include_hand
        self.include_face = include_face
        return self.raw_poses


def test_controlnet_adapter_requests_body_only() -> None:
    raw_keypoint = SimpleNamespace(x=0.25, y=0.75, score=0.9, id=3)
    raw_body = SimpleNamespace(
        keypoints=[raw_keypoint, None],
        total_score=0.9,
        total_parts=1,
    )
    raw_detector = RawDetector([SimpleNamespace(body=raw_body)])

    result = ControlNetAuxOpenPoseAdapter(raw_detector).detect(
        np.zeros((10, 10, 3), dtype=np.uint8)
    )

    assert raw_detector.include_hand is False
    assert raw_detector.include_face is False
    assert result == [
        BodyPose(
            keypoints=(BodyKeypoint(0.25, 0.75, 0.9, 3), None),
            total_score=0.9,
            total_parts=1,
        )
    ]


def test_public_body_checkpoint_download_ignores_environment_token() -> None:
    with (
        patch(
            "huggingface_hub.hf_hub_download",
            return_value="/temporary/body_pose_model.pth",
        ) as download,
        patch("controlnet_aux.open_pose.body.Body") as body_class,
        patch("controlnet_aux.open_pose.OpenposeDetector") as detector_class,
    ):
        adapter = ControlNetAuxOpenPoseAdapter.from_pretrained()

    download.assert_called_once_with(
        repo_id="lllyasviel/Annotators",
        filename="body_pose_model.pth",
        cache_dir=None,
        token=False,
        local_files_only=False,
    )
    body_class.return_value.model.to.assert_called_once_with("cpu")
    detector_class.assert_called_once_with(body_estimation=body_class.return_value)
    assert isinstance(adapter, ControlNetAuxOpenPoseAdapter)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("max_input_side", 0, "positive integer"),
        ("canvas_size", (0, 512), "positive integers"),
        ("margin_fraction", 0.5, "less than 0.5"),
    ],
)
def test_extractor_rejects_invalid_settings(
    keyword: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PoseExtractor(FakeDetector([]), **{keyword: value})  # type: ignore[arg-type]
