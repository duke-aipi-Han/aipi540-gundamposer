"""CPU body-pose extraction and OpenPose control-map rendering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2
import numpy as np
from PIL import Image

from gundamposer.preprocessing import resize_max_side, to_oriented_rgb


DEFAULT_ANNOTATOR_ID = "lllyasviel/Annotators"
DEFAULT_BODY_MODEL_FILE = "body_pose_model.pth"
DEFAULT_INPUT_MAX_SIDE = 768
DEFAULT_CANVAS_SIZE = (384, 512)
DEFAULT_MARGIN_FRACTION = 0.08

_BODY_LIMBS = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)

_BODY_COLORS = (
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
)


class PoseExtractionError(ValueError):
    """Base error for invalid pose inputs or detections."""


class NoPersonDetectedError(PoseExtractionError):
    """Raised when no person is detected."""


class MultiplePeopleDetectedError(PoseExtractionError):
    """Raised when more than one person is detected."""


class InvalidPoseError(PoseExtractionError):
    """Raised when a detected person has no usable body keypoints."""


@dataclass(frozen=True)
class BodyKeypoint:
    x: float
    y: float
    score: float = 1.0
    id: int = -1


@dataclass(frozen=True)
class BodyPose:
    keypoints: tuple[BodyKeypoint | None, ...]
    total_score: float = 0.0
    total_parts: int = 0


@dataclass(frozen=True)
class PoseExtractionResult:
    pose_image: Image.Image
    person_count: int
    detected_body_keypoints: int
    detector_input_size: tuple[int, int]


@dataclass(frozen=True)
class PosePreviewResult:
    overlay_image: Image.Image
    pose_image: Image.Image
    person_count: int
    detected_body_keypoints: int
    detector_input_size: tuple[int, int]


class BodyPoseDetector(Protocol):
    def detect(self, image: np.ndarray) -> Sequence[BodyPose]:
        """Return body-only poses detected in an RGB uint8 image."""


def _validate_size(size: tuple[int, int], name: str) -> None:
    if len(size) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in size
    ):
        raise ValueError(f"{name} must contain two positive integers")


class ControlNetAuxOpenPoseAdapter:
    """Body-only adapter for the pinned ``controlnet-aux`` detector."""

    def __init__(self, detector: Any) -> None:
        self._detector = detector

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_ANNOTATOR_ID,
        *,
        filename: str = DEFAULT_BODY_MODEL_FILE,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> "ControlNetAuxOpenPoseAdapter":
        """Load only the body estimator and keep it explicitly on CPU."""

        from controlnet_aux.open_pose import OpenposeDetector
        from controlnet_aux.open_pose.body import Body
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            token=False,
            local_files_only=local_files_only,
        )
        body_estimation = Body(model_path)
        body_estimation.model.to("cpu")
        return cls(OpenposeDetector(body_estimation=body_estimation))

    def detect(self, image: np.ndarray) -> list[BodyPose]:
        raw_poses = self._detector.detect_poses(
            image,
            include_hand=False,
            include_face=False,
        )
        return [self._to_body_pose(raw_pose) for raw_pose in raw_poses]

    @staticmethod
    def _to_body_pose(raw_pose: Any) -> BodyPose:
        body = raw_pose.body
        keypoints = tuple(
            None
            if keypoint is None
            else BodyKeypoint(
                x=float(keypoint.x),
                y=float(keypoint.y),
                score=float(keypoint.score),
                id=int(keypoint.id),
            )
            for keypoint in body.keypoints
        )
        return BodyPose(
            keypoints=keypoints,
            total_score=float(body.total_score),
            total_parts=int(body.total_parts),
        )


def _usable_keypoints(pose: BodyPose) -> list[BodyKeypoint]:
    return [
        point
        for point in pose.keypoints
        if point is not None and math.isfinite(point.x) and math.isfinite(point.y)
    ]


def normalize_body_pose(
    pose: BodyPose,
    *,
    source_size: tuple[int, int],
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
    margin_fraction: float = DEFAULT_MARGIN_FRACTION,
) -> BodyPose:
    """Center and scale a body pose while preserving its pixel-space shape."""

    source_width, source_height = source_size
    canvas_width, canvas_height = canvas_size
    _validate_size(source_size, "source_size")
    _validate_size(canvas_size, "canvas_size")
    if not 0 <= margin_fraction < 0.5:
        raise ValueError("margin_fraction must be at least 0 and less than 0.5")

    usable = _usable_keypoints(pose)
    if not usable:
        raise InvalidPoseError("A person was detected, but no body keypoints were usable.")

    pixel_x = [point.x * source_width for point in usable]
    pixel_y = [point.y * source_height for point in usable]
    min_x, max_x = min(pixel_x), max(pixel_x)
    min_y, max_y = min(pixel_y), max(pixel_y)
    pose_width = max_x - min_x
    pose_height = max_y - min_y
    available_width = canvas_width * (1 - 2 * margin_fraction)
    available_height = canvas_height * (1 - 2 * margin_fraction)

    scale_candidates = []
    if pose_width > 0:
        scale_candidates.append(available_width / pose_width)
    if pose_height > 0:
        scale_candidates.append(available_height / pose_height)
    scale = min(scale_candidates) if scale_candidates else 1.0
    pose_center_x = (min_x + max_x) / 2
    pose_center_y = (min_y + max_y) / 2

    transformed: list[BodyKeypoint | None] = []
    for point in pose.keypoints:
        if point is None or not math.isfinite(point.x) or not math.isfinite(point.y):
            transformed.append(None)
            continue
        target_x = ((point.x * source_width - pose_center_x) * scale) + (
            canvas_width / 2
        )
        target_y = ((point.y * source_height - pose_center_y) * scale) + (
            canvas_height / 2
        )
        transformed.append(
            BodyKeypoint(
                x=min(1.0, max(0.0, target_x / canvas_width)),
                y=min(1.0, max(0.0, target_y / canvas_height)),
                score=point.score,
                id=point.id,
            )
        )

    return BodyPose(
        keypoints=tuple(transformed),
        total_score=pose.total_score,
        total_parts=pose.total_parts,
    )


def render_body_pose(
    pose: BodyPose,
    *,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
) -> Image.Image:
    """Render a body-only OpenPose map on a black RGB canvas."""

    canvas_width, canvas_height = canvas_size
    _validate_size(canvas_size, "canvas_size")
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    for (first_index, second_index), color in zip(_BODY_LIMBS, _BODY_COLORS):
        if max(first_index, second_index) >= len(pose.keypoints):
            continue
        first = pose.keypoints[first_index]
        second = pose.keypoints[second_index]
        if first is None or second is None:
            continue
        first_pixel = (round(first.x * canvas_width), round(first.y * canvas_height))
        second_pixel = (
            round(second.x * canvas_width),
            round(second.y * canvas_height),
        )
        limb_color = tuple(round(channel * 0.6) for channel in color)
        cv2.line(canvas, first_pixel, second_pixel, limb_color, thickness=8)

    for point, color in zip(pose.keypoints, _BODY_COLORS):
        if point is None:
            continue
        pixel = (round(point.x * canvas_width), round(point.y * canvas_height))
        cv2.circle(canvas, pixel, radius=4, color=color, thickness=-1)

    return Image.fromarray(canvas)


def overlay_pose_map(
    image: Image.Image,
    pose_image: Image.Image,
    *,
    opacity: float = 0.75,
) -> Image.Image:
    """Overlay non-black pose pixels on an aligned source image."""

    if not isinstance(opacity, (int, float)) or isinstance(opacity, bool):
        raise ValueError("opacity must be a number between 0 and 1")
    if not 0 <= opacity <= 1:
        raise ValueError("opacity must be between 0 and 1")

    base = to_oriented_rgb(image)
    pose = to_oriented_rgb(pose_image)
    if base.size != pose.size:
        raise ValueError("image and pose_image must have matching dimensions")

    pose_array = np.asarray(pose, dtype=np.uint8)
    alpha_array = np.where(
        pose_array.any(axis=2),
        round(255 * opacity),
        0,
    ).astype(np.uint8)
    colored_pose = pose.convert("RGBA")
    colored_pose.putalpha(Image.fromarray(alpha_array))
    return Image.alpha_composite(base.convert("RGBA"), colored_pose).convert("RGB")


class PoseExtractor:
    """Extract exactly one body pose without retaining the source image."""

    def __init__(
        self,
        detector: BodyPoseDetector,
        *,
        max_input_side: int = DEFAULT_INPUT_MAX_SIDE,
        canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
        margin_fraction: float = DEFAULT_MARGIN_FRACTION,
    ) -> None:
        if (
            isinstance(max_input_side, bool)
            or not isinstance(max_input_side, int)
            or max_input_side <= 0
        ):
            raise ValueError("max_input_side must be a positive integer")
        _validate_size(canvas_size, "canvas_size")
        if not 0 <= margin_fraction < 0.5:
            raise ValueError("margin_fraction must be at least 0 and less than 0.5")
        self._detector = detector
        self._max_input_side = max_input_side
        self._canvas_size = canvas_size
        self._margin_fraction = margin_fraction

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_ANNOTATOR_ID,
        *,
        filename: str = DEFAULT_BODY_MODEL_FILE,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        max_input_side: int = DEFAULT_INPUT_MAX_SIDE,
        canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
        margin_fraction: float = DEFAULT_MARGIN_FRACTION,
    ) -> "PoseExtractor":
        detector = ControlNetAuxOpenPoseAdapter.from_pretrained(
            model_id,
            filename=filename,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return cls(
            detector,
            max_input_side=max_input_side,
            canvas_size=canvas_size,
            margin_fraction=margin_fraction,
        )

    def _detect_one(self, image: Image.Image) -> tuple[Image.Image, BodyPose]:
        normalized = to_oriented_rgb(image)
        detector_input = resize_max_side(normalized, self._max_input_side)
        detector_array = np.asarray(detector_input, dtype=np.uint8)
        detected = list(self._detector.detect(detector_array))

        if not detected:
            raise NoPersonDetectedError(
                "No person was detected. Upload a full-body photo of one person."
            )
        if len(detected) > 1:
            raise MultiplePeopleDetectedError(
                "Multiple people were detected. Upload a photo containing one person."
            )

        return detector_input, detected[0]

    def extract(self, image: Image.Image) -> PoseExtractionResult:
        detector_input, detected_pose = self._detect_one(image)

        normalized_pose = normalize_body_pose(
            detected_pose,
            source_size=detector_input.size,
            canvas_size=self._canvas_size,
            margin_fraction=self._margin_fraction,
        )
        pose_image = render_body_pose(
            normalized_pose,
            canvas_size=self._canvas_size,
        )
        return PoseExtractionResult(
            pose_image=pose_image,
            person_count=1,
            detected_body_keypoints=len(_usable_keypoints(detected_pose)),
            detector_input_size=detector_input.size,
        )

    def preview(
        self,
        image: Image.Image,
        *,
        opacity: float = 0.75,
    ) -> PosePreviewResult:
        """Return a detector-aligned pose map and temporary visual overlay."""

        detector_input, detected_pose = self._detect_one(image)
        if not _usable_keypoints(detected_pose):
            raise InvalidPoseError(
                "A person was detected, but no body keypoints were usable."
            )
        aligned_pose = render_body_pose(
            detected_pose,
            canvas_size=detector_input.size,
        )
        overlay = overlay_pose_map(
            detector_input,
            aligned_pose,
            opacity=opacity,
        )
        return PosePreviewResult(
            overlay_image=overlay,
            pose_image=aligned_pose,
            person_count=1,
            detected_body_keypoints=len(_usable_keypoints(detected_pose)),
            detector_input_size=detector_input.size,
        )
