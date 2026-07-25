#!/usr/bin/env python3
"""Audit and prepare split-safe image folders for LoRA training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

from gundamposer.preprocessing import prepare_training_image, to_oriented_rgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "gunpla-yolov7"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_CONTACT_SHEET = PROJECT_ROOT / "outputs" / "dataset-contact-sheet.png"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
CAPTION = "a full-body hwmecha humanoid mecha model with detailed mechanical armor"
PERCEPTUAL_HASH_THRESHOLD = 4
FEATURE_MATCH_RATIO = 0.75
FEATURE_MIN_INLIERS = 12
FEATURE_MIN_INLIER_RATIO = 0.5


class DatasetPreparationError(ValueError):
    """Raised when source data or an output request is unsafe or inconsistent."""


@dataclass(frozen=True)
class SourceRecord:
    split: str
    image_path: Path
    label_path: Path
    source_sha256: str
    perceptual_hash: int
    source_size: tuple[int, int]
    feature_points: tuple[tuple[float, float], ...] = field(
        repr=False,
        compare=False,
    )
    feature_descriptors: np.ndarray | None = field(repr=False, compare=False)


@dataclass(frozen=True)
class AuditReport:
    source_root: Path
    records: tuple[SourceRecord, ...]

    @property
    def counts(self) -> dict[str, int]:
        counts = Counter(record.split for record in self.records)
        return {split: counts[split] for split in SPLITS}

    @property
    def total(self) -> int:
        return len(self.records)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _difference_hash(image: Image.Image) -> int:
    grayscale = to_oriented_rgb(image).convert("L").resize(
        (9, 8),
        Image.Resampling.LANCZOS,
    )
    pixels = grayscale.tobytes()
    result = 0
    for row in range(8):
        for column in range(8):
            offset = row * 9 + column
            result = (result << 1) | (pixels[offset] > pixels[offset + 1])
    return result


def _local_features(
    image: Image.Image,
) -> tuple[tuple[tuple[float, float], ...], np.ndarray | None]:
    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = grayscale.shape
    scale = min(1.0, 512 / max(width, height))
    if scale < 1.0:
        grayscale = cv2.resize(
            grayscale,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    detector = cv2.ORB_create(nfeatures=750)
    keypoints, descriptors = detector.detectAndCompute(grayscale, None)
    return tuple(keypoint.pt for keypoint in keypoints), descriptors


def _format_paths(paths: Iterable[Path], root: Path) -> str:
    return ", ".join(str(path.relative_to(root)) for path in sorted(paths))


def _load_record(split: str, image_path: Path, label_path: Path) -> SourceRecord:
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            oriented = to_oriented_rgb(image)
            perceptual_hash = _difference_hash(oriented)
            source_size = oriented.size
            feature_points, feature_descriptors = _local_features(oriented)
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise DatasetPreparationError(
            f"Unreadable image {image_path}: {error}"
        ) from error

    if not label_path.read_text(encoding="utf-8").strip():
        raise DatasetPreparationError(f"Empty label file: {label_path}")

    return SourceRecord(
        split=split,
        image_path=image_path,
        label_path=label_path,
        source_sha256=_sha256(image_path),
        perceptual_hash=perceptual_hash,
        source_size=source_size,
        feature_points=feature_points,
        feature_descriptors=feature_descriptors,
    )


def _feature_inliers(first: SourceRecord, second: SourceRecord) -> tuple[int, float]:
    if (
        first.feature_descriptors is None
        or second.feature_descriptors is None
        or len(first.feature_descriptors) < 2
        or len(second.feature_descriptors) < 2
    ):
        return 0, 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    candidate_matches = matcher.knnMatch(
        first.feature_descriptors,
        second.feature_descriptors,
        k=2,
    )
    good_matches = [
        best
        for best, runner_up in candidate_matches
        if best.distance < FEATURE_MATCH_RATIO * runner_up.distance
    ]
    if len(good_matches) < FEATURE_MIN_INLIERS:
        return 0, 0.0

    first_points = np.float32(
        [first.feature_points[match.queryIdx] for match in good_matches]
    ).reshape(-1, 1, 2)
    second_points = np.float32(
        [second.feature_points[match.trainIdx] for match in good_matches]
    ).reshape(-1, 1, 2)
    _, inlier_mask = cv2.findHomography(
        first_points,
        second_points,
        cv2.RANSAC,
        5.0,
    )
    if inlier_mask is None:
        return 0, 0.0
    inliers = int(inlier_mask.sum())
    return inliers, inliers / len(good_matches)


def _check_cross_split_duplicates(
    records: Sequence[SourceRecord],
    *,
    perceptual_threshold: int,
) -> None:
    exact_groups: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        exact_groups[record.source_sha256].append(record)

    exact_leaks = [
        group
        for group in exact_groups.values()
        if len({record.split for record in group}) > 1
    ]
    if exact_leaks:
        details = "; ".join(
            ", ".join(
                f"{record.split}/{record.image_path.name}" for record in group
            )
            for group in exact_leaks
        )
        raise DatasetPreparationError(
            f"Exact image duplicates cross dataset splits: {details}"
        )

    perceptual_leaks: list[tuple[SourceRecord, SourceRecord, int]] = []
    feature_leaks: list[tuple[SourceRecord, SourceRecord, int, float]] = []
    for index, first in enumerate(records):
        for second in records[index + 1 :]:
            if first.split == second.split:
                continue
            distance = (first.perceptual_hash ^ second.perceptual_hash).bit_count()
            if distance <= perceptual_threshold:
                perceptual_leaks.append((first, second, distance))
                continue
            inliers, inlier_ratio = _feature_inliers(first, second)
            if (
                inliers >= FEATURE_MIN_INLIERS
                and inlier_ratio >= FEATURE_MIN_INLIER_RATIO
            ):
                feature_leaks.append((first, second, inliers, inlier_ratio))

    if perceptual_leaks:
        details = "; ".join(
            f"{first.split}/{first.image_path.name} and "
            f"{second.split}/{second.image_path.name} (distance {distance})"
            for first, second, distance in perceptual_leaks
        )
        raise DatasetPreparationError(
            f"Likely perceptual duplicates cross dataset splits: {details}"
        )
    if feature_leaks:
        displayed = feature_leaks[:20]
        details = "; ".join(
            f"{first.split}/{first.image_path.name} and "
            f"{second.split}/{second.image_path.name} "
            f"({inliers} geometric feature matches, {inlier_ratio:.0%} inliers)"
            for first, second, inliers, inlier_ratio in displayed
        )
        remaining = len(feature_leaks) - len(displayed)
        if remaining:
            details += f"; and {remaining} more pair(s)"
        raise DatasetPreparationError(
            "Likely crop or resize variants cross dataset splits: " + details
        )


def audit_dataset(
    source_root: Path,
    *,
    perceptual_threshold: int = PERCEPTUAL_HASH_THRESHOLD,
    allow_cross_split_duplicates: bool = False,
) -> AuditReport:
    """Validate split structure, image-label pairing, integrity, and leakage."""

    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise DatasetPreparationError(f"Source directory does not exist: {source_root}")
    if perceptual_threshold < 0:
        raise DatasetPreparationError("perceptual threshold must be non-negative")

    records: list[SourceRecord] = []
    for split in SPLITS:
        images_root = source_root / split / "images"
        labels_root = source_root / split / "labels"
        if not images_root.is_dir() or not labels_root.is_dir():
            raise DatasetPreparationError(
                f"Split {split!r} must contain images/ and labels/ directories."
            )

        image_paths = sorted(
            path
            for path in images_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        label_paths = sorted(labels_root.glob("*.txt"))
        images_by_stem = {path.stem: path for path in image_paths}
        labels_by_stem = {path.stem: path for path in label_paths}
        missing_labels = images_by_stem.keys() - labels_by_stem.keys()
        orphan_labels = labels_by_stem.keys() - images_by_stem.keys()
        if missing_labels or orphan_labels:
            problems = []
            if missing_labels:
                problems.append(
                    "images without labels: "
                    + _format_paths(
                        (images_by_stem[stem] for stem in missing_labels),
                        source_root,
                    )
                )
            if orphan_labels:
                problems.append(
                    "labels without images: "
                    + _format_paths(
                        (labels_by_stem[stem] for stem in orphan_labels),
                        source_root,
                    )
                )
            raise DatasetPreparationError("; ".join(problems))
        if not image_paths:
            raise DatasetPreparationError(f"Split {split!r} contains no images.")

        for image_path in image_paths:
            records.append(
                _load_record(split, image_path, labels_by_stem[image_path.stem])
            )

    if not allow_cross_split_duplicates:
        _check_cross_split_duplicates(
            records,
            perceptual_threshold=perceptual_threshold,
        )
    return AuditReport(source_root=source_root, records=tuple(records))


def _safe_overwrite_destination(destination: Path) -> bool:
    resolved = destination.resolve()
    return resolved.name == "processed" and resolved.parent.name == "data"


def _write_dataset(report: AuditReport, staging_root: Path) -> None:
    manifest_rows: list[dict[str, object]] = []
    for split in SPLITS:
        split_root = staging_root / split
        split_root.mkdir(parents=True)
        metadata_path = split_root / "metadata.jsonl"
        split_records = [record for record in report.records if record.split == split]
        with metadata_path.open("w", encoding="utf-8", newline="\n") as metadata:
            for index, record in enumerate(split_records, start=1):
                filename = f"image_{index:04d}.png"
                output_path = split_root / filename
                with Image.open(record.image_path) as source_image:
                    prepared = prepare_training_image(source_image)
                    prepared.save(output_path, format="PNG", compress_level=9)

                metadata.write(
                    json.dumps(
                        {"file_name": filename, "text": CAPTION},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                manifest_rows.append(
                    {
                        "split": split,
                        "source_path": str(
                            record.image_path.relative_to(report.source_root)
                        ),
                        "source_sha256": record.source_sha256,
                        "source_size": list(record.source_size),
                        "output_path": f"{split}/{filename}",
                        "output_sha256": _sha256(output_path),
                    }
                )

    with (staging_root / "manifest.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as manifest:
        for row in manifest_rows:
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    perceptual_threshold: int = PERCEPTUAL_HASH_THRESHOLD,
    allow_cross_split_duplicates: bool = False,
) -> AuditReport:
    """Audit and write deterministic processed splits without editing sources."""

    report = audit_dataset(
        source_root,
        perceptual_threshold=perceptual_threshold,
        allow_cross_split_duplicates=allow_cross_split_duplicates,
    )
    if dry_run:
        return report

    destination = output_root.resolve()
    destination_has_content = destination.exists() and any(destination.iterdir())
    if destination_has_content and not overwrite:
        raise DatasetPreparationError(
            f"Destination is not empty: {destination}. Use --overwrite to replace it."
        )
    if overwrite and not _safe_overwrite_destination(destination):
        raise DatasetPreparationError(
            "--overwrite is restricted to a directory named data/processed."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".processed-staging-", dir=destination.parent)
    )
    try:
        _write_dataset(report, staging_root)
        if destination.exists():
            if destination_has_content:
                shutil.rmtree(destination)
            else:
                destination.rmdir()
        staging_root.replace(destination)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    return report


def create_contact_sheet(
    output_root: Path,
    contact_sheet_path: Path,
    *,
    samples_per_split: int = 8,
    thumbnail_size: int = 160,
) -> None:
    """Create a deterministic, labeled sample grid spanning every split."""

    if samples_per_split <= 0 or thumbnail_size <= 0:
        raise DatasetPreparationError("contact sheet sizes must be positive")
    rows: list[tuple[str, list[Path]]] = []
    for split in SPLITS:
        images = sorted((output_root / split).glob("*.png"))
        if not images:
            raise DatasetPreparationError(
                f"Cannot create contact sheet: no processed {split} images."
            )
        sample_count = min(samples_per_split, len(images))
        if sample_count == 1:
            indexes = [0]
        else:
            indexes = [
                round(index * (len(images) - 1) / (sample_count - 1))
                for index in range(sample_count)
            ]
        rows.append((split, [images[index] for index in indexes]))

    label_height = 28
    margin = 8
    columns = max(len(images) for _, images in rows)
    sheet = Image.new(
        "RGB",
        (
            margin + columns * (thumbnail_size + margin),
            margin + len(rows) * (label_height + thumbnail_size + margin),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, (split, images) in enumerate(rows):
        top = margin + row_index * (label_height + thumbnail_size + margin)
        draw.text((margin, top + 6), split, fill="black")
        for column, image_path in enumerate(images):
            with Image.open(image_path) as image:
                thumbnail = image.copy()
            thumbnail.thumbnail((thumbnail_size, thumbnail_size))
            left = margin + column * (thumbnail_size + margin)
            image_top = top + label_height
            sheet.paste(thumbnail, (left, image_top))

    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(contact_sheet_path, format="PNG", compress_level=9)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-cross-split-duplicates",
        action="store_true",
        help=(
            "Preserve source split assignments even when duplicate or variant "
            "images cross splits."
        ),
    )
    parser.add_argument(
        "--perceptual-threshold",
        type=int,
        default=PERCEPTUAL_HASH_THRESHOLD,
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        nargs="?",
        const=DEFAULT_CONTACT_SHEET,
        help="Optionally write a deterministic cross-split sample grid.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = prepare_dataset(
            args.source,
            args.output,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            perceptual_threshold=args.perceptual_threshold,
            allow_cross_split_duplicates=args.allow_cross_split_duplicates,
        )
        action = "Audit complete" if args.dry_run else "Preparation complete"
        print(
            f"{action}: "
            + ", ".join(f"{split}={count}" for split, count in report.counts.items())
            + f", total={report.total}"
        )
        if args.contact_sheet is not None:
            if args.dry_run:
                raise DatasetPreparationError(
                    "--contact-sheet cannot be combined with --dry-run."
                )
            create_contact_sheet(args.output.resolve(), args.contact_sheet.resolve())
            print(f"Contact sheet: {args.contact_sheet.resolve()}")
    except DatasetPreparationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
