#!/usr/bin/env python3
"""Create reproducible baseline-versus-LoRA comparison images."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from gundamposer.pipeline import DEFAULT_LORA_STRENGTH, GundamPoserPipeline
from gundamposer.pose import PoseExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "evaluation"
DEFAULT_LORA_PATH = PROJECT_ROOT / "outputs" / "gundamposer_lora.safetensors"
RATING_METRICS = (
    "pose_match",
    "mecha_appearance",
    "full_body_completeness",
    "artifact_quality",
)


class EvaluationError(ValueError):
    """Raised when an evaluation request or rating file is invalid."""


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    photo_path: Path
    prompt: str
    seed: int


@dataclass(frozen=True)
class Rating:
    example: str
    variant: str
    pose_match: int
    mecha_appearance: int
    full_body_completeness: int
    artifact_quality: int
    notes: str = ""


DEFAULT_SPECS = (
    ComparisonSpec(
        name="running",
        photo_path=PROJECT_ROOT / "assets" / "pose-examples" / "running.jpg",
        prompt=(
            "a wide full-body shot of a life-sized humanoid warrior wearing "
            "hwmecha mechanical armor, running with the same arm and leg positions "
            "as the control pose, head-to-toe entirely visible, futuristic city, "
            "detailed articulated armor, dynamic action photograph"
        ),
        seed=42,
    ),
    ComparisonSpec(
        name="action-balance",
        photo_path=(
            PROJECT_ROOT / "assets" / "pose-examples" / "action-balance.jpg"
        ),
        prompt=(
            "a wide full-body shot of a life-sized humanoid warrior wearing "
            "hwmecha mechanical armor, balancing on one leg with the same arm and "
            "leg positions as the control pose, head-to-toe entirely visible, "
            "neutral studio background, detailed articulated armor"
        ),
        seed=314,
    ),
    ComparisonSpec(
        name="celebration",
        photo_path=(
            PROJECT_ROOT / "assets" / "pose-examples" / "celebration.jpg"
        ),
        prompt=(
            "a wide full-body shot of a triumphant life-sized humanoid warrior "
            "wearing hwmecha mechanical armor, matching the raised-arm celebration "
            "pose, head-to-toe entirely visible, industrial hangar, detailed "
            "articulated armor, cinematic photograph"
        ),
        seed=2718,
    ),
    ComparisonSpec(
        name="martial-arts",
        photo_path=(
            PROJECT_ROOT / "assets" / "pose-examples" / "martial-arts.jpg"
        ),
        prompt=(
            "a wide full-body shot of a life-sized humanoid warrior wearing "
            "hwmecha mechanical armor, performing an extreme vertical side kick "
            "with the same arm and leg positions as the control pose, head-to-toe "
            "entirely visible, neutral studio background, detailed articulated armor"
        ),
        seed=1337,
    ),
)


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def create_comparison_image(
    pose: Image.Image,
    baseline: Image.Image,
    trained: Image.Image,
) -> Image.Image:
    """Return a labeled pose, baseline, and trained image strip."""

    panels = [image.convert("RGB").resize((384, 512)) for image in (pose, baseline, trained)]
    labels = ("Pose control", "Base SD 1.5", "Trained LoRA")
    label_height = 46
    result = Image.new("RGB", (384 * 3, 512 + label_height), "white")
    draw = ImageDraw.Draw(result)
    font = _font(20)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        left = index * 384
        result.paste(panel, (left, label_height))
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text(
            (left + (384 - text_width) / 2, 12),
            label,
            fill="black",
            font=font,
        )
    return result


def create_showcase_grid(comparisons: Sequence[Image.Image]) -> Image.Image:
    """Arrange up to four labeled comparisons in a slide-friendly grid."""

    if not comparisons:
        raise EvaluationError("At least one comparison is required.")
    panels = [comparison.convert("RGB") for comparison in comparisons]
    panel_width = max(panel.width for panel in panels)
    panel_height = max(panel.height for panel in panels)
    columns = 2 if len(panels) > 1 else 1
    rows = (len(panels) + columns - 1) // columns
    grid = Image.new("RGB", (panel_width * columns, panel_height * rows), "white")
    for index, panel in enumerate(panels):
        grid.paste(panel, ((index % columns) * panel_width, (index // columns) * panel_height))
    return grid


def _metadata(result: object) -> dict[str, object]:
    metadata = getattr(result, "metadata")
    return asdict(metadata)


def _write_rating_template(path: Path, names: Iterable[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["example", "variant", *RATING_METRICS, "notes"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name in names:
            for variant in ("baseline", "trained"):
                writer.writerow({"example": name, "variant": variant})


def validate_output_root(path: Path) -> Path:
    """Restrict generated and overwritten files to this project's outputs."""

    resolved = path.expanduser().resolve()
    allowed_root = (PROJECT_ROOT / "outputs").resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise EvaluationError(
            f"Output directory must be a subdirectory of {allowed_root}."
        )
    return resolved


def generate_comparisons(
    specs: Sequence[ComparisonSpec],
    *,
    output_root: Path,
    lora_path: Path,
    device: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Generate seed-matched pairs and return the showcase-grid path."""

    if not specs:
        raise EvaluationError("At least one comparison specification is required.")
    missing = [str(spec.photo_path) for spec in specs if not spec.photo_path.is_file()]
    if missing:
        raise EvaluationError(f"Missing pose photo(s): {', '.join(missing)}")
    if not lora_path.is_file():
        raise EvaluationError(f"LoRA adapter does not exist: {lora_path}")
    output_root = validate_output_root(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise EvaluationError(
                f"Output directory is not empty: {output_root}. Use --overwrite."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    extractor = PoseExtractor.from_pretrained()
    pipeline = GundamPoserPipeline.load(
        device=device,
        lora_path=lora_path,
    )
    comparison_images: list[Image.Image] = []
    manifest: list[dict[str, object]] = []

    for spec in specs:
        with Image.open(spec.photo_path) as photo:
            preview = extractor.preview(photo.convert("RGB"), opacity=0.75)
        baseline = pipeline.generate(
            preview.pose_image,
            "",
            spec.seed,
            lora_strength=0.0,
            prompt_override=spec.prompt,
        )
        trained = pipeline.generate(
            preview.pose_image,
            "",
            spec.seed,
            lora_strength=DEFAULT_LORA_STRENGTH,
            prompt_override=spec.prompt,
        )

        example_root = output_root / spec.name
        example_root.mkdir()
        preview.pose_image.save(example_root / "pose.png")
        baseline.image.convert("RGB").save(example_root / "baseline.png")
        trained.image.convert("RGB").save(example_root / "trained.png")
        comparison = create_comparison_image(
            preview.pose_image,
            baseline.image,
            trained.image,
        )
        comparison.save(example_root / "comparison.png")
        comparison_images.append(comparison)

        record = {
            "example": spec.name,
            "source": str(spec.photo_path.relative_to(PROJECT_ROOT)),
            "prompt": spec.prompt,
            "seed": spec.seed,
            "detected_body_keypoints": preview.detected_body_keypoints,
            "baseline": _metadata(baseline),
            "trained": _metadata(trained),
        }
        (example_root / "metadata.json").write_text(
            json.dumps(record, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest.append(record)

    showcase_path = output_root / "showcase-grid.png"
    create_showcase_grid(comparison_images).save(showcase_path)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_rating_template(output_root / "ratings.csv", (spec.name for spec in specs))
    return showcase_path


def read_ratings(path: Path) -> list[Rating]:
    """Read complete 1-5 ratings from a CSV file."""

    ratings: list[Rating] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                values = {metric: int(row[metric]) for metric in RATING_METRICS}
            except (KeyError, TypeError, ValueError) as error:
                raise EvaluationError(
                    f"Row {row_number} must contain integer scores for every metric."
                ) from error
            if any(not 1 <= value <= 5 for value in values.values()):
                raise EvaluationError(f"Row {row_number} scores must be from 1 to 5.")
            variant = row.get("variant", "")
            if variant not in {"baseline", "trained"}:
                raise EvaluationError(
                    f"Row {row_number} variant must be baseline or trained."
                )
            ratings.append(
                Rating(
                    example=row.get("example", ""),
                    variant=variant,
                    notes=row.get("notes", ""),
                    **values,
                )
            )
    if not ratings:
        raise EvaluationError("The ratings file is empty.")
    return ratings


def summarize_ratings(ratings: Sequence[Rating]) -> dict[str, object]:
    """Compute variant averages and trained-minus-baseline deltas."""

    grouped = {
        variant: [rating for rating in ratings if rating.variant == variant]
        for variant in ("baseline", "trained")
    }
    if not all(grouped.values()):
        raise EvaluationError("Ratings must include baseline and trained rows.")
    averages: dict[str, dict[str, float]] = {}
    for variant, rows in grouped.items():
        averages[variant] = {
            metric: round(
                sum(getattr(rating, metric) for rating in rows) / len(rows),
                2,
            )
            for metric in RATING_METRICS
        }
    deltas = {
        metric: round(averages["trained"][metric] - averages["baseline"][metric], 2)
        for metric in RATING_METRICS
    }
    return {
        "scale": "1-5; higher artifact_quality means fewer visible artifacts",
        "examples_rated": len({rating.example for rating in ratings}),
        "averages": averages,
        "trained_minus_baseline": deltas,
        "primary_success_criterion_met": (
            deltas["mecha_appearance"] > 0 and deltas["pose_match"] >= 0
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="Generate comparison images.")
    generate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    generate.add_argument("--lora-path", type=Path, default=DEFAULT_LORA_PATH)
    generate.add_argument("--device", choices=("cuda", "mps", "cpu"))
    generate.add_argument("--overwrite", action="store_true")
    summarize = subparsers.add_parser("summarize", help="Summarize completed ratings.")
    summarize.add_argument("ratings", type=Path)
    summarize.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "generate":
        showcase = generate_comparisons(
            DEFAULT_SPECS,
            output_root=args.output_root,
            lora_path=args.lora_path,
            device=args.device,
            overwrite=args.overwrite,
        )
        print(f"Created {len(DEFAULT_SPECS)} comparisons: {showcase}")
        return 0
    summary = summarize_ratings(read_ratings(args.ratings))
    serialized = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
        print(f"Wrote evaluation summary: {args.output}")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
