from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
import pytest

from scripts.create_comparisons import (
    EvaluationError,
    Rating,
    create_comparison_image,
    create_showcase_grid,
    read_ratings,
    summarize_ratings,
    validate_output_root,
)


def test_comparison_strip_has_three_labeled_panels() -> None:
    comparison = create_comparison_image(
        Image.new("RGB", (384, 512), "black"),
        Image.new("RGB", (384, 512), "blue"),
        Image.new("RGB", (384, 512), "red"),
    )

    assert comparison.size == (1152, 558)
    assert comparison.getpixel((10, 100)) == (0, 0, 0)
    assert comparison.getpixel((394, 100)) == (0, 0, 255)
    assert comparison.getpixel((778, 100)) == (255, 0, 0)


def test_showcase_grid_arranges_four_comparisons_in_two_rows() -> None:
    comparisons = [
        Image.new("RGB", (1152, 558), (index, index, index)) for index in range(4)
    ]

    assert create_showcase_grid(comparisons).size == (2304, 1116)


def test_showcase_grid_requires_an_image() -> None:
    with pytest.raises(EvaluationError, match="At least one"):
        create_showcase_grid([])


def test_output_root_is_restricted_to_project_outputs(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="subdirectory"):
        validate_output_root(tmp_path)


def test_rating_summary_reports_averages_deltas_and_success() -> None:
    ratings = [
        Rating("one", "baseline", 4, 2, 4, 3),
        Rating("one", "trained", 4, 5, 4, 4),
        Rating("two", "baseline", 2, 3, 2, 2),
        Rating("two", "trained", 3, 4, 3, 2),
    ]

    summary = summarize_ratings(ratings)

    assert summary["examples_rated"] == 2
    assert summary["averages"]["baseline"]["pose_match"] == 3.0
    assert summary["averages"]["trained"]["mecha_appearance"] == 4.5
    assert summary["trained_minus_baseline"]["pose_match"] == 0.5
    assert summary["primary_success_criterion_met"] is True


def test_rating_summary_rejects_a_missing_variant() -> None:
    with pytest.raises(EvaluationError, match="baseline and trained"):
        summarize_ratings([Rating("one", "baseline", 3, 3, 3, 3)])


def test_read_ratings_requires_complete_scores_in_range(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    with ratings_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example",
                "variant",
                "pose_match",
                "mecha_appearance",
                "full_body_completeness",
                "artifact_quality",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "example": "one",
                "variant": "baseline",
                "pose_match": 6,
                "mecha_appearance": 3,
                "full_body_completeness": 3,
                "artifact_quality": 3,
            }
        )

    with pytest.raises(EvaluationError, match="1 to 5"):
        read_ratings(ratings_path)
