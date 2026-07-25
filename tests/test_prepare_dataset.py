from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import shutil

from PIL import Image
import pytest

from scripts.prepare_dataset import (
    CAPTION,
    DatasetPreparationError,
    SPLITS,
    audit_dataset,
    create_contact_sheet,
    prepare_dataset,
)


def _write_noise_image(path: Path, seed: int, size: tuple[int, int] = (31, 19)) -> None:
    randomizer = random.Random(seed)
    image = Image.new("RGB", size)
    image.putdata(
        [
            (
                randomizer.randrange(256),
                randomizer.randrange(256),
                randomizer.randrange(256),
            )
            for _ in range(size[0] * size[1])
        ]
    )
    image.save(path)


@pytest.fixture
def source_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    for split_index, split in enumerate(SPLITS):
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        for image_index in range(2):
            stem = f"sample_{image_index}"
            _write_noise_image(
                images / f"{stem}.jpg",
                seed=split_index * 10 + image_index,
            )
            (labels / f"{stem}.txt").write_text(
                "0 0.1 0.1 0.9 0.9\n",
                encoding="utf-8",
            )
    return root


def test_audit_reports_split_counts_and_pairings(source_dataset: Path) -> None:
    report = audit_dataset(source_dataset)

    assert report.counts == {"train": 2, "valid": 2, "test": 2}
    assert report.total == 6
    assert all(record.label_path.exists() for record in report.records)


def test_audit_rejects_missing_label(source_dataset: Path) -> None:
    (source_dataset / "valid" / "labels" / "sample_0.txt").unlink()

    with pytest.raises(DatasetPreparationError, match="without labels"):
        audit_dataset(source_dataset)


def test_audit_rejects_corrupt_image(source_dataset: Path) -> None:
    (source_dataset / "test" / "images" / "sample_1.jpg").write_bytes(b"broken")

    with pytest.raises(DatasetPreparationError, match="Unreadable image"):
        audit_dataset(source_dataset)


def test_audit_rejects_exact_cross_split_duplicate(source_dataset: Path) -> None:
    shutil.copyfile(
        source_dataset / "train" / "images" / "sample_0.jpg",
        source_dataset / "valid" / "images" / "sample_0.jpg",
    )

    with pytest.raises(DatasetPreparationError, match="Exact image duplicates"):
        audit_dataset(source_dataset)


def test_audit_can_preserve_source_splits_with_duplicates(
    source_dataset: Path,
) -> None:
    shutil.copyfile(
        source_dataset / "train" / "images" / "sample_0.jpg",
        source_dataset / "valid" / "images" / "sample_0.jpg",
    )

    report = audit_dataset(
        source_dataset,
        allow_cross_split_duplicates=True,
    )

    assert report.counts == {"train": 2, "valid": 2, "test": 2}


def test_audit_rejects_perceptual_cross_split_duplicate(source_dataset: Path) -> None:
    source = source_dataset / "train" / "images" / "sample_0.jpg"
    target = source_dataset / "test" / "images" / "sample_0.jpg"
    with Image.open(source) as image:
        image.save(target, quality=60)

    with pytest.raises(DatasetPreparationError, match="perceptual duplicates"):
        audit_dataset(source_dataset)


def test_audit_rejects_crop_variant_across_splits(source_dataset: Path) -> None:
    source = source_dataset / "train" / "images" / "sample_0.jpg"
    target = source_dataset / "valid" / "images" / "sample_0.jpg"
    _write_noise_image(source, seed=123, size=(320, 240))
    with Image.open(source) as image:
        image.crop((30, 20, 290, 220)).resize((320, 240)).save(target, quality=85)

    with pytest.raises(DatasetPreparationError, match="crop or resize variants"):
        audit_dataset(source_dataset)


def test_dry_run_does_not_create_destination(
    source_dataset: Path,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "data" / "processed"

    report = prepare_dataset(source_dataset, destination, dry_run=True)

    assert report.total == 6
    assert not destination.exists()


def test_prepare_writes_split_metadata_manifest_and_square_images(
    source_dataset: Path,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "data" / "processed"

    prepare_dataset(source_dataset, destination)

    for split in SPLITS:
        images = sorted((destination / split).glob("*.png"))
        metadata_rows = [
            json.loads(line)
            for line in (destination / split / "metadata.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [path.name for path in images] == [
            "image_0001.png",
            "image_0002.png",
        ]
        assert metadata_rows == [
            {"file_name": path.name, "text": CAPTION} for path in images
        ]
        for image_path in images:
            with Image.open(image_path) as image:
                assert image.mode == "RGB"
                assert image.size == (512, 512)

    manifest = [
        json.loads(line)
        for line in (destination / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(manifest) == 6
    for row in manifest:
        output_path = destination / row["output_path"]
        assert output_path.exists()
        assert hashlib.sha256(output_path.read_bytes()).hexdigest() == row[
            "output_sha256"
        ]


def test_prepare_can_keep_duplicate_source_assignments(
    source_dataset: Path,
    tmp_path: Path,
) -> None:
    shutil.copyfile(
        source_dataset / "train" / "images" / "sample_0.jpg",
        source_dataset / "test" / "images" / "sample_0.jpg",
    )
    destination = tmp_path / "data" / "processed"

    report = prepare_dataset(
        source_dataset,
        destination,
        allow_cross_split_duplicates=True,
    )

    assert report.counts == {"train": 2, "valid": 2, "test": 2}
    assert len(list(destination.rglob("*.png"))) == 6


def test_nonempty_destination_requires_overwrite(
    source_dataset: Path,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "data" / "processed"
    destination.mkdir(parents=True)
    (destination / "unexpected.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(DatasetPreparationError, match="not empty"):
        prepare_dataset(source_dataset, destination)

    assert (destination / "unexpected.txt").read_text(encoding="utf-8") == "keep"


def test_overwrite_is_deterministic_and_restricted(
    source_dataset: Path,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "data" / "processed"
    prepare_dataset(source_dataset, destination)
    first_hashes = {
        path.relative_to(destination): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in destination.rglob("*")
        if path.is_file()
    }

    prepare_dataset(source_dataset, destination, overwrite=True)
    second_hashes = {
        path.relative_to(destination): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in destination.rglob("*")
        if path.is_file()
    }

    assert first_hashes == second_hashes
    with pytest.raises(DatasetPreparationError, match="restricted"):
        prepare_dataset(source_dataset, tmp_path / "elsewhere", overwrite=True)


def test_contact_sheet_spans_all_splits(
    source_dataset: Path,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "data" / "processed"
    contact_sheet = tmp_path / "contact-sheet.png"
    prepare_dataset(source_dataset, destination)

    create_contact_sheet(destination, contact_sheet, samples_per_split=1)

    with Image.open(contact_sheet) as image:
        assert image.width > 0
        assert image.height > image.width
