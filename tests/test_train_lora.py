from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from gundamposer.config import load_training_config
from scripts.train_lora import (
    TrainingError,
    TrainingImageDataset,
    TrainingOverrides,
    effective_training_settings,
    inspect_training_run,
    main,
    resolve_resume_checkpoint,
)


CONFIG_PATH = Path("configs/training.yaml")


class FakeTokenizer:
    model_max_length = 8

    def __call__(self, text: str, **_: object) -> SimpleNamespace:
        assert text == "a hwmecha model"
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]]))


def _training_split(tmp_path: Path, count: int = 2) -> Path:
    root = tmp_path / "train"
    root.mkdir()
    rows = []
    for index in range(count):
        filename = f"image_{index + 1:04d}.png"
        Image.new("RGB", (512, 512), (index * 20, 50, 100)).save(root / filename)
        rows.append({"file_name": filename, "text": "a hwmecha model"})
    (root / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return root


def test_training_dataset_loads_normalized_pixels_and_tokens(tmp_path: Path) -> None:
    dataset = TrainingImageDataset(
        _training_split(tmp_path),
        FakeTokenizer(),
        resolution=512,
    )

    item = dataset[0]

    assert len(dataset) == 2
    assert item["pixel_values"].shape == (3, 512, 512)
    assert item["pixel_values"].dtype == torch.float32
    assert np.isclose(float(item["pixel_values"].min()), -1.0)
    assert item["input_ids"].tolist() == [1, 2, 3, 0, 0, 0, 0, 0]


def test_training_dataset_rejects_wrong_dimensions(tmp_path: Path) -> None:
    root = _training_split(tmp_path, count=1)
    Image.new("RGB", (256, 512)).save(root / "image_0001.png")
    dataset = TrainingImageDataset(root, FakeTokenizer(), resolution=512)

    with pytest.raises(TrainingError, match="512x512"):
        dataset[0]


def test_mps_training_uses_full_precision() -> None:
    settings = effective_training_settings(
        load_training_config(CONFIG_PATH),
        device="mps",
    )

    assert settings.mixed_precision == "no"
    assert settings.weight_dtype == torch.float32


def test_cuda_training_honors_configured_fp16() -> None:
    settings = effective_training_settings(
        load_training_config(CONFIG_PATH),
        device="cuda",
    )

    assert settings.mixed_precision == "fp16"
    assert settings.weight_dtype == torch.float16


def test_step_overrides_are_validated() -> None:
    with pytest.raises(TrainingError, match="cannot exceed"):
        effective_training_settings(
            load_training_config(CONFIG_PATH),
            device="cpu",
            overrides=TrainingOverrides(
                max_train_steps=5,
                checkpointing_steps=6,
            ),
        )


def test_latest_checkpoint_uses_highest_step(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-000005").mkdir()
    (tmp_path / "checkpoint-000010").mkdir()

    assert resolve_resume_checkpoint(tmp_path, "latest") == (
        tmp_path / "checkpoint-000010"
    )


def test_inspect_training_run_reports_effective_batch(
    tmp_path: Path,
) -> None:
    summary = inspect_training_run(
        load_training_config(CONFIG_PATH),
        _training_split(tmp_path),
        device="cpu",
        overrides=TrainingOverrides(max_train_steps=5, checkpointing_steps=5),
    )

    assert summary.image_count == 2
    assert summary.max_train_steps == 5
    assert summary.checkpointing_steps == 5
    assert summary.effective_batch_size == 4
    assert summary.mixed_precision == "no"


def test_cli_dry_run_does_not_load_models(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--train-data",
            str(_training_split(tmp_path)),
            "--device",
            "cpu",
            "--max-train-steps",
            "5",
            "--checkpointing-steps",
            "5",
            "--dry-run",
        ]
    )

    assert result == 0
    assert '"image_count": 2' in capsys.readouterr().out
