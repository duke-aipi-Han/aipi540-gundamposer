from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from gundamposer.config import ConfigError, load_training_config, parse_training_config


CONFIG_PATH = Path("configs/training.yaml")


@pytest.fixture
def valid_data() -> dict[str, object]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_load_training_config() -> None:
    config = load_training_config(CONFIG_PATH)

    assert config.base_model == "stable-diffusion-v1-5/stable-diffusion-v1-5"
    assert config.resolution == 512
    assert config.seed == 42
    assert config.lora.rank == 8
    assert config.lora.alpha == 8
    assert config.lora.train_text_encoder is False
    assert config.training.max_train_steps == 1500
    assert config.training.train_batch_size == 1
    assert config.training.gradient_accumulation_steps == 4
    assert config.training.learning_rate == pytest.approx(0.0001)
    assert config.training.mixed_precision == "fp16"
    assert config.training.gradient_checkpointing is True
    assert config.training.checkpointing_steps == 250
    assert config.validation.seed == 42
    assert len(config.validation.prompts) == 3


def test_missing_required_key_is_rejected(valid_data: dict[str, object]) -> None:
    data = deepcopy(valid_data)
    del data["base_model"]

    with pytest.raises(ConfigError, match="config.base_model"):
        parse_training_config(data)


def test_resolution_must_be_divisible_by_eight(
    valid_data: dict[str, object],
) -> None:
    data = deepcopy(valid_data)
    data["resolution"] = 510

    with pytest.raises(ConfigError, match="divisible by 8"):
        parse_training_config(data)


def test_unsupported_precision_is_rejected(valid_data: dict[str, object]) -> None:
    data = deepcopy(valid_data)
    training = data["training"]
    assert isinstance(training, dict)
    training["mixed_precision"] = "fp8"

    with pytest.raises(ConfigError, match="mixed_precision"):
        parse_training_config(data)


def test_checkpoint_interval_cannot_exceed_training(
    valid_data: dict[str, object],
) -> None:
    data = deepcopy(valid_data)
    training = data["training"]
    assert isinstance(training, dict)
    training["checkpointing_steps"] = 2000

    with pytest.raises(ConfigError, match="cannot exceed"):
        parse_training_config(data)


def test_validation_prompts_cannot_be_empty(valid_data: dict[str, object]) -> None:
    data = deepcopy(valid_data)
    validation = data["validation"]
    assert isinstance(validation, dict)
    validation["prompts"] = []

    with pytest.raises(ConfigError, match="non-empty list"):
        parse_training_config(data)


def test_missing_config_file_has_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError, match="Could not read configuration"):
        load_training_config(missing)


def test_invalid_yaml_has_clear_error(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("training: [", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_training_config(invalid)

