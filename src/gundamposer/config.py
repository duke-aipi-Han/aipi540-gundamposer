"""Validated loading for LoRA training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


SUPPORTED_PRECISIONS = frozenset({"no", "fp16", "bf16"})


class ConfigError(ValueError):
    """Raised when a configuration value is missing or invalid."""


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: int
    train_text_encoder: bool


@dataclass(frozen=True)
class TrainingRunConfig:
    max_train_steps: int
    train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    mixed_precision: str
    gradient_checkpointing: bool
    checkpointing_steps: int


@dataclass(frozen=True)
class ValidationConfig:
    seed: int
    prompts: tuple[str, ...]


@dataclass(frozen=True)
class TrainingConfig:
    base_model: str
    resolution: int
    seed: int
    lora: LoraConfig
    training: TrainingRunConfig
    validation: ValidationConfig


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required configuration key: {location}.{key}")
    return mapping[key]


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{location} must be a positive integer")
    return value


def _integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{location} must be an integer")
    return value


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{location} must be true or false")
    return value


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location} must be a non-empty string")
    return value.strip()


def _positive_float(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{location} must be a positive number")
    return float(value)


def parse_training_config(data: Any) -> TrainingConfig:
    """Validate parsed YAML data and return an immutable configuration."""

    root = _mapping(data, "config")
    lora = _mapping(_required(root, "lora", "config"), "config.lora")
    training = _mapping(
        _required(root, "training", "config"),
        "config.training",
    )
    validation = _mapping(
        _required(root, "validation", "config"),
        "config.validation",
    )

    resolution = _positive_int(
        _required(root, "resolution", "config"),
        "config.resolution",
    )
    if resolution % 8:
        raise ConfigError("config.resolution must be divisible by 8")

    precision = _nonempty_string(
        _required(training, "mixed_precision", "config.training"),
        "config.training.mixed_precision",
    )
    if precision not in SUPPORTED_PRECISIONS:
        supported = ", ".join(sorted(SUPPORTED_PRECISIONS))
        raise ConfigError(
            f"config.training.mixed_precision must be one of: {supported}"
        )

    raw_prompts = _required(validation, "prompts", "config.validation")
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ConfigError("config.validation.prompts must be a non-empty list")
    prompts = tuple(
        _nonempty_string(prompt, f"config.validation.prompts[{index}]")
        for index, prompt in enumerate(raw_prompts)
    )

    parsed = TrainingConfig(
        base_model=_nonempty_string(
            _required(root, "base_model", "config"),
            "config.base_model",
        ),
        resolution=resolution,
        seed=_integer(_required(root, "seed", "config"), "config.seed"),
        lora=LoraConfig(
            rank=_positive_int(
                _required(lora, "rank", "config.lora"),
                "config.lora.rank",
            ),
            alpha=_positive_int(
                _required(lora, "alpha", "config.lora"),
                "config.lora.alpha",
            ),
            train_text_encoder=_boolean(
                _required(lora, "train_text_encoder", "config.lora"),
                "config.lora.train_text_encoder",
            ),
        ),
        training=TrainingRunConfig(
            max_train_steps=_positive_int(
                _required(training, "max_train_steps", "config.training"),
                "config.training.max_train_steps",
            ),
            train_batch_size=_positive_int(
                _required(training, "train_batch_size", "config.training"),
                "config.training.train_batch_size",
            ),
            gradient_accumulation_steps=_positive_int(
                _required(
                    training,
                    "gradient_accumulation_steps",
                    "config.training",
                ),
                "config.training.gradient_accumulation_steps",
            ),
            learning_rate=_positive_float(
                _required(training, "learning_rate", "config.training"),
                "config.training.learning_rate",
            ),
            mixed_precision=precision,
            gradient_checkpointing=_boolean(
                _required(
                    training,
                    "gradient_checkpointing",
                    "config.training",
                ),
                "config.training.gradient_checkpointing",
            ),
            checkpointing_steps=_positive_int(
                _required(training, "checkpointing_steps", "config.training"),
                "config.training.checkpointing_steps",
            ),
        ),
        validation=ValidationConfig(
            seed=_integer(
                _required(validation, "seed", "config.validation"),
                "config.validation.seed",
            ),
            prompts=prompts,
        ),
    )

    if parsed.training.checkpointing_steps > parsed.training.max_train_steps:
        raise ConfigError(
            "config.training.checkpointing_steps cannot exceed max_train_steps"
        )
    return parsed


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load and validate a UTF-8 YAML training configuration file."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file)
    except OSError as error:
        raise ConfigError(f"Could not read configuration: {config_path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in configuration: {config_path}") from error
    return parse_training_config(data)

