"""Prompt constants and construction helpers."""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Mapping


TRIGGER_WORD = "hwmecha"
DEFAULT_SCENE = "neutral studio background"

SCENE_PRESETS: Mapping[str, str] = MappingProxyType(
    {
        "Neutral studio": DEFAULT_SCENE,
        "Futuristic city": "futuristic city",
        "Space station": "space station",
        "Forest battlefield": "forest battlefield",
        "Desert ruins": "desert ruins",
        "Industrial hangar": "industrial hangar",
    }
)

PROMPT_TEMPLATE = (
    "a full-body life-sized humanoid warrior wearing hwmecha mechanical armor, "
    "human proportions, articulated armor panels, detailed mechanical joints, "
    "complete body, {scene_prompt}, high detail"
)

NEGATIVE_PROMPT = (
    "toy, miniature, tabletop, display stand, plastic model photography, "
    "cropped body, missing limbs, extra limbs, duplicate body, malformed arms, "
    "malformed legs, blurry, low detail, text, logo, watermark"
)

_TRIGGER_PATTERN = re.compile(rf"\b{re.escape(TRIGGER_WORD)}\b", re.IGNORECASE)


def normalize_prompt_text(text: str) -> str:
    """Collapse repeated whitespace and tidy spacing around commas."""

    if not isinstance(text, str):
        raise TypeError("prompt text must be a string")
    normalized = " ".join(text.split())
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    return normalized.strip(" ,")


def build_scene_prompt(preset: str, description: str = "") -> str:
    """Combine a known scene preset with optional user-provided details."""

    if preset not in SCENE_PRESETS:
        choices = ", ".join(SCENE_PRESETS)
        raise ValueError(f"Unknown scene preset {preset!r}. Expected one of: {choices}")
    if not isinstance(description, str):
        raise TypeError("description must be a string")

    details = normalize_prompt_text(description)
    if not details:
        return SCENE_PRESETS[preset]
    return normalize_prompt_text(f"{SCENE_PRESETS[preset]}, {details}")


def build_prompt(scene_prompt: str = DEFAULT_SCENE) -> str:
    """Build the positive generation prompt with one trigger word."""

    scene = normalize_prompt_text(scene_prompt)
    scene = normalize_prompt_text(_TRIGGER_PATTERN.sub("", scene))
    if not scene:
        scene = DEFAULT_SCENE

    prompt = normalize_prompt_text(PROMPT_TEMPLATE.format(scene_prompt=scene))
    if len(_TRIGGER_PATTERN.findall(prompt)) != 1:
        raise ValueError("The generated prompt must contain the trigger word exactly once.")
    return prompt

