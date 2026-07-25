from __future__ import annotations

import pytest

from gundamposer.prompts import (
    NEGATIVE_PROMPT,
    SCENE_PRESETS,
    TRIGGER_WORD,
    build_prompt,
    build_scene_prompt,
    normalize_prompt_text,
)


def test_scene_presets_match_the_public_choices() -> None:
    assert tuple(SCENE_PRESETS) == (
        "Neutral studio",
        "Futuristic city",
        "Space station",
        "Forest battlefield",
        "Desert ruins",
        "Industrial hangar",
    )


def test_build_scene_prompt_combines_preset_and_description() -> None:
    result = build_scene_prompt("Futuristic city", "  at night,  neon lights ")
    assert result == "futuristic city, at night, neon lights"


def test_build_scene_prompt_uses_preset_when_description_is_blank() -> None:
    assert build_scene_prompt("Neutral studio", " \n ") == "neutral studio background"


def test_build_scene_prompt_rejects_unknown_preset() -> None:
    with pytest.raises(ValueError, match="Unknown scene preset"):
        build_scene_prompt("Ocean")


def test_build_prompt_normalizes_text_and_has_one_trigger() -> None:
    result = build_prompt(" futuristic \n city,  at night ")

    assert "futuristic city, at night" in result
    assert result.count(TRIGGER_WORD) == 1
    assert "  " not in result


def test_build_prompt_removes_trigger_from_scene_text() -> None:
    result = build_prompt("hwmecha industrial hangar HWMECHA")
    assert result.lower().count(TRIGGER_WORD) == 1


def test_build_prompt_falls_back_to_default_scene() -> None:
    assert "neutral studio background" in build_prompt("  ")


def test_normalize_prompt_text_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        normalize_prompt_text(None)  # type: ignore[arg-type]


def test_negative_prompt_contains_required_exclusions() -> None:
    for term in ("toy", "cropped body", "extra limbs", "text", "watermark"):
        assert term in NEGATIVE_PROMPT

