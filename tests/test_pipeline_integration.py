from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
import pytest

from gundamposer.pipeline import GundamPoserPipeline, resolve_device


_POSE_PATH = os.getenv("GUNDAMPOSER_CONTROL_IMAGE")
_REQUESTED_DEVICE = os.getenv("GUNDAMPOSER_DEVICE") or None
_DEVICE = resolve_device(_REQUESTED_DEVICE)


@pytest.mark.integration
@pytest.mark.skipif(
    not _POSE_PATH or _DEVICE == "cpu",
    reason="Set GUNDAMPOSER_CONTROL_IMAGE and run with CUDA or MPS.",
)
def test_real_baseline_generation() -> None:
    cache_dir = os.getenv("GUNDAMPOSER_MODEL_CACHE")
    pipeline = GundamPoserPipeline.load(device=_DEVICE, cache_dir=cache_dir)

    with Image.open(Path(_POSE_PATH or "")) as image:
        control_image = image.convert("RGB")
    result = pipeline.generate(control_image, "neutral studio background", 42)

    assert result.image.size == (384, 512)
    assert result.metadata.seed == 42
    assert result.metadata.device in {"cuda", "mps"}
    assert result.metadata.lora_strength == 0.0
