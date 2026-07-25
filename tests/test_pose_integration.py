from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
import pytest

from gundamposer.pose import PoseExtractor


_SAMPLE_PATH = os.getenv("GUNDAMPOSER_POSE_SAMPLE")


@pytest.mark.integration
@pytest.mark.skipif(
    not _SAMPLE_PATH,
    reason="Set GUNDAMPOSER_POSE_SAMPLE to an authorized one-person photo.",
)
def test_real_body_pose_extraction() -> None:
    sample_path = Path(_SAMPLE_PATH or "")
    cache_dir = os.getenv("GUNDAMPOSER_MODEL_CACHE")
    extractor = PoseExtractor.from_pretrained(cache_dir=cache_dir)

    with Image.open(sample_path) as image:
        result = extractor.extract(image)

    assert result.pose_image.size == (384, 512)
    assert result.person_count == 1
    assert result.detected_body_keypoints > 0

