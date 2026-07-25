from __future__ import annotations

from PIL import Image, ImageChops
import pytest

from gundamposer.preprocessing import (
    letterbox,
    prepare_training_image,
    resize_max_side,
    to_oriented_rgb,
)


@pytest.mark.parametrize(
    ("source_size", "expected_content_box"),
    [
        ((200, 100), (0, 128, 512, 384)),
        ((100, 200), (128, 0, 384, 512)),
        ((200, 200), (0, 0, 512, 512)),
    ],
)
def test_letterbox_preserves_aspect_ratio_and_centers(
    source_size: tuple[int, int],
    expected_content_box: tuple[int, int, int, int],
) -> None:
    source = Image.new("RGB", source_size, "black")
    result = letterbox(source, (512, 512))
    white_canvas = Image.new("RGB", result.size, "white")

    assert result.size == (512, 512)
    assert ImageChops.difference(result, white_canvas).getbbox() == expected_content_box


@pytest.mark.parametrize(
    ("mode", "color"),
    [
        ("RGB", (10, 20, 30)),
        ("L", 20),
        ("RGBA", (10, 20, 30, 128)),
    ],
)
def test_training_preprocessing_handles_common_modes(
    mode: str,
    color: int | tuple[int, ...],
) -> None:
    source = Image.new(mode, (300, 500), color)
    result = prepare_training_image(source)

    assert result.mode == "RGB"
    assert result.size == (512, 512)


def test_transparency_is_composited_over_background() -> None:
    transparent = Image.new("RGBA", (2, 2), (10, 20, 30, 0))

    result = to_oriented_rgb(transparent)

    assert result.getpixel((0, 0)) == (255, 255, 255)


def test_exif_orientation_is_applied() -> None:
    source = Image.new("RGB", (20, 40), "black")
    source.getexif()[274] = 6

    result = to_oriented_rgb(source)

    assert result.size == (40, 20)


@pytest.mark.parametrize(
    ("source_size", "max_side", "expected_size"),
    [
        ((1600, 800), 768, (768, 384)),
        ((800, 1600), 768, (384, 768)),
        ((320, 240), 768, (320, 240)),
    ],
)
def test_resize_max_side(
    source_size: tuple[int, int],
    max_side: int,
    expected_size: tuple[int, int],
) -> None:
    source = Image.new("RGB", source_size)
    assert resize_max_side(source, max_side).size == expected_size


@pytest.mark.parametrize("invalid_max_side", [0, -1, True, 1.5])
def test_resize_rejects_invalid_max_side(invalid_max_side: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        resize_max_side(Image.new("RGB", (10, 10)), invalid_max_side)  # type: ignore[arg-type]


def test_training_preprocessing_rejects_invalid_resolution() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        prepare_training_image(Image.new("RGB", (10, 10)), resolution=0)


@pytest.mark.parametrize("invalid_size", [(0, 512), (512, -1), (512.5, 512)])
def test_letterbox_rejects_invalid_canvas_size(invalid_size: object) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        letterbox(Image.new("RGB", (10, 10)), invalid_size)  # type: ignore[arg-type]
