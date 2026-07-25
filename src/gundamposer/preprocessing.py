"""Deterministic image preparation utilities."""

from __future__ import annotations

from typing import TypeAlias

from PIL import Image, ImageOps


Color: TypeAlias = tuple[int, int, int]
Size: TypeAlias = tuple[int, int]

DEFAULT_BACKGROUND: Color = (255, 255, 255)
TRAINING_RESOLUTION = 512


def _validate_size(size: Size) -> None:
    if len(size) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in size
    ):
        raise ValueError("Image dimensions must be two positive integers.")


def to_oriented_rgb(
    image: Image.Image,
    *,
    background: Color = DEFAULT_BACKGROUND,
) -> Image.Image:
    """Apply EXIF orientation and return an RGB image.

    Images with transparency are composited over ``background`` so transparent
    pixels do not silently become black.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance")

    oriented = ImageOps.exif_transpose(image)
    if oriented.mode in {"RGBA", "LA"} or (
        oriented.mode == "P" and "transparency" in oriented.info
    ):
        rgba = oriented.convert("RGBA")
        backdrop = Image.new("RGBA", rgba.size, (*background, 255))
        return Image.alpha_composite(backdrop, rgba).convert("RGB")

    return oriented.convert("RGB")


def resize_max_side(
    image: Image.Image,
    max_side: int,
    *,
    allow_upscale: bool = False,
) -> Image.Image:
    """Resize proportionally so neither side exceeds ``max_side``."""

    if isinstance(max_side, bool) or not isinstance(max_side, int) or max_side <= 0:
        raise ValueError("max_side must be a positive integer")

    width, height = image.size
    _validate_size((width, height))
    scale = max_side / max(width, height)
    if not allow_upscale:
        scale = min(1.0, scale)

    resized_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    if resized_size == image.size:
        return image.copy()
    return image.resize(resized_size, Image.Resampling.LANCZOS)


def letterbox(
    image: Image.Image,
    size: Size,
    *,
    background: Color = DEFAULT_BACKGROUND,
) -> Image.Image:
    """Fit an image inside a fixed canvas without stretching or cropping."""

    _validate_size(size)
    normalized = to_oriented_rgb(image, background=background)
    source_width, source_height = normalized.size
    target_width, target_height = size
    scale = min(target_width / source_width, target_height / source_height)
    fitted_size = (
        max(1, round(source_width * scale)),
        max(1, round(source_height * scale)),
    )
    fitted = normalized.resize(fitted_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", size, background)
    offset = (
        (target_width - fitted.width) // 2,
        (target_height - fitted.height) // 2,
    )
    canvas.paste(fitted, offset)
    return canvas


def prepare_training_image(
    image: Image.Image,
    *,
    resolution: int = TRAINING_RESOLUTION,
    background: Color = DEFAULT_BACKGROUND,
) -> Image.Image:
    """Return a square, RGB training image at the requested resolution."""

    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
        raise ValueError("resolution must be a positive integer")
    return letterbox(image, (resolution, resolution), background=background)
