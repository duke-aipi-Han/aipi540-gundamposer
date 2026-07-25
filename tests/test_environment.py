"""Offline checks for the supported project environment."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from typing import get_origin, get_type_hints

import pytest


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.mark.parametrize(
    "module_name",
    [
        "accelerate",
        "cv2",
        "datasets",
        "diffusers",
        "gradio",
        "peft",
        "PIL",
        "safetensors",
        "torch",
        "transformers",
        "yaml",
    ],
)
def test_required_module_imports_offline(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_supported_python_version() -> None:
    assert sys.version_info[:2] == (3, 10)


def test_diffusion_pipeline_classes_import_offline() -> None:
    from diffusers import (
        ControlNetModel,
        DPMSolverMultistepScheduler,
        StableDiffusionControlNetPipeline,
    )

    assert ControlNetModel is not None
    assert DPMSolverMultistepScheduler is not None
    assert StableDiffusionControlNetPipeline is not None


def test_openpose_detector_exposes_pose_instances() -> None:
    from controlnet_aux import OpenposeDetector

    signature = inspect.signature(OpenposeDetector.detect_poses)
    assert tuple(signature.parameters) == (
        "self",
        "oriImg",
        "include_hand",
        "include_face",
    )

    return_type = get_type_hints(OpenposeDetector.detect_poses)["return"]
    assert get_origin(return_type) is list


@pytest.mark.parametrize("pose_count", [0, 1, 2])
def test_person_count_is_the_number_of_pose_results(pose_count: int) -> None:
    pose_results = [object() for _ in range(pose_count)]
    assert len(pose_results) == pose_count

