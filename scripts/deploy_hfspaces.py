#!/usr/bin/env python3
"""Upload the selected LoRA and curated runtime files to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from huggingface_hub import CommitOperationAdd, HfApi, get_token


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE_REPO_ID = "hw391/AIPI540-GundamPoser"
DEFAULT_MODEL_REPO_ID = "hw391/AIPI540-GundamPoser-LoRA"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "gundamposer_lora.safetensors"
MODEL_WEIGHT_NAME = "gundamposer_lora.safetensors"
SPACE_FILE_MAP = (
    ("app.py", "app.py"),
    ("README.md", "README.md"),
    ("requirements-space.txt", "requirements.txt"),
    ("assets/pose-examples/ATTRIBUTION.md", "assets/pose-examples/ATTRIBUTION.md"),
    ("assets/pose-examples/action-balance.jpg", "assets/pose-examples/action-balance.jpg"),
    ("assets/pose-examples/celebration.jpg", "assets/pose-examples/celebration.jpg"),
    ("assets/pose-examples/martial-arts.jpg", "assets/pose-examples/martial-arts.jpg"),
    ("assets/pose-examples/running.jpg", "assets/pose-examples/running.jpg"),
    ("src/gundamposer/__init__.py", "gundamposer/__init__.py"),
    ("src/gundamposer/config.py", "gundamposer/config.py"),
    ("src/gundamposer/pipeline.py", "gundamposer/pipeline.py"),
    ("src/gundamposer/pose.py", "gundamposer/pose.py"),
    ("src/gundamposer/preprocessing.py", "gundamposer/preprocessing.py"),
    ("src/gundamposer/prompts.py", "gundamposer/prompts.py"),
)
SPACE_FILES = tuple(source for source, _ in SPACE_FILE_MAP)

MODEL_CARD = """---
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
library_name: diffusers
pipeline_tag: text-to-image
tags:
  - lora
  - stable-diffusion
  - controlnet
  - mecha-inspired
---

# GundamPoser LoRA

Private UNet-only LoRA adapter for the GundamPoser proof-of-concept. The
adapter uses the trigger word `hwmecha` and is intended for Stable Diffusion
1.5 pose-guided inference with OpenPose ControlNet.

This project describes generated outputs as mecha-inspired and does not claim
affiliation with Gundam, Bandai, or other rights holders.
"""


class DeploymentError(ValueError):
    """Raised when a deployment request is incomplete or unsafe."""


def collect_space_files(project_root: Path = PROJECT_ROOT) -> tuple[Path, ...]:
    """Return the explicit Space allowlist after validating every file."""

    root = project_root.expanduser().resolve()
    paths = tuple(root / relative_path for relative_path in SPACE_FILES)
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise DeploymentError(f"Missing Space file(s): {', '.join(missing)}")
    return paths


def build_model_operations(model_path: Path) -> list[CommitOperationAdd]:
    """Build the model commit without reading or copying token values."""

    resolved = model_path.expanduser().resolve()
    if not resolved.is_file():
        raise DeploymentError(f"LoRA adapter does not exist: {resolved}")
    if resolved.suffix != ".safetensors":
        raise DeploymentError("The LoRA adapter must be a .safetensors file.")
    return [
        CommitOperationAdd(
            path_in_repo=MODEL_WEIGHT_NAME,
            path_or_fileobj=resolved,
        ),
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=MODEL_CARD.encode("utf-8"),
        ),
    ]


def build_space_operations(
    project_root: Path = PROJECT_ROOT,
) -> list[CommitOperationAdd]:
    """Build a Space commit from only the reviewed runtime allowlist."""

    root = project_root.expanduser().resolve()
    return [
        CommitOperationAdd(
            path_in_repo=destination,
            path_or_fileobj=root / source,
        )
        for source, destination in SPACE_FILE_MAP
    ]


def describe_deployment(
    *,
    space_repo_id: str,
    model_repo_id: str,
    model_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> str:
    """Return a token-free, human-reviewable deployment summary."""

    root = project_root.expanduser().resolve()
    space_paths = collect_space_files(root)
    resolved_model = model_path.expanduser().resolve()
    build_model_operations(resolved_model)
    lines = [
        f"Private model repository: {model_repo_id}",
        f"  {resolved_model} -> {MODEL_WEIGHT_NAME}",
        "  generated private model card -> README.md",
        f"Protected Space repository: {space_repo_id}",
    ]
    for (source, destination), path in zip(SPACE_FILE_MAP, space_paths):
        suffix = "" if source == destination else f" -> {destination}"
        lines.append(f"  {path.relative_to(root)}{suffix}")
    return "\n".join(lines)


def deploy(
    api: Any,
    *,
    space_repo_id: str,
    model_repo_id: str,
    model_path: Path,
    project_root: Path = PROJECT_ROOT,
    upload_model: bool = True,
) -> tuple[Any | None, Any]:
    """Upload the private model first, followed by curated Space files."""

    model_commit = None
    if upload_model:
        model_commit = api.create_commit(
            repo_id=model_repo_id,
            repo_type="model",
            operations=build_model_operations(model_path),
            commit_message="Upload selected GundamPoser LoRA adapter",
        )
    space_commit = api.create_commit(
        repo_id=space_repo_id,
        repo_type="space",
        operations=build_space_operations(project_root),
        commit_message="Deploy GundamPoser ZeroGPU app",
    )
    return model_commit, space_commit


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-repo-id", default=DEFAULT_SPACE_REPO_ID)
    parser.add_argument("--model-repo-id", default=DEFAULT_MODEL_REPO_ID)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform both Hub commits. Without this flag, only print the file plan.",
    )
    parser.add_argument(
        "--space-only",
        action="store_true",
        help="Skip the model commit when retrying after the adapter was uploaded.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    print(
        describe_deployment(
            space_repo_id=args.space_repo_id,
            model_repo_id=args.model_repo_id,
            model_path=args.model_path,
        )
    )
    if args.space_only:
        print("Model commit: skipped (--space-only)")
    if not args.apply:
        print("Dry run only. Add --apply to perform the listed commit(s).")
        return 0

    token = get_token()
    if not token:
        raise DeploymentError(
            "No local Hugging Face token is available. Authenticate locally or set "
            "HF_TOKEN before using --apply."
        )
    model_commit, space_commit = deploy(
        HfApi(token=token),
        space_repo_id=args.space_repo_id,
        model_repo_id=args.model_repo_id,
        model_path=args.model_path,
        upload_model=not args.space_only,
    )
    if model_commit is not None:
        print(f"Model commit: {model_commit.commit_url}")
    print(f"Space commit: {space_commit.commit_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
