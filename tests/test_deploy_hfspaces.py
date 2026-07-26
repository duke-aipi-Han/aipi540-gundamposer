from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.deploy_hfspaces import (
    DeploymentError,
    MODEL_WEIGHT_NAME,
    SPACE_FILE_MAP,
    SPACE_FILES,
    build_model_operations,
    build_space_operations,
    collect_space_files,
    deploy,
    main,
)


def _write_space_files(root: Path) -> None:
    for relative_path in SPACE_FILES:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"runtime file")


def test_space_upload_uses_only_the_explicit_runtime_allowlist(tmp_path: Path) -> None:
    _write_space_files(tmp_path)
    excluded = tmp_path / "outputs" / "secret.safetensors"
    excluded.parent.mkdir()
    excluded.write_bytes(b"do not upload")

    operations = build_space_operations(tmp_path)

    assert [operation.path_in_repo for operation in operations] == [
        destination for _, destination in SPACE_FILE_MAP
    ]
    assert all("outputs" not in operation.path_in_repo for operation in operations)


def test_missing_space_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DeploymentError, match="Missing Space file"):
        collect_space_files(tmp_path)


def test_model_upload_uses_fixed_weight_name_and_generated_card(tmp_path: Path) -> None:
    adapter = tmp_path / "selected.safetensors"
    adapter.write_bytes(b"adapter")

    operations = build_model_operations(adapter)

    assert [operation.path_in_repo for operation in operations] == [
        MODEL_WEIGHT_NAME,
        "README.md",
    ]


def test_non_safetensors_model_is_rejected(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter.bin"
    adapter.write_bytes(b"adapter")

    with pytest.raises(DeploymentError, match="safetensors"):
        build_model_operations(adapter)


def test_deploy_uploads_model_before_space(tmp_path: Path) -> None:
    _write_space_files(tmp_path)
    adapter = tmp_path / "selected.safetensors"
    adapter.write_bytes(b"adapter")
    calls: list[dict[str, object]] = []

    class FakeApi:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(commit_url=f"https://example/{len(calls)}")

    deploy(
        FakeApi(),
        space_repo_id="owner/space",
        model_repo_id="owner/model",
        model_path=adapter,
        project_root=tmp_path,
    )

    assert [call["repo_type"] for call in calls] == ["model", "space"]
    assert [call["repo_id"] for call in calls] == ["owner/model", "owner/space"]


def test_space_only_retry_skips_model_commit(tmp_path: Path) -> None:
    _write_space_files(tmp_path)
    adapter = tmp_path / "selected.safetensors"
    adapter.write_bytes(b"adapter")
    calls: list[dict[str, object]] = []

    class FakeApi:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(commit_url="https://example/space")

    model_commit, _ = deploy(
        FakeApi(),
        space_repo_id="owner/space",
        model_repo_id="owner/model",
        model_path=adapter,
        project_root=tmp_path,
        upload_model=False,
    )

    assert model_commit is None
    assert [call["repo_type"] for call in calls] == ["space"]


def test_command_defaults_to_a_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.deploy_hfspaces.describe_deployment",
        lambda **kwargs: "safe plan",
    )
    monkeypatch.setattr(
        "scripts.deploy_hfspaces.deploy",
        lambda *args, **kwargs: pytest.fail("dry run must not deploy"),
    )

    assert main([]) == 0
