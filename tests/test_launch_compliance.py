"""Published compliance default, local off switches, and saved job consent."""

from __future__ import annotations

from pathlib import Path

import pytest

from barrett_faceswap.cli import main
from barrett_faceswap.config import load_config
from barrett_faceswap.errors import ConsentError
from barrett_faceswap.jobs import JobStore
from barrett_faceswap.pipeline import FaceSwapPipeline
from barrett_faceswap.types import SwapRequest
from tests.conftest import REPO, write_config, write_png


def test_env_and_local_yaml_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_config(tmp_path, enabled=True)
    assert load_config(path).compliance.enabled is True

    monkeypatch.setenv("BARRETT_DEV_MODE", "1")
    assert load_config(path).compliance.enabled is False

    monkeypatch.setenv("BARRETT_COMPLIANCE", "off")
    assert load_config(path).compliance.enabled is False

    monkeypatch.setenv("BARRETT_COMPLIANCE", "on")
    assert load_config(path).compliance.enabled is True

    monkeypatch.delenv("BARRETT_COMPLIANCE")
    monkeypatch.delenv("BARRETT_DEV_MODE")
    (tmp_path / "local.yaml").write_text("compliance:\n  enabled: false\n", encoding="utf-8")
    assert load_config(path).compliance.enabled is False

    monkeypatch.setenv("BARRETT_DEV_MODE", "1")
    monkeypatch.setenv("BARRETT_COMPLIANCE", "on")
    assert load_config(path).compliance.enabled is True


def test_shipped_dry_run_requires_acknowledgement(tmp_path: Path) -> None:
    config = load_config(REPO / "configs" / "default.yaml")
    compliance = config.compliance.model_copy(
        update={"audit": config.compliance.audit.model_copy(update={"path": tmp_path / "audit.jsonl"})}
    )
    config = config.model_copy(update={"compliance": compliance})
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    output = tmp_path / "out.png"
    write_png(source, (20, 40, 60))
    write_png(target, (0, 0, 0))
    request = SwapRequest(source_path=source, target_path=target, output_path=output, dry_run=True)
    pipeline = FaceSwapPipeline(config, admin_path=tmp_path / "missing-admin.toml")
    with pytest.raises(ConsentError):
        pipeline.run(request)
    result = pipeline.run(
        request.model_copy(update={"acknowledge_authorized_use": True, "consent_subject": "synthetic"})
    )
    assert result.compliance_enabled is True
    assert output.is_file()
    assert (tmp_path / "audit.jsonl").is_file()
    assert not (REPO / "outputs" / "audit" / "audit.jsonl").exists()


def test_jobs_submit_saves_acknowledgement(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, enabled=True)
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + f"\njobs:\n  root: {tmp_path / 'jobs'}\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    write_png(source)
    write_png(target)
    code = main(
        [
            "jobs",
            "submit",
            "--source",
            str(source),
            "--target",
            str(target),
            "--output",
            str(tmp_path / "out.png"),
            "--config",
            str(config_path),
            "--dry-run",
            "--job-id",
            "clip-a",
            "--acknowledge-authorized-use",
            "--consent-subject",
            "synthetic",
        ]
    )
    assert code == 0
    state = JobStore(tmp_path / "jobs").load("clip-a")
    assert state.spec["acknowledge_authorized_use"] is True
    assert state.spec["consent_subject"] == "synthetic"
