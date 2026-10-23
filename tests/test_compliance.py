"""Consent, watermark, audit log, and the content-filter permission gate."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest
from PIL import Image

from barrett_faceswap.cli import main
from barrett_faceswap.compliance.content_filter import FilterDecision, create_filter
from barrett_faceswap.compliance import content_filter as content_filter_module
from barrett_faceswap.compliance.content_filter import FilterSettings
from barrett_faceswap.config import load_config
from barrett_faceswap.errors import CompliancePermissionError, ConsentError, ContentBlockedError, ContentFilterError
from barrett_faceswap.i18n import t
from barrett_faceswap.pipeline import FaceSwapPipeline
from barrett_faceswap.types import SwapRequest
from tests.conftest import write_config, write_consent, write_png


def _run(tmp_path: Path, *, enabled: bool, request: SwapRequest, provider: str = "stub"):
    config = load_config(write_config(tmp_path, enabled=enabled, provider=provider))
    return FaceSwapPipeline(config, admin_path=tmp_path / "absent-admin.toml").run(request)


def _request(tmp_path: Path, **overrides) -> SwapRequest:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    if not source.exists():
        write_png(source, (12, 24, 36))
    if not target.exists():
        write_png(target, (0, 0, 0))
    values = {
        "source_path": source,
        "target_path": target,
        "output_path": tmp_path / "out.png",
        "dry_run": True,
    }
    values.update(overrides)
    return SwapRequest(**values)


def test_enabled_run_requires_consent(tmp_path: Path) -> None:
    with pytest.raises(ConsentError):
        _run(tmp_path, enabled=True, request=_request(tmp_path, enable_compliance=False))
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["event"] == "refused"


def test_cli_flag_enables_enforcement_even_when_config_is_off(tmp_path: Path) -> None:
    with pytest.raises(ConsentError):
        _run(tmp_path, enabled=False, request=_request(tmp_path, enable_compliance=True))


def test_consent_file_watermarks_and_audits(tmp_path: Path) -> None:
    consent = tmp_path / "consent.json"
    write_consent(consent)
    result = _run(
        tmp_path,
        enabled=True,
        request=_request(tmp_path, consent_file=consent),
    )
    assert result.compliance_enabled is True
    image = cv2.imread(str(result.output_path))
    assert image[-1].mean() > 0
    with Image.open(result.output_path) as marked:
        assert marked.info.get("barrett.faceswap.ai_generated") == "true"
        assert marked.info.get("Description") == "AI-generated"
    sidecar = Path(str(result.output_path) + ".meta.json")
    assert sidecar.is_file()
    record = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert record["event"] == "completed"
    assert record["watermark_applied"] is True
    assert record["consent_id"].startswith("file:")
    assert record["source_sha256"]
    assert record["output_sha256"]


def test_acknowledgement_is_enough_consent(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        enabled=True,
        request=_request(
            tmp_path,
            acknowledge_authorized_use=True,
            consent_subject="actor-1",
        ),
    )
    record = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert record["consent_id"].startswith("ack:")
    assert result.output_path.is_file()


def test_no_watermark_is_explicit_and_logged(tmp_path: Path) -> None:
    consent = tmp_path / "consent.json"
    write_consent(consent)
    result = _run(
        tmp_path,
        enabled=True,
        request=_request(tmp_path, consent_file=consent, no_watermark=True),
    )
    image = cv2.imread(str(result.output_path))
    assert image[-1].mean() == 0
    with Image.open(result.output_path) as marked:
        assert "barrett.faceswap.ai_generated" not in marked.info
    record = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert record["watermark_disabled"] is True
    assert record["watermark_applied"] is False


def test_audit_log_appends(tmp_path: Path) -> None:
    consent = tmp_path / "consent.json"
    write_consent(consent)
    request = _request(tmp_path, consent_file=consent)
    _run(tmp_path, enabled=True, request=request)
    second = request.model_copy(update={"output_path": tmp_path / "out-2.png"})
    _run(tmp_path, enabled=True, request=second)
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert json.loads(lines[1])["output_path"] != first["output_path"]


def test_disable_filter_without_permission_is_refused(tmp_path: Path) -> None:
    consent = tmp_path / "consent.json"
    write_consent(consent)
    with pytest.raises(CompliancePermissionError):
        _run(
            tmp_path,
            enabled=True,
            request=_request(tmp_path, consent_file=consent, disable_content_filter=True),
        )


def test_disable_filter_with_permission_is_audited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BARRETT_FACESWAP_ALLOW_FILTER_DISABLE", "1")
    consent = tmp_path / "consent.json"
    write_consent(consent)
    _run(
        tmp_path,
        enabled=True,
        request=_request(
            tmp_path,
            consent_file=consent,
            disable_content_filter=True,
            content_filter_provider="none",
        ),
    )
    record = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert record["content_filter"]["permission_used"] is True
    assert "cli_disable" in record["content_filter"]["permission_reasons"]


def test_admin_toml_can_grant_permission(tmp_path: Path) -> None:
    admin = tmp_path / "admin.toml"
    admin.write_text("allow_filter_disable = true\n", encoding="utf-8")
    consent = tmp_path / "consent.json"
    write_consent(consent)
    config = load_config(write_config(tmp_path, enabled=True))
    try:
        FaceSwapPipeline(config, admin_path=admin).run(
            _request(tmp_path, consent_file=consent, content_filter_provider="nudenet")
        )
    except ContentFilterError as exc:
        assert "nudenet" in str(exc)
    record = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert record["content_filter"]["permission_used"] is True


def test_nudenet_provider_requires_the_optional_package(tmp_path: Path) -> None:
    consent = tmp_path / "consent.json"
    write_consent(consent)
    with pytest.raises(ContentFilterError) as caught:
        _run(tmp_path, enabled=True, provider="nudenet", request=_request(tmp_path, consent_file=consent))
    assert "nudenet" in str(caught.value)


def test_disabled_master_switch_does_not_call_the_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(settings: FilterSettings):
        raise AssertionError(settings.provider)

    monkeypatch.setattr(content_filter_module, "create_filter", explode)
    result = _run(tmp_path, enabled=False, request=_request(tmp_path))
    assert result.output_path.is_file()
    assert t("compliance.disabled_note") in result.notes


def test_blocking_provider_stops_an_enabled_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Blocker:
        name = "blocker"
        supports_minor_detection = True

        def assess(self, image, *, role: str) -> FilterDecision:
            del image, role
            return FilterDecision(
                allowed=False,
                provider="blocker",
                reason="test block",
                nsfw_flag=True,
                minor_flag=False,
            )

    monkeypatch.setattr(content_filter_module, "create_filter", lambda settings: Blocker())
    consent = tmp_path / "consent.json"
    write_consent(consent)
    with pytest.raises(ContentBlockedError):
        _run(tmp_path, enabled=True, request=_request(tmp_path, consent_file=consent))
    record = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert record["event"] == "refused"


def test_cli_refuses_enable_compliance_without_consent(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    write_png(source)
    write_png(target)
    code = main(
        [
            "swap",
            "--source",
            str(source),
            "--target",
            str(target),
            "--output",
            str(tmp_path / "out.png"),
            "--config",
            str(write_config(tmp_path, enabled=False)),
            "--dry-run",
            "--enable-compliance",
        ]
    )
    assert code == 3


def test_stub_builder_allows_without_a_real_detector() -> None:
    settings = FilterSettings(
        enabled=True,
        provider="stub",
        threshold=0.6,
        block_on_minor=True,
        custom_entrypoint=None,
        configured_provider="stub",
        provider_changed=False,
        permission_used=False,
    )
    decision = create_filter(settings).assess(
        __import__("numpy").zeros((8, 8, 3), dtype="uint8"),
        role="source",
    )
    assert decision.allowed is True
    assert decision.provider == "stub"
