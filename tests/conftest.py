"""Shared fixtures. Compliance tests opt in; the shipped default stays off."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from barrett_faceswap.i18n import set_language

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_language_and_permission(monkeypatch: pytest.MonkeyPatch):
    set_language("en")
    monkeypatch.delenv("BARRETT_FACESWAP_ALLOW_FILTER_DISABLE", raising=False)
    monkeypatch.delenv("BARRETT_FACESWAP_LANG", raising=False)
    yield
    set_language("en")


def write_png(path: Path, color: tuple[int, int, int] = (0, 0, 0), size: tuple[int, int] = (320, 240)) -> None:
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = color
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {path}")


def write_consent(path: Path, **overrides: object) -> None:
    record = {
        "subject_id": "synthetic-demo",
        "statement": "Synthetic test pattern only. No real person is depicted.",
        "consented_on": "2026-01-01",
        "scope": "unit-test",
        "expires_on": None,
    }
    record.update(overrides)
    path.write_text(json.dumps(record), encoding="utf-8")


def write_config(directory: Path, *, enabled: bool, provider: str = "stub") -> Path:
    audit = directory / "audit.jsonl"
    models = directory / "weights"
    text = f"""
ui:
  language: en
models:
  detector:
    path: {models / "det_10g.onnx"}
    sha256: null
  recognizer:
    path: {models / "w600k_r50.onnx"}
    sha256: null
  swapper:
    path: {models / "inswapper_128.onnx"}
    sha256: null
compliance:
  enabled: {"true" if enabled else "false"}
  content_filter:
    enabled: true
    provider: {provider}
    threshold: 0.6
    block_on_minor: true
  audit:
    enabled: true
    path: {audit}
"""
    path = directory / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path
