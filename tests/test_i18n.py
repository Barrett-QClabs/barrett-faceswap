"""Locale parity and language selection."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from barrett_faceswap.i18n import (
    LanguageError,
    activate_language,
    catalog_keys,
    get_language,
    placeholders,
    set_language,
    t,
)
from barrett_faceswap.model_registry import BUILTIN
from barrett_faceswap.ui import UI_KEYS, gradio_labels

_CALL = re.compile(r"""\bt\(\s*["']([A-Za-z0-9_.]+)["']""")
_SRC = Path(__file__).resolve().parents[1] / "src"


def test_every_key_exists_in_both_locales() -> None:
    english = catalog_keys("en")
    chinese = catalog_keys("zh-CN")
    assert english == chinese
    assert "compliance.disabled_note" in english
    assert "ui.source" in english
    for key in UI_KEYS:
        assert key in english
    for record in BUILTIN.values():
        assert record.license_key in english
    used = set()
    for path in _SRC.rglob("*.py"):
        used.update(_CALL.findall(path.read_text(encoding="utf-8")))
    missing = sorted(used - english)
    assert missing == []
    for key in english:
        assert placeholders("en", key) == placeholders("zh-CN", key)


def test_activate_prefers_cli_over_env_and_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BARRETT_FACESWAP_LANG", "en")
    assert activate_language(cli="zh-CN", env="en", config="en") == "zh-CN"
    assert get_language() == "zh-CN"
    assert "合规" in t("compliance.disabled_note")


def test_unknown_language_is_rejected() -> None:
    with pytest.raises(LanguageError):
        set_language("fr")


def test_gradio_labels_follow_the_active_language() -> None:
    set_language("en")
    assert gradio_labels()["ui.source"] == "Photo"
    assert gradio_labels()["ui.source_mode_video"] == "Video (face + voice, recommended)"
    set_language("zh-CN")
    assert gradio_labels()["ui.source"] == "照片"
    assert gradio_labels()["ui.source_mode_video"] == "视频（脸+声音，推荐）"
    assert set(gradio_labels()) == set(UI_KEYS)
