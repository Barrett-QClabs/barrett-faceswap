"""Locale parity and language selection."""

from __future__ import annotations

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
from barrett_faceswap.ui import UI_KEYS, gradio_labels


def test_every_key_exists_in_both_locales() -> None:
    english = catalog_keys("en")
    chinese = catalog_keys("zh-CN")
    assert english == chinese
    assert "compliance.disabled_note" in english
    assert "ui.source" in english
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
    assert gradio_labels()["ui.source"] == "Source"
    set_language("zh-CN")
    assert gradio_labels()["ui.source"] == "源图像"
    assert set(gradio_labels()) == set(UI_KEYS)
