"""Message catalogs for English and Simplified Chinese.

Call :func:`activate_language` once at startup. :func:`t` then returns the
string for the active language. CLI help, logs, errors, compliance text, and
future Gradio labels all go through this module.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

SUPPORTED_LANGUAGES = ("en", "zh-CN")
_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_current = "en"


class LanguageError(ValueError):
    """The requested language code is not one of the shipped catalogs."""


def normalize_language(value: str | None) -> str | None:
    """Map ``en`` / ``zh-CN`` aliases onto a catalog name. Unknown codes return ``None``."""

    if value is None:
        return None
    key = str(value).strip().replace("_", "-").lower()
    if key == "":
        return None
    if key in {"en", "en-us", "en-gb"}:
        return "en"
    if key in {"zh", "zh-cn", "zh-hans"}:
        return "zh-CN"
    return None


def set_language(language: str) -> str:
    """Select the catalog used by :func:`t`."""

    global _current
    normalized = normalize_language(language)
    if normalized is None:
        raise LanguageError(t("config.bad_language", value=language))
    _current = normalized
    return _current


def get_language() -> str:
    return _current


def activate_language(*, cli: str | None, env: str | None, config: str | None) -> str:
    """Apply CLI, then environment, then config, then English."""

    for candidate in (cli, env, config):
        if candidate is None or str(candidate).strip() == "":
            continue
        normalized = normalize_language(str(candidate))
        if normalized is None:
            raise LanguageError(t("config.bad_language", value=candidate))
        return set_language(normalized)
    return set_language("en")


def t(key: str, **kwargs: object) -> str:
    """Translate ``key``. Extra keywords fill ``{placeholders}``."""

    catalogs = _catalogs()
    text = catalogs.get(_current, {}).get(key)
    if text is None:
        text = catalogs.get("en", {}).get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text


def catalog_keys(language: str) -> set[str]:
    """Flat key set for one catalog. Used by the parity test."""

    normalized = normalize_language(language)
    if normalized is None:
        raise LanguageError(t("config.bad_language", value=language))
    return set(_catalogs()[normalized])


def placeholders(language: str, key: str) -> set[str]:
    return set(_PLACEHOLDER.findall(_catalogs()[normalize_language(language) or "en"][key]))


@lru_cache(maxsize=1)
def _catalogs() -> dict[str, dict[str, str]]:
    root = Path(__file__).resolve().parent / "locales"
    loaded: dict[str, dict[str, str]] = {}
    for language, filename in (("en", "en.yaml"), ("zh-CN", "zh_CN.yaml")):
        path = root / filename
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RuntimeError(f"Locale file is not a mapping: {path}")
        loaded[language] = _flatten(raw)
    return loaded


def _flatten(node: dict, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in node.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, full))
        else:
            flat[full] = str(value)
    return flat
