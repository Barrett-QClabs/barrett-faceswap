"""Pluggable content-filter providers.

The shipped default provider is ``stub``: it allows every frame and logs that
no real detector is loaded. ``nudenet`` is constructed only when that provider
is selected and the optional package is installed. ``none`` and a disabled
filter are permission-gated weakenings. See ``docs/COMPLIANCE.md``.

Providers are defense in depth. They do not replace a consent record.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from barrett_faceswap.errors import CompliancePermissionError, ContentFilterError
from barrett_faceswap.i18n import t

logger = logging.getLogger(__name__)

PROVIDER_NAMES = ("none", "stub", "nudenet", "custom")

# Labels treated as a block signal when a NudeNet-style detector is opt-in.
_NUDENET_BLOCK_LABELS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "EXPOSED_BREAST_F",
    "EXPOSED_GENITALIA_F",
    "EXPOSED_GENITALIA_M",
    "EXPOSED_ANUS",
    "EXPOSED_BUTTOCKS",
}


@dataclass(frozen=True)
class FilterSettings:
    """Effective filter settings for one run."""

    enabled: bool
    provider: str
    threshold: float
    block_on_minor: bool
    custom_entrypoint: str | None
    configured_provider: str
    provider_changed: bool
    permission_used: bool
    permission_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class FilterDecision:
    """One provider judgement for a single image."""

    allowed: bool
    provider: str
    reason: str
    categories: dict[str, float] = field(default_factory=dict)
    nsfw_flag: bool = False
    minor_flag: bool = False
    minor_detection: str = "unavailable"


class ContentFilter(Protocol):
    """Backend used by the policy. Implementations must be safe to construct lazily."""

    name: str
    supports_minor_detection: bool

    def assess(self, image_bgr: np.ndarray, *, role: str) -> FilterDecision:
        """Return a decision. Raising aborts the run."""


@dataclass(frozen=True)
class FilterOverrides:
    """CLI requests applied on top of the YAML profile."""

    disable: bool = False
    provider: str | None = None
    threshold: float | None = None
    block_on_minor: bool | None = None


def resolve_filter_settings(
    *,
    enabled: bool,
    provider: str,
    threshold: float,
    block_on_minor: bool,
    custom_entrypoint: str | None,
    overrides: FilterOverrides,
    permission: bool,
) -> FilterSettings:
    """Apply CLI overrides. Weakenings require ``permission``.

    A change of provider relative to the configured value also requires
    permission, including a change toward a stricter backend. Edit the YAML
    to opt into ``nudenet`` or ``custom`` without the environment switch.
    """

    reasons: list[str] = []
    next_enabled = enabled
    next_provider = provider
    next_threshold = threshold
    next_block = block_on_minor
    provider_changed = False

    if not enabled:
        reasons.append("config_disabled")
    if provider == "none":
        reasons.append("config_provider_none")
    if not block_on_minor:
        reasons.append("config_block_on_minor_false")
    if overrides.disable:
        next_enabled = False
        reasons.append("cli_disable")
    if overrides.provider is not None and overrides.provider != provider:
        if overrides.provider not in PROVIDER_NAMES:
            allowed = ", ".join(PROVIDER_NAMES)
            raise ContentFilterError(t("filter.unknown_provider", provider=overrides.provider, allowed=allowed))
        next_provider = overrides.provider
        provider_changed = True
        reasons.append("cli_provider_change")
    if overrides.threshold is not None:
        if not 0.0 <= overrides.threshold <= 1.0:
            raise ContentFilterError(t("filter.threshold"))
        if overrides.threshold > threshold:
            reasons.append("cli_threshold_increase")
        next_threshold = overrides.threshold
    if overrides.block_on_minor is not None and overrides.block_on_minor != block_on_minor:
        if block_on_minor and not overrides.block_on_minor:
            reasons.append("cli_block_on_minor_disabled")
        next_block = overrides.block_on_minor
    if next_provider == "none" or not next_enabled:
        # Already recorded when the config or the CLI asked for it.
        pass
    if next_provider == "custom" and not (custom_entrypoint and custom_entrypoint.strip()):
        raise ContentFilterError(t("filter.custom_empty"))
    unique_reasons = tuple(dict.fromkeys(reasons))
    if unique_reasons and not permission:
        raise CompliancePermissionError(t("filter.permission", reasons=", ".join(unique_reasons)))
    return FilterSettings(
        enabled=next_enabled,
        provider=next_provider,
        threshold=next_threshold,
        block_on_minor=next_block,
        custom_entrypoint=custom_entrypoint,
        configured_provider=provider,
        provider_changed=provider_changed,
        permission_used=bool(unique_reasons and permission),
        permission_reasons=unique_reasons,
    )


_BUILDERS: dict[str, object] = {}


def register_provider(name: str):
    """Register a ``FilterSettings -> ContentFilter`` builder."""

    def decorator(builder):
        _BUILDERS[name] = builder
        return builder

    return decorator


def unregister_provider(name: str) -> None:
    """Remove a test provider. Built-in names can be restored by re-importing."""

    _BUILDERS.pop(name, None)


def create_filter(settings: FilterSettings) -> ContentFilter:
    """Construct the provider named by ``settings``. Import side effects stay local."""

    if settings.provider == "none" or not settings.enabled:
        return _DisabledFilter()
    try:
        builder = _BUILDERS[settings.provider]
    except KeyError as exc:
        known = ", ".join(sorted(_BUILDERS))
        raise ContentFilterError(t("filter.unregistered", provider=settings.provider, known=known)) from exc
    return builder(settings)


class _DisabledFilter:
    name = "none"
    supports_minor_detection = False

    def assess(self, image_bgr: np.ndarray, *, role: str) -> FilterDecision:
        del image_bgr, role
        return FilterDecision(
            allowed=True,
            provider=self.name,
            reason=t("filter.none_reason"),
            minor_detection="unavailable",
        )


class StubContentFilter:
    """Allows every frame and says so. This is not a safety classifier."""

    name = "stub"
    supports_minor_detection = False

    def __init__(self) -> None:
        self._warned = False

    def assess(self, image_bgr: np.ndarray, *, role: str) -> FilterDecision:
        del image_bgr
        if not self._warned:
            logger.warning(t("filter.stub_warning"))
            self._warned = True
        return FilterDecision(
            allowed=True,
            provider=self.name,
            reason=t("filter.stub_reason", role=role),
            minor_detection="unavailable",
        )


class NudeNetContentFilter:
    """Optional backend. The ``nudenet`` package is imported only from this class."""

    name = "nudenet"
    supports_minor_detection = False

    def __init__(self, settings: FilterSettings) -> None:
        self.threshold = settings.threshold
        self._detector = _load_nudenet()

    def assess(self, image_bgr: np.ndarray, *, role: str) -> FilterDecision:
        import tempfile
        from pathlib import Path

        import cv2

        categories: dict[str, float] = {}
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temp_path = handle.name
        try:
            if not cv2.imwrite(temp_path, image_bgr):
                raise ContentFilterError(t("filter.nudenet_write"))
            raw = self._detector.detect(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
        if not isinstance(raw, list):
            raise ContentFilterError(t("filter.nudenet_shape"))
        flagged = False
        peak = 0.0
        for item in raw:
            if not isinstance(item, dict):
                continue
            label = str(item.get("class") or item.get("label") or "")
            score = float(item.get("score", 0.0))
            categories[label] = max(categories.get(label, 0.0), score)
            if label in _NUDENET_BLOCK_LABELS and score >= self.threshold:
                flagged = True
                peak = max(peak, score)
        if flagged:
            return FilterDecision(
                allowed=False,
                provider=self.name,
                reason=t("filter.nudenet_block", role=role, score=f"{peak:.3f}", threshold=f"{self.threshold:.3f}"),
                categories=categories,
                nsfw_flag=True,
                minor_detection="unavailable",
            )
        return FilterDecision(
            allowed=True,
            provider=self.name,
            reason=t("filter.nudenet_allow", role=role, threshold=f"{self.threshold:.3f}"),
            categories=categories,
            minor_detection="unavailable",
        )


def _load_nudenet():
    try:
        from nudenet import NudeDetector
    except ImportError as exc:
        raise ContentFilterError(t("filter.nudenet_missing")) from exc
    return NudeDetector()


def _build_custom(settings: FilterSettings) -> ContentFilter:
    entrypoint = settings.custom_entrypoint or ""
    module_name, separator, qualname = entrypoint.partition(":")
    if not separator or not module_name or not qualname:
        raise ContentFilterError(t("filter.custom_form"))
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, qualname)
    except (ImportError, AttributeError) as exc:
        raise ContentFilterError(t("filter.custom_load", entrypoint=entrypoint, error=exc)) from exc
    instance = factory(settings)
    if not hasattr(instance, "assess"):
        raise ContentFilterError(t("filter.custom_assess", entrypoint=entrypoint))
    return instance


@register_provider("stub")
def _build_stub(settings: FilterSettings) -> ContentFilter:
    del settings
    return StubContentFilter()


@register_provider("nudenet")
def _build_nudenet(settings: FilterSettings) -> ContentFilter:
    return NudeNetContentFilter(settings)


@register_provider("custom")
def _build_custom_provider(settings: FilterSettings) -> ContentFilter:
    return _build_custom(settings)
