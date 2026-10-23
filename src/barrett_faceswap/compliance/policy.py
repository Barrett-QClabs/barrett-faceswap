"""Central compliance gate.

:class:`FaceSwapPipeline` calls :meth:`before_process` and :meth:`after_process`
on every run. When ``compliance.enabled`` is false and the CLI did not pass
``--enable-compliance``, both methods skip consent, watermark, filter blocks,
and the audit log, and they leave a single debug note.

When enforcement is on, weakening the content filter requires
``BARRETT_FACESWAP_ALLOW_FILTER_DISABLE=1`` or ``configs/admin.toml``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from barrett_faceswap.compliance import audit_log, content_filter
from barrett_faceswap.compliance.audit_log import append_event, hash_file
from barrett_faceswap.compliance.consent import (
    ConsentRecord,
    consent_from_acknowledgement,
    load_consent_file,
)
from barrett_faceswap.compliance.content_filter import (
    FilterDecision,
    FilterOverrides,
    FilterSettings,
    resolve_filter_settings,
)
from barrett_faceswap.compliance.watermark import apply_video_metadata, apply_visible_watermark, save_image
from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import (
    BarrettFaceswapError,
    ComplianceError,
    ConsentError,
    ContentBlockedError,
)
from barrett_faceswap.i18n import t
from barrett_faceswap.types import SwapRequest, SwapResult
from barrett_faceswap.vision import is_video, read_image

logger = logging.getLogger(__name__)


@dataclass
class ComplianceContext:
    """State carried from the pre-check to the post-check."""

    enabled: bool
    request: SwapRequest
    notes: list[str] = field(default_factory=list)
    audited: bool = False
    consent: ConsentRecord | None = None
    filter_settings: FilterSettings | None = None
    watermark_enabled: bool = False
    watermark_text: str = ""
    watermark_disabled: bool = False
    source_sha256: str | None = None
    target_sha256: str | None = None
    decisions: list[dict] = field(default_factory=list)
    checker: object | None = None


class CompliancePolicy:
    """Enforce or explicitly skip the compliance profile."""

    def __init__(self, config: AppConfig, admin_path: Path | None = None) -> None:
        self.config = config
        self.admin_path = admin_path if admin_path is not None else Path("configs/admin.toml")

    def enforcement_enabled(self, request: SwapRequest) -> bool:
        return bool(self.config.compliance.enabled or request.enable_compliance)

    def before_process(
        self,
        request: SwapRequest,
        *,
        target_samples: Sequence[np.ndarray] | None = None,
    ) -> ComplianceContext:
        if not self.enforcement_enabled(request):
            note = t("compliance.disabled_note")
            logger.warning(note)
            return ComplianceContext(enabled=False, request=request, notes=[note])
        context = ComplianceContext(
            enabled=True,
            request=request,
            notes=[t("compliance.enabled_note")],
            source_sha256=hash_file(request.source_path),
            target_sha256=hash_file(request.target_path),
        )
        try:
            context.consent = self._consent(request)
            context.filter_settings = self._filter_settings(request)
            context.watermark_enabled, context.watermark_text, context.watermark_disabled = (
                self._watermark(request, context)
            )
            self._screen_inputs(request, context, target_samples=target_samples)
        except BarrettFaceswapError as exc:
            self._write(context, "refused", reason=str(exc))
            raise
        return context

    def after_process(self, context: ComplianceContext, frame, request: SwapRequest) -> SwapResult:
        if not context.enabled:
            save_image(frame, request.output_path, watermark=False, text="")
            return SwapResult(
                output_path=request.output_path,
                dry_run=request.dry_run,
                compliance_enabled=False,
                notes=list(context.notes),
            )
        decision = self._assess(frame, context, role="output")
        if not decision.allowed or (decision.minor_flag and context.filter_settings and context.filter_settings.block_on_minor):
            reason = t("filter.blocked", role="output", reason=decision.reason)
            self._write(context, "refused", reason=reason)
            raise ContentBlockedError(reason)
        save_image(
            frame,
            request.output_path,
            watermark=context.watermark_enabled,
            text=context.watermark_text,
            font_path=self.config.compliance.watermark.font_path,
        )
        self._write(context, "completed", reason=None)
        return SwapResult(
            output_path=request.output_path,
            dry_run=request.dry_run,
            compliance_enabled=True,
            notes=list(context.notes),
        )

    def screen_frame(self, context: ComplianceContext, frame: np.ndarray, *, role: str) -> None:
        """Filter one frame. No-op while enforcement is off. A block is audited as refused."""

        if not context.enabled:
            return
        decision = self._assess(frame, context, role=role)
        if decision.minor_flag and context.filter_settings and context.filter_settings.block_on_minor:
            reason = t("filter.minor_blocked", role=role)
            self._write(context, "refused", reason=reason)
            raise ContentBlockedError(reason)
        if not decision.allowed:
            reason = t("filter.blocked", role=role, reason=decision.reason)
            self._write(context, "refused", reason=reason)
            raise ContentBlockedError(reason)

    def mark_frame(self, context: ComplianceContext, frame: np.ndarray) -> np.ndarray:
        """Watermark one frame when enforcement and the watermark are both on."""

        if not context.enabled or not context.watermark_enabled:
            return frame
        return apply_visible_watermark(
            frame,
            context.watermark_text or t("watermark.label"),
            font_path=self.config.compliance.watermark.font_path,
        )

    def complete_output(self, context: ComplianceContext, request: SwapRequest) -> SwapResult:
        """Audit a file the caller already wrote. Used by the video encoder."""

        if not context.enabled:
            return SwapResult(
                output_path=request.output_path,
                dry_run=request.dry_run,
                compliance_enabled=False,
                notes=list(context.notes),
            )
        if context.watermark_enabled and not request.live_session:
            marked = apply_video_metadata(request.output_path, context.watermark_text or t("watermark.label"))
            context.notes.append(marked.detail)
        elif context.watermark_enabled and request.live_session:
            context.notes.append(t("live.session_marked"))
        self._write(context, "completed", reason=None)
        return SwapResult(
            output_path=request.output_path,
            dry_run=request.dry_run,
            compliance_enabled=True,
            notes=list(context.notes),
        )

    def abort(self, context: ComplianceContext, exc: BaseException) -> None:
        if not context.enabled or context.audited:
            return
        self._write(context, "failed", reason=str(exc))

    def _consent(self, request: SwapRequest) -> ConsentRecord:
        del self
        has_file = request.consent_file is not None
        has_ack = request.acknowledge_authorized_use
        if has_file and has_ack:
            raise ConsentError(t("consent.both"))
        if has_file and request.consent_file is not None:
            return load_consent_file(request.consent_file)
        if has_ack:
            return consent_from_acknowledgement(request.consent_subject, request.consent_statement)
        raise ConsentError(t("consent.need_one"))

    def _filter_settings(self, request: SwapRequest) -> FilterSettings:
        profile = self.config.compliance.content_filter
        overrides = FilterOverrides(
            disable=request.disable_content_filter,
            provider=request.content_filter_provider,
            threshold=request.content_filter_threshold,
            block_on_minor=False if request.no_block_on_minor else None,
        )
        return resolve_filter_settings(
            enabled=profile.enabled,
            provider=profile.provider,
            threshold=profile.threshold,
            block_on_minor=profile.block_on_minor,
            custom_entrypoint=profile.custom_entrypoint,
            overrides=overrides,
            permission=self.filter_override_permitted(),
        )

    def _watermark(self, request: SwapRequest, context: ComplianceContext) -> tuple[bool, str, bool]:
        configured = self.config.compliance.watermark.enabled
        text = self.config.compliance.watermark.text or t("watermark.label")
        if request.no_watermark:
            context.notes.append(t("watermark.disabled_logged"))
            return False, text, True
        if not configured:
            raise ComplianceError(t("watermark.config_blocked"))
        return True, text, False

    def _screen_inputs(
        self,
        request: SwapRequest,
        context: ComplianceContext,
        *,
        target_samples: Sequence[np.ndarray] | None,
    ) -> None:
        self._reject_frame(read_image(request.source_path), context, role="source")
        if request.live_session:
            if not target_samples:
                raise ComplianceError(t("compliance.live_samples_missing"))
            for index, frame in enumerate(target_samples):
                self._reject_frame(frame, context, role=f"live[{index}]")
            return
        if is_video(request.target_path):
            if not target_samples:
                raise ComplianceError(t("compliance.video_samples_missing"))
            for index, frame in enumerate(target_samples):
                self._reject_frame(frame, context, role=f"target[{index}]")
            return
        self._reject_frame(read_image(request.target_path), context, role="target")

    def _reject_frame(self, image: np.ndarray, context: ComplianceContext, *, role: str) -> None:
        decision = self._assess(image, context, role=role)
        if decision.minor_flag and context.filter_settings and context.filter_settings.block_on_minor:
            raise ContentBlockedError(t("filter.minor_blocked", role=role))
        if not decision.allowed:
            raise ContentBlockedError(t("filter.blocked", role=role, reason=decision.reason))

    def _assess(self, image, context: ComplianceContext, *, role: str) -> FilterDecision:
        settings = context.filter_settings
        if settings is None:
            raise ComplianceError(t("filter.none_reason"))
        if context.checker is None:
            context.checker = content_filter.create_filter(settings)
        checker = context.checker
        decision = checker.assess(image, role=role)
        context.decisions.append(
            {
                "role": role,
                "allowed": decision.allowed,
                "provider": decision.provider,
                "reason": decision.reason,
                "nsfw_flag": decision.nsfw_flag,
                "minor_flag": decision.minor_flag,
                "minor_detection": decision.minor_detection,
            }
        )
        return decision

    def filter_override_permitted(self) -> bool:
        if os.environ.get("BARRETT_FACESWAP_ALLOW_FILTER_DISABLE") == "1":
            return True
        return _toml_flag(self.admin_path)

    def _write(self, context: ComplianceContext, event: str, *, reason: str | None) -> None:
        if not self.config.compliance.audit.enabled:
            context.audited = True
            return
        settings = context.filter_settings
        output = context.request.output_path
        output_hash = hash_file(output) if event == "completed" else None
        append_event(
            self.config.compliance.audit.path,
            {
                "event": event,
                "compliance_enabled": True,
                "source": str(context.request.source_path),
                "target": str(context.request.target_path),
                "output_path": str(output) if event == "completed" else None,
                "source_sha256": context.source_sha256,
                "target_sha256": context.target_sha256,
                "output_sha256": output_hash,
                "consent_id": context.consent.consent_id if context.consent else None,
                "consent_subject": context.consent.subject_id if context.consent else None,
                "consent_scope": context.consent.scope if context.consent else None,
                "dry_run": context.request.dry_run,
                "watermark_applied": bool(context.watermark_enabled and event == "completed"),
                "watermark_disabled": context.watermark_disabled,
                "content_filter": {
                    "enabled": settings.enabled if settings else None,
                    "provider": settings.provider if settings else None,
                    "configured_provider": settings.configured_provider if settings else None,
                    "threshold": settings.threshold if settings else None,
                    "block_on_minor": settings.block_on_minor if settings else None,
                    "provider_changed": settings.provider_changed if settings else False,
                    "permission_used": settings.permission_used if settings else False,
                    "permission_reasons": list(settings.permission_reasons) if settings else [],
                    "decisions": context.decisions,
                },
                "reason": reason,
            },
        )
        context.audited = True


def _toml_flag(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None
    if tomllib is not None:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        return bool(data.get("allow_filter_disable", False))
    for line in path.read_text(encoding="utf-8").splitlines():
        body = line.split("#", 1)[0].strip()
        if body.startswith("allow_filter_disable"):
            value = body.partition("=")[2].strip().lower()
            return value in {"1", "true", "yes"}
    return False
