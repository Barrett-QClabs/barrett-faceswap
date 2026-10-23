"""Catalog labels for the Gradio shell.

:func:`gradio_labels` reads the active language. The app in ``ui.app`` uses
the same keys and does not import Gradio until ``barrett-faceswap ui`` runs.
"""

from __future__ import annotations

from barrett_faceswap.i18n import t

UI_KEYS = (
    "ui.source",
    "ui.target",
    "ui.output",
    "ui.processors",
    "ui.execution",
    "ui.preview",
    "ui.preview_image",
    "ui.run",
    "ui.consent",
    "ui.consent_subject",
    "ui.watermark",
    "ui.language",
    "ui.lang_en",
    "ui.lang_zh",
    "ui.dry_run",
    "ui.compliance",
    "ui.detector",
    "ui.swapper",
    "ui.enhancer",
    "ui.enhancer_blend",
    "ui.enhancer_weight",
    "ui.swapper_model",
    "ui.pixel_boost",
    "ui.many_faces",
    "ui.reference",
    "ui.face_map",
    "ui.blend_mode",
    "ui.color_transfer",
    "ui.video_mode",
    "ui.video_encoder",
    "ui.trim_start",
    "ui.trim_end",
    "ui.keep_audio",
    "ui.mask",
    "ui.audit",
    "ui.threshold",
    "ui.title",
    "ui.intro",
    "ui.status",
    "ui.need_files",
    "ui.need_processor",
    "ui.done",
    "ui.preview_still",
    "ui.preview_video",
    "ui.gradio_missing",
    "ui.control_mismatch",
)


def gradio_labels() -> dict[str, str]:
    """Return translated captions keyed by the catalog name."""

    return {key: t(key) for key in UI_KEYS}
