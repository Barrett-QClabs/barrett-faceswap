"""Labels for a future Gradio shell.

Phase 1 does not import Gradio. When a UI arrives, it should call
:func:`gradio_labels` instead of hard-coding captions. The strings live in
the ``ui.*`` catalog entries.
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
    "ui.run",
    "ui.consent",
    "ui.watermark",
    "ui.language",
    "ui.dry_run",
    "ui.compliance",
    "ui.detector",
    "ui.swapper",
    "ui.enhancer",
    "ui.mask",
    "ui.audit",
    "ui.threshold",
)


def gradio_labels() -> dict[str, str]:
    """Return translated captions keyed by the catalog name."""

    return {key: t(key) for key in UI_KEYS}
