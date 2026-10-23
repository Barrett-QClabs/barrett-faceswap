"""Reserved hook for a later GFPGAN or CodeFormer stage.

The class is registered so the chain can list it. A real run refuses to pretend
the enhancer ran. A dry-run records a note and returns the frame unchanged.
"""

from __future__ import annotations

import numpy as np

from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import ProcessorError
from barrett_faceswap.i18n import t
from barrett_faceswap.processors.registry import register_processor
from barrett_faceswap.types import ProcessContext


class FaceEnhancer:
    """Phase 2 placeholder. It does not load an enhancer graph."""

    name = "face_enhancer"

    def pre_check(self, ctx: ProcessContext) -> None:
        if ctx.dry_run:
            ctx.notes.append(t("processor.enhancer_dry_run"))
            return
        raise ProcessorError(t("processor.enhancer_missing"))

    def process_frame(self, frame: np.ndarray, ctx: ProcessContext) -> np.ndarray:
        return frame

    def post_process(self) -> None:
        return None


@register_processor("face_enhancer")
def _build(config: AppConfig, providers: list[str]) -> FaceEnhancer:
    del config, providers
    return FaceEnhancer()
