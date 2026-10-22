"""Detect, align, swap, and paste one face."""

from __future__ import annotations

import numpy as np

from barrett_faceswap.blend import match_color, paste_back
from barrett_faceswap.i18n import t
from barrett_faceswap.masks import composite_mask
from barrett_faceswap.config import AppConfig
from barrett_faceswap.detect import FaceDetector, select_face
from barrett_faceswap.align import align_face
from barrett_faceswap.models import ModelStore
from barrett_faceswap.processors.registry import register_processor
from barrett_faceswap.swap import IdentitySwapper
from barrett_faceswap.types import ProcessContext


class FaceSwapper:
    """Default processor. Dry-runs skip the model store entirely."""

    name = "face_swapper"

    def __init__(self, config: AppConfig, providers: list[str]) -> None:
        self.config = config
        self.providers = providers
        self.store: ModelStore | None = None

    def pre_check(self, ctx: ProcessContext) -> None:
        if ctx.dry_run:
            ctx.notes.append(t("processor.swapper_dry_run"))
            return
        self.store = ModelStore(self.config, self.providers)
        self.store.ensure_present()

    def process_frame(self, frame: np.ndarray, ctx: ProcessContext) -> np.ndarray:
        if ctx.dry_run:
            return frame
        if self.store is None:
            self.pre_check(ctx)
        assert self.store is not None
        detector = FaceDetector(
            self.store.session("detector"),
            det_size=self.config.detect.det_size,
            score_threshold=self.config.detect.score_threshold,
            nms_threshold=self.config.detect.nms_threshold,
        )
        source_face = select_face(detector.detect(ctx.source_bgr), 0)
        target_face = select_face(detector.detect(frame), ctx.target_face_index)
        resolution = self.config.swap.resolution
        crop, matrix = align_face(frame, target_face.landmarks, resolution)
        swapper = IdentitySwapper(self.store, resolution)
        embedding = swapper.embed(ctx.source_bgr, source_face.landmarks)
        swapped = swapper.forward(crop, embedding)
        mask = composite_mask(resolution, self.config.blend.mask_blur)
        matched = match_color(swapped, crop, mask)
        ctx.notes.append(
            t(
                "processor.swapper_done",
                source_score=f"{source_face.score:.3f}",
                target_score=f"{target_face.score:.3f}",
                index=ctx.target_face_index,
            )
        )
        return paste_back(frame, matched, matrix, mask)

    def post_process(self) -> None:
        return None


@register_processor("face_swapper")
def _build(config: AppConfig, providers: list[str]) -> FaceSwapper:
    return FaceSwapper(config, providers)
