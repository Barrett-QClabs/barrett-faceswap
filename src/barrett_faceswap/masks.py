"""Paste masks. Phase 1 ships geometry stubs and merges them with a minimum.

``box_feather`` is an inset rectangle. ``region`` is an ellipse over the crop.
``occlusion`` is a stub of ones: no occlusion network is loaded, and the
minimum merge therefore leaves the other two masks unchanged. A later model
can return a real occlusion map without changing the paste path.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from barrett_faceswap.i18n import t

logger = logging.getLogger(__name__)


def box_feather_mask(size: int, blur: int, inset: float = 0.12) -> np.ndarray:
    """Inset box, then blur the edges."""

    mask = np.zeros((size, size), dtype=np.float32)
    margin = max(1, int(size * inset))
    mask[margin : size - margin, margin : size - margin] = 1.0
    return _blur(mask, blur)


def region_mask(size: int, blur: int) -> np.ndarray:
    """Elliptical face region inside the aligned crop."""

    mask = np.zeros((size, size), dtype=np.float32)
    pad = max(1, int(size * 0.08))
    axes = (max(1, size // 2 - pad), max(1, int(size * 0.48) - pad))
    cv2.ellipse(mask, (size // 2, size // 2), axes, 0, 0, 360, 1.0, thickness=-1)
    return _blur(mask, blur)


def occlusion_mask(size: int) -> np.ndarray:
    """Stub occlusion map. Every pixel is visible."""

    if not getattr(occlusion_mask, "_noted", False):
        logger.info(t("masks.occlusion_stub"))
        setattr(occlusion_mask, "_noted", True)
    return np.ones((size, size), dtype=np.float32)


def merge_min(masks: list[np.ndarray]) -> np.ndarray:
    """Element-wise minimum. A pixel survives only where every mask allows it."""

    if not masks:
        raise ValueError("merge_min requires at least one mask")
    stacked = np.stack([np.asarray(mask, dtype=np.float32) for mask in masks], axis=0)
    return np.min(stacked, axis=0)


def composite_mask(size: int, blur: int) -> np.ndarray:
    """Box, region, and occlusion, combined with :func:`merge_min`."""

    return merge_min(
        [
            box_feather_mask(size, blur),
            region_mask(size, blur),
            occlusion_mask(size),
        ]
    )


def _blur(mask: np.ndarray, blur: int) -> np.ndarray:
    kernel = int(blur)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        return cv2.GaussianBlur(mask, (kernel, kernel), 0)
    return mask
