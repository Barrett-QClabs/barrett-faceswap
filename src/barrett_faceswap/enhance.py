"""GFPGAN 1.4 and CodeFormer ONNX runners.

Both graphs are treated as square face restorers. The crop is normalized to
[-1, 1]. A second, non-image input is fed the CodeFormer weight when the graph
has one. GFPGAN exports usually have a single input; the weight is then unused.
The blend keeps a fraction of the original crop. Weights are never downloaded.
"""

from __future__ import annotations

import cv2
import numpy as np

from barrett_faceswap.onnx_io import (
    bgr_to_nchw,
    feed_session,
    first_spatial_output,
    nchw_to_bgr,
    session_channels,
    session_hw,
)


def blend_images(enhanced_bgr: np.ndarray, original_bgr: np.ndarray, amount: float) -> np.ndarray:
    """Mix ``enhanced_bgr`` into ``original_bgr``. ``amount`` 1 keeps the restoration."""

    factor = float(np.clip(amount, 0.0, 1.0))
    restored = enhanced_bgr
    if restored.shape[:2] != original_bgr.shape[:2]:
        restored = cv2.resize(
            restored,
            (original_bgr.shape[1], original_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    mixed = restored.astype(np.float32) * factor + original_bgr.astype(np.float32) * (1.0 - factor)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def run_enhancer(session, crop_bgr: np.ndarray, *, weight: float) -> np.ndarray:
    """Restore one aligned BGR crop and return a BGR crop of the same size."""

    height, width = crop_bgr.shape[:2]
    network_h, network_w = session_hw(session, (height, width))
    channels = session_channels(session, 3)
    blob = bgr_to_nchw(crop_bgr, network_w, network_h, mode="minus_one_one", channels=channels)
    outputs = feed_session(session, blob, weight)
    restored = nchw_to_bgr(first_spatial_output(outputs))
    if restored.shape[0] != height or restored.shape[1] != width:
        restored = cv2.resize(restored, (width, height), interpolation=cv2.INTER_LINEAR)
    return restored
