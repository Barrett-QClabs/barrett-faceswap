"""xseg-style occlusion mask from an operator-supplied ONNX file.

The graph receives a resized crop in 0..1 RGB (or a single gray channel when
that is the static input rank). The output is one channel. Higher values mean
the pixel stays visible. ``invert`` flips graphs that predict the occluder
instead of the face. A missing file is not an error: the caller keeps the
all-ones stub.
"""

from __future__ import annotations

import cv2
import numpy as np

from barrett_faceswap.errors import ProcessorError
from barrett_faceswap.i18n import t
from barrett_faceswap.masks import _blur
from barrett_faceswap.onnx_io import (
    bgr_to_nchw,
    feed_session,
    first_spatial_output,
    session_channels,
    session_hw,
)


def run_occlusion(
    session,
    crop_bgr: np.ndarray,
    size: int,
    blur: int,
    *,
    invert: bool,
) -> np.ndarray:
    """Return a float mask of shape ``(size, size)`` in 0..1."""

    network_h, network_w = session_hw(session, (size, size))
    channels = session_channels(session, 3)
    blob = bgr_to_nchw(crop_bgr, network_w, network_h, mode="zero_one", channels=channels)
    data = np.squeeze(first_spatial_output(feed_session(session, blob)).astype(np.float32))
    if data.ndim == 3:
        data = data[0] if data.shape[0] <= 4 else data[..., 0]
    if data.ndim != 2:
        raise ProcessorError(t("masks.occlusion_rank", rank=int(data.ndim)))
    finite = data[np.isfinite(data)]
    if finite.size and float(finite.max()) > 1.5:
        data = data / 255.0
    data = np.clip(data, 0.0, 1.0)
    if invert:
        data = 1.0 - data
    if data.shape != (size, size):
        data = cv2.resize(data, (size, size), interpolation=cv2.INTER_LINEAR)
    return _blur(data.astype(np.float32), blur)
