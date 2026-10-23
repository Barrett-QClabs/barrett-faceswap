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
    session_image_layout,
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

    layout, network_h, network_w, channels = session_image_layout(session, (size, size))
    blob = bgr_to_nchw(crop_bgr, network_w, network_h, mode="zero_one", channels=channels)
    if layout == "nhwc":
        # Channels-last graphs reject an NCHW feed with INVALID_ARGUMENT.
        blob = np.ascontiguousarray(np.transpose(blob, (0, 2, 3, 1)))
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


def refine_occlusion(session, crop_bgr: np.ndarray, coarse: np.ndarray, size: int) -> np.ndarray:
    """Run an operator-supplied SAM 2 style graph: image plus coarse mask in, mask out.

    The graph is not downloaded. Inputs are bound by rank. A 4-D tensor with one
    channel is the coarse mask. Any other 4-D tensor is the 0..1 RGB crop.
    """

    image = bgr_to_nchw(crop_bgr, size, size, mode="zero_one", channels=3)
    mask = np.asarray(coarse, dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[None, None, ...]
    elif mask.ndim == 3:
        mask = mask[None, ...]
    feeds: dict[str, np.ndarray] = {}
    for item in session.get_inputs():
        shape = list(item.shape)
        if len(shape) == 4 and (shape[1] == 1 or shape[-1] == 1):
            feeds[item.name] = mask if shape[1] == 1 or not isinstance(shape[1], int) else np.transpose(mask, (0, 2, 3, 1))
            if shape[-1] == 1 and shape[1] != 1:
                feeds[item.name] = np.transpose(mask, (0, 2, 3, 1))
        elif len(shape) == 4:
            feeds[item.name] = image
        else:
            feeds[item.name] = np.asarray([1.0], dtype=np.float32)
    data = np.squeeze(first_spatial_output(session.run(None, feeds)).astype(np.float32))
    if data.ndim == 3:
        data = data[0] if data.shape[0] <= 4 else data[..., 0]
    if data.ndim != 2:
        raise ProcessorError(t("masks.occlusion_rank", rank=int(data.ndim)))
    data = np.clip(data, 0.0, 1.0)
    if data.shape != (size, size):
        data = cv2.resize(data, (size, size), interpolation=cv2.INTER_LINEAR)
    return data.astype(np.float32)
