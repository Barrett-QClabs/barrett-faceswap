"""Run a BiSeNet-style face-parsing ONNX graph and keep selected regions.

Input pixels use ImageNet mean and standard deviation. Outputs are either
class scores (argmax over the class axis) or an integer label map. Pixels
whose label is in ``regions`` become 1. A missing file falls back to the
ellipse stub in :mod:`barrett_faceswap.masks`.
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
from barrett_faceswap.parsing import PARSING_REGIONS


def labels_from_output(array: np.ndarray) -> np.ndarray:
    """Reduce a parsing tensor to a 2-D integer label map."""

    data = np.asarray(array)
    if data.ndim == 4:
        data = data[0]
    if data.ndim == 3:
        if data.shape[0] == 1:
            return data[0].astype(np.int32)
        if data.shape[-1] == 1:
            return data[..., 0].astype(np.int32)
        # 19 is the CelebAMask-HQ class count. A tiny test crop can be smaller than that.
        if data.shape[0] in {19, 20} or (
            data.shape[0] <= 32 and data.shape[0] < min(data.shape[1], data.shape[2])
        ):
            return np.argmax(data, axis=0).astype(np.int32)
        if data.shape[-1] in {19, 20} or (
            data.shape[-1] <= 32 and data.shape[-1] < min(data.shape[0], data.shape[1])
        ):
            return np.argmax(data, axis=-1).astype(np.int32)
    if data.ndim == 2:
        return data.astype(np.int32)
    raise ProcessorError(t("masks.parser_rank", shape=tuple(int(dim) for dim in np.asarray(array).shape)))


def run_parser(
    session,
    crop_bgr: np.ndarray,
    regions: list[str],
    size: int,
    blur: int,
) -> np.ndarray:
    """Return a float mask of shape ``(size, size)`` for the selected regions."""

    network_h, network_w = session_hw(session, (size, size))
    channels = session_channels(session, 3)
    blob = bgr_to_nchw(crop_bgr, network_w, network_h, mode="imagenet", channels=channels)
    labels = labels_from_output(first_spatial_output(feed_session(session, blob)))
    selected = [PARSING_REGIONS[name] for name in regions if name in PARSING_REGIONS]
    if not selected:
        raise ProcessorError(t("masks.parser_regions"))
    mask = np.isin(labels, np.asarray(selected, dtype=np.int32)).astype(np.float32)
    if mask.shape != (size, size):
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return _blur(mask, blur)
