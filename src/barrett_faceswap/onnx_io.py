"""Small ONNX Runtime helpers shared by the enhancer and the mask models.

Sessions are queried for rank and static spatial size. Names are not hard-coded.
"""

from __future__ import annotations

import cv2
import numpy as np


def static_dim(value: object) -> int | None:
    """Return a positive static dimension, or ``None`` when the axis is dynamic."""

    if isinstance(value, int) and value > 0:
        return value
    return None


def session_hw(session, fallback: tuple[int, int]) -> tuple[int, int]:
    """Height and width of the first 4-D input, or ``fallback`` when dynamic."""

    inputs = session.get_inputs()
    if not inputs:
        return fallback
    shape = list(inputs[0].shape)
    if len(shape) >= 4:
        height = static_dim(shape[-2])
        width = static_dim(shape[-1])
        if height is not None and width is not None:
            return height, width
    return fallback


def tensor_layout(shape: list) -> str:
    """``nhwc`` when the last axis is 1 or 3 channels and axis 1 is not.

    facefusion ``xseg_1.onnx`` is ``[N, 256, 256, 3]``. A channels-first graph
    keeps the channel count on axis 1, including a dynamic spatial size.
    """

    if len(shape) == 4 and shape[-1] in (1, 3) and shape[1] not in (1, 3):
        return "nhwc"
    return "nchw"


def session_image_layout(session, fallback: tuple[int, int]) -> tuple[str, int, int, int]:
    """Layout, height, width, and channels for the first image input."""

    inputs = session.get_inputs()
    shape = list(inputs[0].shape) if inputs else []
    layout = tensor_layout(shape)
    if layout == "nhwc":
        height = static_dim(shape[1]) or fallback[0]
        width = static_dim(shape[2]) or fallback[1]
        channels = int(shape[-1])
        return layout, int(height), int(width), channels
    height, width = session_hw(session, fallback)
    return layout, height, width, session_channels(session, 3)


def session_channels(session, fallback: int = 3) -> int:
    """Channel count of the first 4-D input."""

    inputs = session.get_inputs()
    if not inputs:
        return fallback
    shape = list(inputs[0].shape)
    if len(shape) >= 4:
        channels = static_dim(shape[1] if len(shape) == 4 else shape[-3])
        if channels is not None:
            return channels
    return fallback


def bgr_to_nchw(
    image_bgr: np.ndarray,
    width: int,
    height: int,
    *,
    mode: str,
    channels: int,
) -> np.ndarray:
    """Resize a BGR image and normalize it to a 1xCxHxW float32 tensor.

    ``mode`` is ``minus_one_one`` (GFPGAN / CodeFormer), ``zero_one`` (occlusion),
    or ``imagenet`` (face parsing).
    """

    resized = cv2.resize(image_bgr, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    if channels == 1:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        rgb = gray[:, :, None]
    if mode == "minus_one_one":
        rgb = (rgb / 255.0 - 0.5) / 0.5
    elif mode == "zero_one":
        rgb = rgb / 255.0
    elif mode == "imagenet":
        rgb = rgb / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        if channels == 1:
            mean = np.array([float(mean.mean())], dtype=np.float32)
            std = np.array([float(std.mean())], dtype=np.float32)
        rgb = (rgb - mean) / std
    else:
        raise ValueError(f"unknown normalization mode: {mode}")
    return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)


def fit_scalar(value: float, shape: object) -> np.ndarray:
    """Broadcast a scalar into the static shape of a side input."""

    if not isinstance(shape, (list, tuple)) or not shape:
        return np.asarray([value], dtype=np.float32)
    dims: list[int] = []
    for dim in shape:
        parsed = static_dim(dim)
        dims.append(parsed if parsed is not None else 1)
    return np.full(dims, float(value), dtype=np.float32)


def feed_session(session, image_nchw: np.ndarray, weight: float | None = None):
    """Run ``session``. 4-D inputs get the image. Other inputs get ``weight``."""

    feeds: dict[str, np.ndarray] = {}
    for item in session.get_inputs():
        shape = list(item.shape)
        if len(shape) == 4:
            feeds[item.name] = image_nchw
        else:
            feeds[item.name] = fit_scalar(0.5 if weight is None else weight, shape)
    return session.run(None, feeds)


def first_spatial_output(outputs: list) -> np.ndarray:
    """Pick the first tensor that looks like an image or a label map."""

    for item in outputs:
        array = np.asarray(item)
        if array.ndim >= 2:
            return array
    return np.asarray(outputs[0])


def nchw_to_bgr(tensor: np.ndarray) -> np.ndarray:
    """Convert a network image to uint8 BGR.

    Values that go below zero are treated as [-1, 1]. Values above 1.5 are
    treated as 0..255. Anything else is treated as 0..1.
    """

    data = np.asarray(tensor, dtype=np.float32)
    if data.ndim == 4:
        data = data[0]
    if data.ndim == 3 and data.shape[0] in {1, 3}:
        data = np.transpose(data, (1, 2, 0))
    if data.ndim == 2:
        data = data[:, :, None]
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=-1)
    finite = data[np.isfinite(data)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 1.0
    if low < -0.01:
        data = (data + 1.0) / 2.0
    elif high > 1.5:
        data = data / 255.0
    data = np.clip(data, 0.0, 1.0)
    rgb = (data * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
