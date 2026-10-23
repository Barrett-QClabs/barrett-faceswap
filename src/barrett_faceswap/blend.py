"""Paste a swapped crop back onto the original frame."""

from __future__ import annotations

import cv2
import numpy as np


def soft_mask(size: int, blur: int) -> np.ndarray:
    """Elliptical face mask in crop space, blurred so the seam is soft."""

    mask = np.zeros((size, size), dtype=np.float32)
    pad = max(1, int(size * 0.08))
    axes = (max(1, size // 2 - pad), max(1, int(size * 0.48) - pad))
    cv2.ellipse(mask, (size // 2, size // 2), axes, 0, 0, 360, 1.0, thickness=-1)
    kernel = int(blur)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    return mask


def match_color(swapped_bgr: np.ndarray, target_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shift the swapped crop toward the target crop's per-channel mean and deviation."""

    out = swapped_bgr.astype(np.float32)
    target = target_bgr.astype(np.float32)
    region = mask > 0.4
    if int(region.sum()) < 10:
        return swapped_bgr
    for channel in range(3):
        source_values = out[..., channel][region]
        target_values = target[..., channel][region]
        source_std = float(source_values.std()) + 1e-6
        target_std = float(target_values.std()) + 1e-6
        out[..., channel] = (out[..., channel] - float(source_values.mean())) * (target_std / source_std)
        out[..., channel] += float(target_values.mean())
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_color(mode: str, swapped_bgr: np.ndarray, target_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Recolor ``swapped_bgr`` toward ``target_bgr`` inside ``mask``."""

    if mode == "none":
        return swapped_bgr
    if mode == "mean_std":
        return match_color(swapped_bgr, target_bgr, mask)
    if mode == "lab":
        return _match_lab(swapped_bgr, target_bgr, mask)
    if mode == "histogram":
        return _match_histogram(swapped_bgr, target_bgr, mask)
    from barrett_faceswap.errors import ConfigError
    from barrett_faceswap.i18n import t

    raise ConfigError(t("config.color", mode=mode, allowed="none, mean_std, lab, histogram"))


def _match_lab(swapped_bgr: np.ndarray, target_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match per-channel mean and deviation in Lab."""

    region = mask > 0.4
    if int(region.sum()) < 10:
        return swapped_bgr
    swapped = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    for channel in range(3):
        source_values = swapped[..., channel][region]
        target_values = target[..., channel][region]
        source_std = float(source_values.std()) + 1e-6
        target_std = float(target_values.std()) + 1e-6
        swapped[..., channel] = (swapped[..., channel] - float(source_values.mean())) * (target_std / source_std)
        swapped[..., channel] += float(target_values.mean())
    clipped = np.clip(swapped, 0, 255).astype(np.uint8)
    return cv2.cvtColor(clipped, cv2.COLOR_LAB2BGR)


def _match_histogram(swapped_bgr: np.ndarray, target_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match each BGR channel's cumulative histogram inside the mask."""

    region = mask > 0.4
    if int(region.sum()) < 10:
        return swapped_bgr
    output = swapped_bgr.copy()
    for channel in range(3):
        source = swapped_bgr[..., channel][region].astype(np.uint8)
        reference = target_bgr[..., channel][region].astype(np.uint8)
        mapping = _histogram_lut(source, reference)
        output[..., channel] = mapping[swapped_bgr[..., channel]]
    return output


def _histogram_lut(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source_hist = np.bincount(source.reshape(-1), minlength=256).astype(np.float64)
    reference_hist = np.bincount(reference.reshape(-1), minlength=256).astype(np.float64)
    source_cdf = np.cumsum(source_hist)
    reference_cdf = np.cumsum(reference_hist)
    if source_cdf[-1] <= 0 or reference_cdf[-1] <= 0:
        return np.arange(256, dtype=np.uint8)
    source_cdf /= source_cdf[-1]
    reference_cdf /= reference_cdf[-1]
    mapping = np.zeros(256, dtype=np.uint8)
    ref_index = 0
    for value in range(256):
        while ref_index < 255 and reference_cdf[ref_index] < source_cdf[value]:
            ref_index += 1
        mapping[value] = ref_index
    return mapping


def paste_back(
    base_bgr: np.ndarray,
    crop_bgr: np.ndarray,
    matrix: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Inverse-warp ``crop_bgr`` and composite it with ``mask``."""

    height, width = base_bgr.shape[:2]
    inverse = cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float32))
    warped = cv2.warpAffine(crop_bgr, inverse, (width, height), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpAffine(mask, inverse, (width, height), flags=cv2.INTER_LINEAR)
    alpha = warped_mask[..., None].astype(np.float32)
    blended = warped.astype(np.float32) * alpha + base_bgr.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def paste_poisson(
    base_bgr: np.ndarray,
    crop_bgr: np.ndarray,
    matrix: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Inverse-warp and composite with ``cv2.seamlessClone``.

    The clone needs a mask that does not touch the frame border. When that
    is not true, or OpenCV refuses the clone, the alpha paste is used.
    """

    height, width = base_bgr.shape[:2]
    inverse = cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float32))
    warped = cv2.warpAffine(crop_bgr, inverse, (width, height), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpAffine(mask, inverse, (width, height), flags=cv2.INTER_LINEAR)
    binary = np.where(warped_mask > 0.5, 255, 0).astype(np.uint8)
    if binary.shape[0] > 2 and binary.shape[1] > 2:
        binary[0, :] = 0
        binary[-1, :] = 0
        binary[:, 0] = 0
        binary[:, -1] = 0
    ys, xs = np.where(binary > 0)
    if xs.size < 10:
        return paste_back(base_bgr, crop_bgr, matrix, mask)
    center = (int(round(float(xs.mean()))), int(round(float(ys.mean()))))
    try:
        return cv2.seamlessClone(warped, base_bgr, binary, center, cv2.NORMAL_CLONE)
    except cv2.error:
        return paste_back(base_bgr, crop_bgr, matrix, mask)
