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
