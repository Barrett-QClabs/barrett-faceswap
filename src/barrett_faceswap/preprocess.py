"""Crop normalization for each swapper contract.

The numbers are the mean and standard deviation each published graph was
exported with. This module does not load a weight and does not download one.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_crop(crop_bgr: np.ndarray, spec) -> np.ndarray:
    """BGR crop to NCHW float32 using ``spec.mean`` and ``spec.std``."""

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray(spec.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(spec.std, dtype=np.float32).reshape(1, 1, 3)
    rgb = (rgb - mean) / np.maximum(std, 1e-8)
    return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)


def denormalize_nchw(array: np.ndarray, spec) -> np.ndarray:
    """Undo ``normalize_crop`` when the graph was trained in that space."""

    data = np.asarray(array, dtype=np.float32)
    scale = np.asarray(spec.std, dtype=np.float32).reshape(1, 3, 1, 1)
    shift = np.asarray(spec.mean, dtype=np.float32).reshape(1, 3, 1, 1)
    return data * scale + shift
