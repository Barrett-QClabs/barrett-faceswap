"""Evaluation numbers for a swapper you train yourself.

SSIM is the global (single-window) form. Identity similarity is cosine
distance on two embeddings the caller already computed. Neither function
loads a network.
"""

from __future__ import annotations

import numpy as np

from barrett_faceswap.facesel import cosine


def ssim(left: np.ndarray, right: np.ndarray) -> float:
    """Global SSIM of two images. Shape and channel count must match."""

    image_a = _gray(left)
    image_b = _gray(right)
    if image_a.shape != image_b.shape:
        raise ValueError("ssim inputs must share a shape")
    flat_a = image_a.astype(np.float64)
    flat_b = image_b.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = float(flat_a.mean())
    mu_b = float(flat_b.mean())
    var_a = float(flat_a.var())
    var_b = float(flat_b.var())
    cov = float(((flat_a - mu_a) * (flat_b - mu_b)).mean())
    numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    return float(numerator / max(denominator, 1e-12))


def identity_cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity of two ArcFace-sized vectors."""

    return cosine(left, right)


def _gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 3:
        return array.mean(axis=2)
    raise ValueError("ssim expects a gray image or a 3-channel image")
