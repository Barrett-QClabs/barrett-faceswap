"""Feathered eye masks and a small unsharp restore.

Inswapper's 128 pixel crop often blurs the eyes and loses the target gaze.
The mask is built in the aligned crop. Five-point landmarks become two
ellipses. A denser set (106 points) uses the convex hull of the points that
cluster around each eye, so the same function serves both layouts.
"""

from __future__ import annotations

import cv2
import numpy as np


def eye_mask(size: int, landmarks: np.ndarray, blur: int) -> np.ndarray:
    """Float mask in ``[0, 1]``. Ones sit on the eyes."""

    points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] >= 106:
        mask = _hull_eyes(size, points)
    elif points.shape[0] >= 2:
        mask = _ellipse_eyes(size, points[:5] if points.shape[0] >= 5 else points)
    else:
        mask = np.zeros((size, size), dtype=np.float32)
    return _feather(mask, blur)


def eye_openness(landmarks: np.ndarray) -> float:
    """How open the eyes are, from 0 (closed) to 1 (open).

    Five-point landmarks have no upper and lower lid, so they are treated as
    open. That keeps the target eye instead of mixing in the narrower
    inswapper lids. A 106-point set uses the vertical spread of each eye
    cluster divided by its width. No fixed index table is required.
    """

    points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 106:
        return 1.0
    median_y = float(np.median(points[:, 1]))
    upper = points[points[:, 1] <= median_y]
    if upper.shape[0] < 6:
        upper = points
    median_x = float(np.median(upper[:, 0]))
    ratios = [
        _cluster_ratio(group)
        for group in (upper[upper[:, 0] < median_x], upper[upper[:, 0] >= median_x])
    ]
    ratios = [ratio for ratio in ratios if ratio is not None]
    if not ratios:
        return 1.0
    ratio = float(np.mean(ratios))
    # A closed cluster is a thin band. An open eye is closer to a square.
    return float(np.clip((ratio - 0.18) / 0.32, 0.0, 1.0))


def restore_eyes(
    mode: str,
    swapped_bgr: np.ndarray,
    target_bgr: np.ndarray,
    landmarks: np.ndarray,
    blur: int,
    sharpen: float,
    mask: np.ndarray | None = None,
    openness: float | None = None,
) -> np.ndarray:
    """Put target eye detail back, then sharpen inside the eye mask.

    ``swap`` keeps the swapped eyes. ``blend`` mixes in the original eyes.
    ``target`` keeps the original eyes. Open eyes raise the blend toward the
    target so the source eyelids do not leave a half-closed outline.
    ``mask``, when set, replaces the ellipse or hull. Sharpening stays inside
    the feathered region.
    """

    if mode not in {"swap", "blend", "target"}:
        mode = "blend"
    amount = float(sharpen)
    if mode == "swap" and amount <= 0:
        return swapped_bgr
    if mask is None:
        mask = eye_mask(int(swapped_bgr.shape[0]), landmarks, blur)
    else:
        mask = _fit_eye_mask(mask, swapped_bgr.shape[0], swapped_bgr.shape[1])
    image = swapped_bgr
    if mode != "swap":
        measured = eye_openness(landmarks) if openness is None else float(np.clip(openness, 0.0, 1.0))
        base = 1.0 if mode == "target" else 0.65
        # 0.65 at closed, about 0.91 when the target eyes are open.
        keep = base if mode == "target" else base + (1.0 - base) * 0.75 * measured
        alpha = (mask * keep)[..., None]
        mixed = target_bgr.astype(np.float32) * alpha + swapped_bgr.astype(np.float32) * (1.0 - alpha)
        image = np.clip(mixed, 0, 255).astype(np.uint8)
    if amount <= 0:
        return image
    return _unsharp_region(image, mask, amount)


def _ellipse_eyes(size: int, points: np.ndarray) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    if points.shape[0] < 2:
        return mask
    left = points[0]
    right = points[1]
    span = float(np.linalg.norm(right - left))
    if span < 1.0:
        span = float(size) * 0.2
    axes = (max(2, int(round(span * 0.42))), max(2, int(round(span * 0.28))))
    for center in (left, right):
        cv2.ellipse(
            mask,
            (int(round(float(center[0]))), int(round(float(center[1])))),
            axes,
            0,
            0,
            360,
            1.0,
            thickness=-1,
        )
    return mask


def _hull_eyes(size: int, points: np.ndarray) -> np.ndarray:
    """Hull the upper-face clusters. Works for a 106-point set without a fixed index table."""

    mask = np.zeros((size, size), dtype=np.uint8)
    median_y = float(np.median(points[:, 1]))
    upper = points[points[:, 1] <= median_y]
    if upper.shape[0] < 6:
        upper = points
    median_x = float(np.median(upper[:, 0]))
    groups = (upper[upper[:, 0] < median_x], upper[upper[:, 0] >= median_x])
    for group in groups:
        core = _core(group)
        if core.shape[0] < 3:
            continue
        hull = cv2.convexHull(core.astype(np.float32))
        if hull is None or len(hull) < 3:
            continue
        cv2.fillConvexPoly(mask, hull.reshape(-1, 2).astype(np.int32), 255)
    return mask.astype(np.float32) / 255.0


def _core(group: np.ndarray) -> np.ndarray:
    if group.shape[0] == 0:
        return group
    center = group.mean(axis=0)
    distance = np.linalg.norm(group - center, axis=1)
    if distance.size <= 4:
        return group
    radius = float(np.percentile(distance, 45)) * 1.6
    kept = group[distance <= max(radius, 1.0)]
    return kept if kept.shape[0] >= 3 else group


def _cluster_ratio(group: np.ndarray) -> float | None:
    if group.shape[0] < 6:
        return None
    width = float(group[:, 0].max() - group[:, 0].min())
    if width < 1.0:
        return None
    median_y = float(np.median(group[:, 1]))
    upper = group[group[:, 1] <= median_y]
    lower = group[group[:, 1] > median_y]
    if upper.shape[0] == 0 or lower.shape[0] == 0:
        return 0.0
    gap = float(lower[:, 1].mean() - upper[:, 1].mean())
    return max(gap, 0.0) / width


def _fit_eye_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    fitted = np.asarray(mask, dtype=np.float32)
    if fitted.ndim == 3:
        fitted = fitted[..., 0]
    if fitted.shape[:2] != (height, width):
        fitted = cv2.resize(fitted, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(fitted, 0.0, 1.0)


def _feather(mask: np.ndarray, blur: int) -> np.ndarray:
    kernel = int(blur)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        return cv2.GaussianBlur(mask.astype(np.float32), (kernel, kernel), 0)
    return mask.astype(np.float32)


def _unsharp_region(image: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.15)
    sharp = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    alpha = mask[..., None].astype(np.float32)
    mixed = sharp.astype(np.float32) * alpha + image.astype(np.float32) * (1.0 - alpha)
    return np.clip(mixed, 0, 255).astype(np.uint8)
