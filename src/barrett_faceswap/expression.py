"""Paste target eyes and a pursed mouth back onto the swapped crop.

The amount follows the eye aspect ratio and the mouth aspect ratio of the
68-point set. A closed eye keeps more of the target. An open eye stays on
the swap. A pursed mouth is mixed in mildly. An open mouth is left alone.
The hulls are drawn in the aligned crop. This path replaces the five-point
eye blend, which would put most of the target eye back on every face.
"""

from __future__ import annotations

import cv2
import numpy as np

from barrett_faceswap.masks import _blur
from barrett_faceswap.recover import map_points

_EYE_GROUPS = (range(36, 48), range(17, 27))
_MOUTH_GROUPS = (range(48, 68),)


def ear68(points68: np.ndarray) -> float:
    """Average of the two six-point eye aspect ratios."""

    pts = np.asarray(points68, dtype=np.float32).reshape(-1, 2)

    def _one(start: int) -> float:
        ring = pts[start : start + 6]
        vertical = np.linalg.norm(ring[1] - ring[5]) + np.linalg.norm(ring[2] - ring[4])
        horizontal = np.linalg.norm(ring[0] - ring[3])
        return float(vertical / (2.0 * horizontal + 1e-6))

    return 0.5 * (_one(36) + _one(42))


def mar68(points68: np.ndarray) -> float:
    """Inner-lip opening divided by the mouth-corner width."""

    pts = np.asarray(points68, dtype=np.float32).reshape(-1, 2)
    return float(np.linalg.norm(pts[62] - pts[66]) / (np.linalg.norm(pts[48] - pts[54]) + 1e-6))


def closedness(ear: float, ear_open: float, ear_closed: float) -> float:
    span = float(ear_open) - float(ear_closed) + 1e-6
    return float(np.clip((float(ear_open) - float(ear)) / span, 0.0, 1.0))


def pursedness(mar: float, mar_open: float, mar_closed: float) -> float:
    span = float(mar_open) - float(mar_closed) + 1e-6
    return float(np.clip((float(mar_open) - float(mar)) / span, 0.0, 1.0))


def gate_amounts(points68: np.ndarray, settings) -> tuple[float, float, float]:
    """Eye paste, mouth paste, and eye closedness, each clipped to 0..1."""

    ear = ear68(points68)
    mar = mar68(points68)
    shut = closedness(ear, settings.ear_open, settings.ear_closed)
    pursed = pursedness(mar, settings.mar_open, settings.mar_closed)
    eye = float(np.clip(float(settings.eye_base) + float(settings.eye_boost) * shut, 0.0, 1.0))
    mouth = float(np.clip(float(settings.mouth_base) + float(settings.mouth_boost) * pursed, 0.0, 1.0))
    return eye, mouth, shut


def hull_mask(size: int, points: np.ndarray, groups, *, dilate_px: int, expand: float, blur: int) -> np.ndarray:
    """Filled convex hulls, expanded about each group's centroid, then blurred."""

    mask = np.zeros((int(size), int(size)), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    for group in groups:
        selected = pts[list(group)]
        if selected.shape[0] < 3:
            continue
        center = selected.mean(axis=0)
        grown = center + (selected - center) * float(expand)
        hull = cv2.convexHull(np.round(grown).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0)
    if int(dilate_px) > 0:
        kernel = int(dilate_px) * 2 + 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        mask = cv2.dilate(mask, element)
    return _blur(mask, int(blur))


def paste_region(swapped: np.ndarray, target: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Mix ``amount`` of the target through ``mask``. ``amount`` is clipped."""

    alpha = np.clip(mask.astype(np.float32), 0.0, 1.0) * float(np.clip(amount, 0.0, 1.0))
    mixed = target.astype(np.float32) * alpha[..., None] + swapped.astype(np.float32) * (1.0 - alpha[..., None])
    return np.clip(mixed, 0, 255).astype(np.uint8)


def preserve_expression(
    swapped: np.ndarray,
    target: np.ndarray,
    points68: np.ndarray,
    matrix: np.ndarray,
    settings,
    *,
    include_eyes: bool = True,
    mouth_scale: float = 1.0,
) -> tuple[np.ndarray, float] | None:
    """Return the pasted crop and ``1 - closedness``, or None when 68 points are absent.

    ``include_eyes`` false leaves the eye and brow hulls on the swap. That is
    the sunglass stage: landmarks under the lenses do not drive expression.
    ``mouth_scale`` multiplies the mouth amount. ``1`` is the normal paste.
    """

    pts = np.asarray(points68, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 68:
        return None
    crop_points = map_points(pts[:68], matrix)
    eye_amount, mouth_amount, shut = gate_amounts(pts[:68], settings)
    mouth_amount *= float(np.clip(mouth_scale, 0.0, 1.0))
    side = int(swapped.shape[0])
    colored = swapped
    if include_eyes:
        eyes = hull_mask(
            side,
            crop_points,
            _EYE_GROUPS,
            dilate_px=int(settings.eye_dilate),
            expand=float(settings.eye_expand),
            blur=int(settings.blur),
        )
        colored = paste_region(colored, target, eyes, eye_amount)
    mouth = hull_mask(
        side,
        crop_points,
        _MOUTH_GROUPS,
        dilate_px=int(settings.mouth_dilate),
        expand=float(settings.mouth_expand),
        blur=int(settings.blur),
    )
    colored = paste_region(colored, target, mouth, mouth_amount)
    if not include_eyes:
        return colored, 1.0
    return colored, float(1.0 - shut)
