"""Paste masks. Geometry stubs merge with optional model maps by a minimum.

``box_feather`` is an inset rectangle. ``region`` is an ellipse over the crop
when no parsing graph is loaded. ``occlusion`` is a stub of ones when no
occlusion graph is loaded, so the minimum leaves the other masks unchanged.
Real maps are passed into :func:`composite_mask` and use the same merge.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from barrett_faceswap.i18n import t

logger = logging.getLogger(__name__)


def box_feather_mask(size: int, blur: int, inset: float = 0.12) -> np.ndarray:
    """Inset box, then blur the edges."""

    mask = np.zeros((size, size), dtype=np.float32)
    margin = max(1, int(size * inset))
    mask[margin : size - margin, margin : size - margin] = 1.0
    return _blur(mask, blur)


def region_mask(size: int, blur: int) -> np.ndarray:
    """Elliptical face region inside the aligned crop."""

    mask = np.zeros((size, size), dtype=np.float32)
    pad = max(1, int(size * 0.08))
    axes = (max(1, size // 2 - pad), max(1, int(size * 0.48) - pad))
    cv2.ellipse(mask, (size // 2, size // 2), axes, 0, 0, 360, 1.0, thickness=-1)
    return _blur(mask, blur)


def occlusion_mask(size: int, path: str | None = None) -> np.ndarray:
    """Stub occlusion map. Every pixel is visible."""

    if not getattr(occlusion_mask, "_noted", False):
        logger.info(t("masks.occlusion_stub", path=path or "-"))
        setattr(occlusion_mask, "_noted", True)
    return np.ones((size, size), dtype=np.float32)


def landmark_hull_mask(size: int, landmarks: np.ndarray, blur: int) -> np.ndarray:
    """Feathered convex hull of the five crop landmarks, expanded past the eyes and mouth.

    The five ArcFace points do not cover the forehead or the cheeks. Extra
    points are derived from the eye line and the eye-to-mouth direction, then
    blurred with the same kernel as the ellipse stub.
    """

    points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 5:
        return region_mask(size, blur)
    expanded = _expand_face_points(points[:5])
    if expanded is None:
        return region_mask(size, blur)
    hull = cv2.convexHull(expanded)
    if hull is None or len(hull) < 3:
        return region_mask(size, blur)
    canvas = np.zeros((size, size), dtype=np.uint8)
    cv2.fillConvexPoly(canvas, hull.reshape(-1, 2).astype(np.int32), 255)
    return _blur(canvas.astype(np.float32) / 255.0, blur)


def parser_stub_mask(
    size: int,
    blur: int,
    path: str | None = None,
    landmarks: np.ndarray | None = None,
) -> np.ndarray:
    """Region used when the parsing model file is absent.

    With crop-space landmarks this is a feathered convex hull. Without them
    it stays the ellipse, which is what the enhancer and older callers use.
    """

    if landmarks is not None:
        if not getattr(parser_stub_mask, "_hull_noted", False):
            logger.info(t("masks.parser_hull", path=path or "-"))
            setattr(parser_stub_mask, "_hull_noted", True)
        return landmark_hull_mask(size, landmarks, blur)
    if not getattr(parser_stub_mask, "_noted", False):
        logger.info(t("masks.parser_stub", path=path or "-"))
        setattr(parser_stub_mask, "_noted", True)
    return region_mask(size, blur)


def _expand_face_points(points: np.ndarray) -> np.ndarray | None:
    """Add forehead, chin, and side points around the five-point template."""

    left_eye, right_eye, _nose, left_mouth, right_mouth = points
    eye_mid = (left_eye + right_eye) * 0.5
    mouth_mid = (left_mouth + right_mouth) * 0.5
    down = mouth_mid - eye_mid
    eye_span = right_eye - left_eye
    span = float(np.linalg.norm(eye_span))
    if span < 1e-3:
        return None
    lateral = np.array([-eye_span[1], eye_span[0]], dtype=np.float32)
    lateral_norm = float(np.linalg.norm(lateral)) or 1.0
    lateral = lateral / lateral_norm * span * 0.55
    extras = np.stack(
        [
            eye_mid - down * 0.85,
            mouth_mid + down * 0.65,
            left_eye - lateral,
            right_eye + lateral,
            left_mouth - lateral * 0.65,
            right_mouth + lateral * 0.65,
        ]
    ).astype(np.float32)
    return np.concatenate([points, extras], axis=0)


def merge_min(masks: list[np.ndarray]) -> np.ndarray:
    """Element-wise minimum. A pixel survives only where every mask allows it."""

    if not masks:
        raise ValueError("merge_min requires at least one mask")
    stacked = np.stack([np.asarray(mask, dtype=np.float32) for mask in masks], axis=0)
    return np.min(stacked, axis=0)


def composite_mask(
    size: int,
    blur: int,
    occlusion: np.ndarray | None = None,
    region: np.ndarray | None = None,
) -> np.ndarray:
    """Box, region, and occlusion, combined with :func:`merge_min`.

    ``occlusion`` and ``region`` replace the stubs when a model produced them.
    """

    return merge_min(
        [
            box_feather_mask(size, blur),
            region if region is not None else region_mask(size, blur),
            occlusion if occlusion is not None else occlusion_mask(size),
        ]
    )


def _blur(mask: np.ndarray, blur: int) -> np.ndarray:
    kernel = int(blur)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        return cv2.GaussianBlur(mask, (kernel, kernel), 0)
    return mask
