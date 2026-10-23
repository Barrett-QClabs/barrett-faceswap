"""Refine detector landmarks with 2DFAN4 when that ONNX file is present.

The crop is a translation warp: the face box is scaled so its longer side is
195 pixels inside a 256 canvas, then centered. The network returns 68 points.
Five alignment points are the eye-ring means, the nose tip, and the mouth
corners. Profile and low-score faces replace the detector's five points.
Frontal faces keep them. The 68-point set is stored either way so expression
can read the eyes and the mouth. A missing file leaves the face unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import cv2
import numpy as np

from barrett_faceswap.i18n import t
from barrett_faceswap.recover import map_points
from barrett_faceswap.types import Face

logger = logging.getLogger(__name__)

FAN_SIZE = 256
FAN_BOX = 195.0
FAN_SCORE_FLOOR = 0.05
_GRID = 64.0


def points68_to_5(points68: np.ndarray) -> np.ndarray:
    """Reduce an iBUG 68-point set to ArcFace's five alignment points."""

    pts = np.asarray(points68, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 68:
        raise ValueError("points68_to_5 expects 68 points")
    return np.stack(
        [
            pts[36:42].mean(axis=0),
            pts[42:48].mean(axis=0),
            pts[30],
            pts[48],
            pts[54],
        ]
    ).astype(np.float32)


def profile_ratio(points5: np.ndarray) -> float:
    """How far the nose sits from the eye midpoint, in units of eye span.

    A frontal face is near 0. A profile is large. Yaw is what this measures.
    """

    pts = np.asarray(points5, dtype=np.float32).reshape(-1, 2)
    span = abs(float(pts[1, 0] - pts[0, 0])) + 1e-6
    mid = 0.5 * (float(pts[0, 0]) + float(pts[1, 0]))
    return abs(float(pts[2, 0]) - mid) / span


def should_replace_alignment(face: Face, settings) -> bool:
    """True when the five detector points should be replaced by the 68-point set."""

    mode = str(settings.refine)
    if mode == "off":
        return False
    if mode == "always":
        return True
    if float(face.score) < float(settings.min_score):
        return True
    points = np.asarray(face.landmarks, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 5:
        return True
    return profile_ratio(points[:5]) >= float(settings.profile_ratio)


def fan_matrix(bbox: np.ndarray) -> np.ndarray:
    """2x3 matrix that places the face box in the 256 crop."""

    x1, y1, x2, y2 = [float(value) for value in np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]]
    scale = FAN_BOX / max(x2 - x1, y2 - y1, 1.0)
    tx = (FAN_SIZE - (x1 + x2) * scale) * 0.5
    ty = (FAN_SIZE - (y1 + y2) * scale) * 0.5
    return np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)


def lift_dark(rgb: np.ndarray) -> np.ndarray:
    """Lift a very dark crop in Lab so the landmark net can see a profile."""

    image = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    if float(lab[:, :, 0].mean()) >= 30.0:
        return image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lifted = lab.copy()
    lifted[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lifted, cv2.COLOR_LAB2RGB)


def fan_tensor(frame_bgr: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Warp to 256, convert BGR to RGB, and return a 0..1 NCHW batch."""

    warped = cv2.warpAffine(frame_bgr, matrix, (FAN_SIZE, FAN_SIZE), flags=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    rgb = lift_dark(rgb)
    array = rgb.astype(np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...]


def bind_fan_image(session, nchw: np.ndarray) -> tuple[str, np.ndarray]:
    """Name the first input. A channels-last static shape is transposed."""

    item = session.get_inputs()[0]
    image = nchw
    shape = list(getattr(item, "shape", []) or [])
    if shape and _static_dim(shape[-1]) == 3:
        image = np.transpose(nchw, (0, 2, 3, 1))
    return str(item.name), np.ascontiguousarray(image)


def heatmap_score(heatmap: np.ndarray) -> float:
    """Mean spatial peak, mapped from [0, 0.9] onto [0, 1]."""

    array = np.asarray(heatmap, dtype=np.float32)
    if array.size == 0:
        return 0.0
    if array.ndim >= 3:
        peaks = np.max(array, axis=(-1, -2))
        raw = float(np.mean(peaks))
    elif array.ndim == 2:
        raw = float(np.mean(np.max(array, axis=-1)))
    else:
        raw = float(np.mean(array))
    return float(np.interp(raw, [0.0, 0.9], [0.0, 1.0]))


def decode_fan(outputs: list, matrix: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Map network landmarks back to the frame. A 64-grid is scaled to 256."""

    if not outputs:
        return None, 0.0
    array = np.asarray(outputs[0], dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] < 2:
        return None, 0.0
    points = array.reshape(-1, array.shape[-1])[:, :2]
    if points.shape[0] < 68:
        return None, 0.0
    points = points[:68].copy()
    if float(np.max(np.abs(points))) <= 96.0:
        points *= FAN_SIZE / _GRID
    inverse = cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float32))
    frame_points = map_points(points, inverse)
    score = heatmap_score(outputs[1]) if len(outputs) > 1 else 1.0
    return frame_points.astype(np.float32), score


def attach_landmarks(face: Face, points68: np.ndarray, settings) -> Face:
    """Store 68 points and, when the policy says so, replace the first five."""

    landmarks = np.asarray(face.landmarks, dtype=np.float32).reshape(-1, 2).copy()
    if should_replace_alignment(face, settings):
        five = points68_to_5(points68)
        if landmarks.shape[0] >= 5:
            landmarks[:5] = five
        else:
            landmarks = five
    return replace(face, landmarks=landmarks, landmarks_68=np.asarray(points68, dtype=np.float32).reshape(68, 2))


def apply_fan(frame_bgr: np.ndarray, face: Face, session, settings) -> Face:
    """Run one 2DFAN4 session and return the face, or the original on a weak score."""

    matrix = fan_matrix(face.bbox)
    name, image = bind_fan_image(session, fan_tensor(frame_bgr, matrix))
    outputs = session.run(None, {name: image})
    points, score = decode_fan(list(outputs), matrix)
    if points is None or score < FAN_SCORE_FLOOR:
        return face
    return attach_landmarks(face, points, settings)


class LandmarkRefiner:
    """Lazy 2DFAN4 session. A missing or broken file does not fail the swap."""

    def __init__(self, store, config) -> None:
        self.store = store
        self.settings = config.landmarks
        self._session = None
        self._failed = False
        self._noted_missing = False

    def refine(self, frame_bgr: np.ndarray, face: Face) -> Face:
        if str(self.settings.refine) == "off" or self._failed:
            return face
        paths = getattr(self.store, "paths", None)
        if not callable(paths):
            return face
        path = paths().get("landmark")
        if path is None or not path.is_file():
            if not self._noted_missing:
                logger.info(t("landmarks.missing", path=path))
                self._noted_missing = True
            return face
        try:
            if self._session is None:
                self._session = self.store.session("landmark")
            return apply_fan(frame_bgr, face, self._session, self.settings)
        except Exception as exc:
            logger.info(t("landmarks.failed", error=exc))
            self._failed = True
            return face


def _static_dim(dim: object) -> int | None:
    if isinstance(dim, (int, np.integer)) and int(dim) > 0:
        return int(dim)
    text = str(dim)
    if text.isdigit() and int(text) > 0:
        return int(text)
    return None
