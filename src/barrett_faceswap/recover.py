"""Video face recovery: a same-threshold retry, then temporal smoothing.

A still image does not use this module. Video calls it so one frame with no
detection does not abort the job. A miss leaves that frame unchanged. The last
box is not pasted onto later empty frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from barrett_faceswap.types import Face

# Small tilts first, then profile-ish right angles. Each runs only after the
# ordinary detect misses, so a normal frame stays on the fast path.
RETRY_ANGLES = (15.0, -15.0, 30.0, -30.0, 90.0, -90.0)
RETRY_SCALE = 1.5


@dataclass
class Recovered:
    """One frame after gap fill. ``faces`` is empty when the frame is unchanged."""

    index: int
    frame: np.ndarray
    faces: list[Face]
    tag: str


@dataclass
class FaceStream:
    """Emit a detected face, and leave a miss unchanged.

    ``max_gap`` is accepted so callers can keep passing ``video.max_face_gap``.
    This stream does not copy the last box across that gap. The identity
    tracker uses the same count to forget a lost face.
    """

    max_gap: int
    smooth: str = "one_euro"
    counts: dict[str, int] = field(
        default_factory=lambda: {"detected": 0, "retried": 0, "filled": 0, "passthrough": 0}
    )
    _box: object | None = None
    _landmarks: object | None = None

    def push(self, index: int, frame: np.ndarray, faces: list[Face], tag: str) -> list[Recovered]:
        if tag == "miss" or not faces:
            return [self._emit(index, frame, [], "passthrough")]
        kind = "retry" if tag == "retry" else "detected"
        return [self._emit(index, frame, faces, kind)]

    def finish(self) -> list[Recovered]:
        return []

    def _emit(self, index: int, frame: np.ndarray, faces: list[Face], tag: str) -> Recovered:
        if tag == "passthrough" or not faces:
            self._reset_smooth()
            self.counts["passthrough"] += 1
            return Recovered(index=index, frame=frame, faces=[], tag="passthrough")
        smoothed = [_smooth_face(self, face) if i == 0 else face for i, face in enumerate(_order(faces))]
        if tag == "detected":
            self.counts["detected"] += 1
        elif tag == "retry":
            self.counts["detected"] += 1
            self.counts["retried"] += 1
        else:
            self.counts["filled"] += 1
        return Recovered(index=index, frame=frame, faces=smoothed, tag=tag)

    def _reset_smooth(self) -> None:
        if self._box is not None:
            self._box.reset()
        if self._landmarks is not None:
            self._landmarks.reset()


def search_faces(detector, image: np.ndarray) -> tuple[list[Face], str]:
    """Detect at ``score_threshold``, then rotate and upscale at that same score.

    A box below the configured score is not a face. The threshold is not
    lowered on a miss, so noise on an empty frame is not swapped.
    Returns ``(faces, tag)`` with tag ``hit``, ``retry``, or ``miss``.
    """

    threshold = float(detector.score_threshold)
    found = _accepted(detector.try_detect(image, threshold), threshold)
    if found:
        return found, "hit"
    for angle in RETRY_ANGLES:
        found = _accepted(_attempt(detector, image, threshold, angle=angle, scale=1.0), threshold)
        if found:
            return found, "retry"
    found = _accepted(_attempt(detector, image, threshold, angle=0.0, scale=RETRY_SCALE), threshold)
    if found:
        return found, "retry"
    for angle in (90.0, -90.0):
        found = _accepted(_attempt(detector, image, threshold, angle=angle, scale=RETRY_SCALE), threshold)
        if found:
            return found, "retry"
    return [], "miss"


def _accepted(faces: list[Face], threshold: float) -> list[Face]:
    """Keep faces that clear ``threshold``. A missing score does not pass."""

    kept: list[Face] = []
    for face in faces:
        if float(face.score) >= float(threshold):
            kept.append(face)
    return kept


def rotate_bgr(image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate around the center and expand the canvas so the face is not clipped."""

    height, width = image.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    cosine = abs(float(matrix[0, 0]))
    sine = abs(float(matrix[0, 1]))
    bound_w = int(math.ceil(height * sine + width * cosine))
    bound_h = int(math.ceil(height * cosine + width * sine))
    matrix = matrix.copy()
    matrix[0, 2] += (bound_w / 2.0) - center[0]
    matrix[1, 2] += (bound_h / 2.0) - center[1]
    rotated = cv2.warpAffine(image, matrix, (bound_w, bound_h), flags=cv2.INTER_LINEAR)
    return rotated, matrix.astype(np.float32)


def map_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix."""

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    return (np.asarray(matrix, dtype=np.float32) @ np.concatenate([pts, ones], axis=1).T).T


def _attempt(detector, image: np.ndarray, threshold: float, *, angle: float, scale: float) -> list[Face]:
    working = image
    if scale != 1.0:
        working = cv2.resize(
            image,
            None,
            fx=float(scale),
            fy=float(scale),
            interpolation=cv2.INTER_LINEAR,
        )
    scaled_h, scaled_w = working.shape[:2]
    matrix = None
    if angle:
        working, matrix = rotate_bgr(working, angle)
    found = detector.try_detect(working, threshold)
    if not found:
        return []
    if matrix is not None:
        found = _unmap_faces(found, matrix, scaled_w, scaled_h)
    if scale != 1.0:
        found = _scale_faces(found, 1.0 / float(scale), image.shape[1], image.shape[0])
    return found


def _unmap_faces(faces: list[Face], matrix: np.ndarray, width: int, height: int) -> list[Face]:
    inverse = cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float32))
    output: list[Face] = []
    for face in faces:
        output.append(
            Face(
                bbox=_clip_box(map_points(face.bbox.reshape(2, 2), inverse).reshape(-1), width, height),
                score=face.score,
                landmarks=_clip_points(map_points(face.landmarks, inverse), width, height),
            )
        )
    return output


def _scale_faces(faces: list[Face], factor: float, width: int, height: int) -> list[Face]:
    output: list[Face] = []
    for face in faces:
        output.append(
            Face(
                bbox=_clip_box(face.bbox * factor, width, height),
                score=face.score,
                landmarks=_clip_points(face.landmarks * factor, width, height),
            )
        )
    return output


def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    values = np.asarray(box, dtype=np.float32).reshape(4).copy()
    values[0::2] = np.clip(values[0::2], 0, max(0, width - 1))
    values[1::2] = np.clip(values[1::2], 0, max(0, height - 1))
    return values


def _clip_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    values[:, 0] = np.clip(values[:, 0], 0, max(0, width - 1))
    values[:, 1] = np.clip(values[:, 1], 0, max(0, height - 1))
    return values


def _order(faces: list[Face]) -> list[Face]:
    return sorted(faces, key=lambda face: _area(face), reverse=True)


def _area(face: Face) -> float:
    x1, y1, x2, y2 = [float(value) for value in face.bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _smooth_face(stream: FaceStream, face: Face) -> Face:
    mode = stream.smooth
    if mode == "none":
        return face
    if stream._box is None:
        stream._box = _OneEuro() if mode == "one_euro" else _Ema(0.45)
        stream._landmarks = _OneEuro() if mode == "one_euro" else _Ema(0.45)
    box = stream._box(np.asarray(face.bbox, dtype=np.float32).reshape(-1))
    landmarks = stream._landmarks(np.asarray(face.landmarks, dtype=np.float32).reshape(-1))
    return Face(
        bbox=np.asarray(box, dtype=np.float32).reshape(4),
        score=float(face.score),
        landmarks=np.asarray(landmarks, dtype=np.float32).reshape(-1, 2),
    )


class LandmarkSmoother:
    """One-Euro or EMA on each face's landmarks before a later alignment."""

    def __init__(self, mode: str = "one_euro") -> None:
        self.mode = mode
        self._points: list = []

    def apply(self, faces: list) -> list:
        if self.mode == "none" or not faces:
            return list(faces)
        while len(self._points) < len(faces):
            self._points.append(_OneEuro() if self.mode == "one_euro" else _Ema(0.45))
        smoothed = []
        for index, face in enumerate(faces):
            points = self._points[index](np.asarray(face.landmarks, dtype=np.float32).reshape(-1))
            smoothed.append(
                Face(
                    bbox=np.array(face.bbox, dtype=np.float32, copy=True),
                    score=float(face.score),
                    landmarks=np.asarray(points, dtype=np.float32).reshape(-1, 2),
                )
            )
        return smoothed


class _OneEuro:
    """1€ filter. Per-element cutoff tracks how fast the landmark is moving."""

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0, beta: float = 0.3, d_cutoff: float = 1.0) -> None:
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = None

    def __call__(self, value: np.ndarray) -> np.ndarray:
        current = np.asarray(value, dtype=np.float64).reshape(-1)
        if self.x_prev is None or self.dx_prev is None:
            self.x_prev = current
            self.dx_prev = np.zeros_like(current)
            return current.astype(np.float32)
        derivative = (current - self.x_prev) * self.freq
        alpha_d = _alpha(self.freq, self.d_cutoff)
        dx_hat = alpha_d * derivative + (1.0 - alpha_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = _alpha(self.freq, cutoff)
        hat = alpha * current + (1.0 - alpha) * self.x_prev
        self.x_prev = hat
        self.dx_prev = dx_hat
        return hat.astype(np.float32)


class _Ema:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.previous: np.ndarray | None = None

    def reset(self) -> None:
        self.previous = None

    def __call__(self, value: np.ndarray) -> np.ndarray:
        current = np.asarray(value, dtype=np.float32).reshape(-1)
        if self.previous is None:
            self.previous = current
            return current
        mixed = self.alpha * current + (1.0 - self.alpha) * self.previous
        self.previous = mixed
        return mixed


def _alpha(freq: float, cutoff) -> np.ndarray:
    safe = np.maximum(np.asarray(cutoff, dtype=np.float64), 1e-3)
    tau = 1.0 / (2.0 * math.pi * safe)
    te = 1.0 / float(freq)
    return 1.0 / (1.0 + tau / te)
