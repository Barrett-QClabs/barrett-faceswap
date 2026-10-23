"""Video face recovery: a harder detect, gap fill, and temporal smoothing.

A still image does not use this module. Video calls it so one frame with no
detection does not abort the job. Beyond ``max_gap`` the frame is left unchanged.
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
    """Look ahead up to ``max_gap`` frames so a hole can use both neighbors."""

    max_gap: int
    smooth: str = "one_euro"
    counts: dict[str, int] = field(
        default_factory=lambda: {"detected": 0, "retried": 0, "filled": 0, "passthrough": 0}
    )
    _pending: list[tuple[int, np.ndarray]] = field(default_factory=list)
    _last: tuple[int, list[Face]] | None = field(default=None)
    _box: object | None = None
    _landmarks: object | None = None

    def push(self, index: int, frame: np.ndarray, faces: list[Face], tag: str) -> list[Recovered]:
        primary = _primary(faces)
        ready: list[Recovered] = []
        if primary is not None and tag != "miss":
            ready.extend(self._flush(index, faces, tag))
            ready.append(self._emit(index, frame, faces, "retry" if tag == "retry" else "detected"))
            self._last = (index, [_copy_face(face) for face in faces])
            return ready
        self._pending.append((index, frame))
        while self._pending and index - self._pending[0][0] >= self.max_gap:
            old_index, old_frame = self._pending.pop(0)
            ready.append(self._from_past(old_index, old_frame))
        return ready

    def finish(self) -> list[Recovered]:
        ready: list[Recovered] = []
        while self._pending:
            index, frame = self._pending.pop(0)
            ready.append(self._from_past(index, frame))
        return ready

    def _flush(self, right_index: int, right_faces: list[Face], tag: str) -> list[Recovered]:
        del tag
        left = self._last
        pending = self._pending
        self._pending = []
        ready: list[Recovered] = []
        right_primary = _primary(right_faces)
        left_face = _primary(left[1]) if left is not None else None
        for index, frame in pending:
            dist_left = index - left[0] if left is not None else 10**9
            dist_right = right_index - index
            if (
                dist_left <= self.max_gap
                and dist_right <= self.max_gap
                and left is not None
                and left_face is not None
                and right_primary is not None
            ):
                span = max(1, right_index - left[0])
                alpha = (index - left[0]) / span
                face = _lerp(left_face, right_primary, alpha)
                ready.append(self._emit(index, frame, [face], "filled"))
            elif dist_left <= dist_right and left_face is not None and dist_left <= self.max_gap:
                ready.append(self._emit(index, frame, [_copy_face(left_face)], "filled"))
            elif right_primary is not None and dist_right <= self.max_gap:
                ready.append(self._emit(index, frame, [_copy_face(right_primary)], "filled"))
            else:
                ready.append(self._emit(index, frame, [], "passthrough"))
        return ready

    def _from_past(self, index: int, frame: np.ndarray) -> Recovered:
        if self._last is not None and index - self._last[0] <= self.max_gap:
            primary = _primary(self._last[1])
            if primary is not None:
                return self._emit(index, frame, [_copy_face(primary)], "filled")
        return self._emit(index, frame, [], "passthrough")

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
    """Ordinary detect, then a lower threshold, rotations, and an upscale.

    Returns ``(faces, tag)`` with tag ``hit``, ``retry``, or ``miss``.
    """

    found = detector.try_detect(image)
    if found:
        return found, "hit"
    low = max(0.2, float(detector.score_threshold) * 0.5)
    if low < float(detector.score_threshold) - 1e-6:
        found = detector.try_detect(image, low)
        if found:
            return found, "retry"
    else:
        low = float(detector.score_threshold)
    for angle in RETRY_ANGLES:
        found = _attempt(detector, image, low, angle=angle, scale=1.0)
        if found:
            return found, "retry"
    found = _attempt(detector, image, low, angle=0.0, scale=RETRY_SCALE)
    if found:
        return found, "retry"
    for angle in (90.0, -90.0):
        found = _attempt(detector, image, low, angle=angle, scale=RETRY_SCALE)
        if found:
            return found, "retry"
    return [], "miss"


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


def _primary(faces: list[Face]) -> Face | None:
    if not faces:
        return None
    return max(faces, key=lambda face: _area(face))


def _order(faces: list[Face]) -> list[Face]:
    return sorted(faces, key=lambda face: _area(face), reverse=True)


def _area(face: Face) -> float:
    x1, y1, x2, y2 = [float(value) for value in face.bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _copy_face(face: Face) -> Face:
    return Face(bbox=np.array(face.bbox, dtype=np.float32, copy=True), score=float(face.score), landmarks=np.array(face.landmarks, dtype=np.float32, copy=True))


def _lerp(left: Face, right: Face, alpha: float) -> Face:
    weight = float(alpha)
    return Face(
        bbox=((1.0 - weight) * left.bbox + weight * right.bbox).astype(np.float32),
        score=(1.0 - weight) * float(left.score) + weight * float(right.score),
        landmarks=((1.0 - weight) * left.landmarks + weight * right.landmarks).astype(np.float32),
    )


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
