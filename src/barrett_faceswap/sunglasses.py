"""Default path when the target face is wearing sunglasses.

Stage 1 pastes a fuller head from the identity swap and leaves expression
off, so eye landmarks under the lenses do not drive the result. Stage 2 may
mix a mild mouth from the non-eye landmarks. A hole in the mid-face is filled
by that identity mask. It is not inpainted.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from barrett_faceswap.masks import box_feather_mask, landmark_hull_mask
from barrett_faceswap.parsing import PARSING_REGIONS

_GLASSES = PARSING_REGIONS["eye_glasses"]


@dataclass(frozen=True)
class SunglassPlan:
    """What the swapper should do for this crop."""

    active: bool
    box_inset: float
    expression: bool
    mouth_scale: float


def sunglass_plan(settings, crop_bgr: np.ndarray, landmarks: np.ndarray, labels: np.ndarray | None = None) -> SunglassPlan:
    """Active only when sunglasses are enabled and this crop is wearing them."""

    inactive = SunglassPlan(False, float(settings.box_inset), False, 0.0)
    if not bool(settings.enabled):
        return inactive
    if not wearing_sunglasses(crop_bgr, landmarks, labels):
        return inactive
    return SunglassPlan(
        True,
        float(settings.box_inset),
        bool(settings.expression),
        float(settings.mouth_scale),
    )


def wearing_sunglasses(crop_bgr: np.ndarray, landmarks: np.ndarray, labels: np.ndarray | None = None) -> bool:
    """True for a glasses bar from the parser, or a dark bar across both eyes.

    A parser map is enough on its own when the glasses class spans the eye
    line. The dark-bar check still runs, because a parser can label lenses as
    eyes. A uniformly dark crop is not sunglasses.
    """

    if labels is not None and _parser_glasses(labels):
        return True
    return _dark_bar(crop_bgr, landmarks)


def apply_sunglass_mask(
    mask: np.ndarray,
    size: int,
    blur: int,
    landmarks: np.ndarray,
    inset: float,
) -> np.ndarray:
    """Widen the paste to the head and fill mid-face holes with identity coverage.

    The returned mask is at least the fuller head. Pixels outside that head
    stay as they were only when the head itself is empty there. No inpaint.
    """

    box = box_feather_mask(int(size), int(blur), inset=float(inset))
    cover = np.minimum(box, landmark_hull_mask(int(size), landmarks, int(blur)))
    base = np.asarray(mask, dtype=np.float32)
    if base.shape[:2] != (int(size), int(size)):
        base = cv2.resize(base, (int(size), int(size)), interpolation=cv2.INTER_LINEAR)
    return np.clip(np.maximum(base, cover), 0.0, 1.0)


def _parser_glasses(labels: np.ndarray) -> bool:
    data = np.asarray(labels)
    if data.ndim != 2:
        return False
    height, width = data.shape
    if height < 8 or width < 8:
        return False
    glasses = data == _GLASSES
    if float(glasses.mean()) < 0.012:
        return False
    band = glasses[int(height * 0.15) : int(height * 0.62)]
    if band.size == 0:
        return False
    return float(band.any(axis=0).mean()) >= 0.28


def _dark_bar(crop_bgr: np.ndarray, landmarks: np.ndarray) -> bool:
    points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2:
        return False
    gray = cv2.cvtColor(np.asarray(crop_bgr), cv2.COLOR_BGR2GRAY).astype(np.float32)
    left, right = points[0], points[1]
    span = float(abs(right[0] - left[0]))
    if span < 8.0:
        return False
    eye_y = float(left[1] + right[1]) * 0.5
    half = 0.28 * span
    x0 = float(min(left[0], right[0]) - 0.15 * span)
    x1 = float(max(left[0], right[0]) + 0.15 * span)
    eye = _window(gray, x0, x1, eye_y - half, eye_y + half * 0.65)
    forehead = _window(gray, x0, x1, eye_y - half * 2.3, eye_y - half)
    cheek = _window(gray, x0, x1, eye_y + 0.55 * span, eye_y + 1.05 * span)
    if eye is None or forehead is None or cheek is None:
        return False
    eye_med = float(np.median(eye))
    fore_med = float(np.median(forehead))
    cheek_med = float(np.median(cheek))
    if fore_med < 25.0 and cheek_med < 25.0:
        return False
    reference = max(fore_med, cheek_med, 1.0)
    if reference - eye_med < 40.0 or eye_med > 0.55 * reference:
        return False
    dark = eye.mean(axis=0) < 0.50 * reference
    return float(np.mean(dark)) >= 0.62


def _window(gray: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray | None:
    height, width = gray.shape
    left = int(np.clip(np.floor(x0), 0, width - 1))
    right = int(np.clip(np.ceil(x1), 0, width))
    top = int(np.clip(np.floor(y0), 0, height - 1))
    bottom = int(np.clip(np.ceil(y1), 0, height))
    if right - left < 8 or bottom - top < 3:
        return None
    return gray[top:bottom, left:right]
