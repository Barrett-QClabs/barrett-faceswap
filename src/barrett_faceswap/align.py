"""Similarity alignment onto the five-point face template.

The reference points are the widely published ArcFace landmark template.
Swap models in the inswapper family were exported against that template at
128 pixels, with an 8 pixel horizontal shift relative to the 112 pixel
recognition crop. The numbers live here so a session can be fed the layout
it was built for.
"""

from __future__ import annotations

import cv2
import numpy as np

from barrett_faceswap.errors import AlignError
from barrett_faceswap.i18n import t

# Left eye, right eye, nose, left mouth corner, right mouth corner.
ARCFACE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def template_for_size(size: int) -> np.ndarray:
    """Return the five destination points for a square crop of ``size`` pixels."""

    if size <= 0:
        raise AlignError(t("align.bad_size"))
    if size % 128 == 0 and size % 112 != 0:
        ratio = float(size) / 128.0
        diff_x = 8.0 * ratio
    else:
        ratio = float(size) / 112.0
        diff_x = 0.0
    destination = ARCFACE_112 * ratio
    destination = destination.copy()
    destination[:, 0] += diff_x
    return destination.astype(np.float32)


def align_face(image_bgr: np.ndarray, landmarks: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Warp ``image_bgr`` so ``landmarks`` land on :func:`template_for_size`.

    Returns the square crop and the 2x3 similarity matrix that produced it.
    """

    source = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    destination = template_for_size(size)
    matrix, _inliers = cv2.estimateAffinePartial2D(source, destination, method=cv2.LMEDS)
    if matrix is None:
        raise AlignError(t("align.failed"))
    crop = cv2.warpAffine(image_bgr, matrix, (int(size), int(size)), borderValue=(0, 0, 0))
    return crop, matrix.astype(np.float32)
