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

# Public FFHQ five-point reference at 512 pixels. Face restorers such as
# GFPGAN and CodeFormer were trained against this layout. Scaled by size / 512.
FFHQ_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19366],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)

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


def ffhq_template(size: int) -> np.ndarray:
    """Five FFHQ destination points scaled to a square of ``size`` pixels."""

    if size <= 0:
        raise AlignError(t("align.bad_size"))
    return (FFHQ_512 * (float(size) / 512.0)).astype(np.float32)


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


def swap_template(kind: str, size: int) -> np.ndarray:
    """Destination points for a swapper template name."""

    if kind == "inswapper":
        return template_for_size(size)
    if kind == "arcface":
        if size <= 0:
            raise AlignError(t("align.bad_size"))
        return (ARCFACE_112 * (float(size) / 112.0)).astype(np.float32)
    if kind == "ffhq":
        return ffhq_template(size)
    raise AlignError(t("align.template_kind", kind=kind))


def align_face(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    size: int,
    template: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp ``image_bgr`` so ``landmarks`` land on the destination template.

    The default destination is :func:`template_for_size`. Face enhancers pass
    :func:`ffhq_template` instead. Returns the square crop and the 2x3 matrix.
    """

    source = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    destination = template_for_size(size) if template is None else np.asarray(template, dtype=np.float32)
    if destination.shape != (5, 2):
        raise AlignError(t("align.template_shape", shape=tuple(int(dim) for dim in destination.shape)))
    matrix, _inliers = cv2.estimateAffinePartial2D(source, destination, method=cv2.LMEDS)
    if matrix is None:
        raise AlignError(t("align.failed"))
    crop = cv2.warpAffine(image_bgr, matrix, (int(size), int(size)), borderValue=(0, 0, 0))
    return crop, matrix.astype(np.float32)
