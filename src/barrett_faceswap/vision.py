"""Image reads shared by the pipeline and the compliance gate."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from barrett_faceswap.errors import BarrettFaceswapError, UnsupportedMediaError
from barrett_faceswap.i18n import t

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def read_image(path: Path) -> np.ndarray:
    """Load a still image as a BGR ``uint8`` array."""

    if path.suffix.lower() in VIDEO_SUFFIXES:
        raise UnsupportedMediaError(t("vision.video", name=path.name))
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise UnsupportedMediaError(
            t("vision.suffix", suffix=path.suffix, allowed=", ".join(sorted(IMAGE_SUFFIXES)))
        )
    if not path.is_file():
        raise BarrettFaceswapError(t("vision.missing", path=path))
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise BarrettFaceswapError(t("vision.decode", path=path))
    return image
