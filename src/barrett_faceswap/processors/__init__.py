"""Import concrete processors so their factories register on package import."""

from __future__ import annotations

from barrett_faceswap.processors.face_enhancer import FaceEnhancer
from barrett_faceswap.processors.face_swapper import FaceSwapper
from barrett_faceswap.processors.registry import available_processors, create_processor

__all__ = [
    "FaceEnhancer",
    "FaceSwapper",
    "available_processors",
    "create_processor",
]
