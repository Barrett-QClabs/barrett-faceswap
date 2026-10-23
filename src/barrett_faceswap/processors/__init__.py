"""Import concrete processors so their factories register on package import."""

from __future__ import annotations

from barrett_faceswap.processors.face_enhancer import FaceEnhancer
from barrett_faceswap.processors.face_swapper import FaceSwapper
from barrett_faceswap.processors.registry import available_processors, create_processor


def resolve_processor_names(config, request) -> list[str]:
    """Processor list for one request. ``--with-enhancer`` appends the restorer."""

    names = list(request.processors) if getattr(request, "processors", None) else list(config.processors)
    if getattr(request, "with_enhancer", False) and "face_enhancer" not in names:
        names = [*names, "face_enhancer"]
    return names


__all__ = [
    "FaceEnhancer",
    "FaceSwapper",
    "available_processors",
    "create_processor",
    "resolve_processor_names",
]
