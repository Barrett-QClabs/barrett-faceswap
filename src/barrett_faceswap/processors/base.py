"""Processor hook used by the frame pipeline.

A processor sees one BGR frame at a time. Later stages such as a face enhancer
implement the same three methods and are appended after ``face_swapper``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from barrett_faceswap.types import ProcessContext


class Processor(Protocol):
    """FaceFusion-style stage: check, transform one frame, release resources."""

    name: str

    def pre_check(self, ctx: ProcessContext) -> None:
        """Fail early when models or settings are missing."""

    def process_frame(self, frame: np.ndarray, ctx: ProcessContext) -> np.ndarray:
        """Return a new frame. Implementations should not write the output file."""

    def post_process(self) -> None:
        """Drop per-run caches. Sessions may stay open on the model store."""
