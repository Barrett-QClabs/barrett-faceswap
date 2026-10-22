"""Barrett FaceSwap: a still-image face-swap pipeline with an opt-in compliance gate.

The package orchestrates detection, alignment, an ONNX identity swap, and
paste-back. It does not train a foundation model and it does not ship weights.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
