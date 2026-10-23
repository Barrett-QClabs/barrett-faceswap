"""Barrett FaceSwap: still-image and video face swap with an opt-in compliance gate.

The package orchestrates detection, alignment, an ONNX identity swap, optional
face restoration, and paste-back. It does not train a foundation model and it
does not ship weights.
"""

from __future__ import annotations

__version__ = "0.3.0"

__all__ = ["__version__"]
