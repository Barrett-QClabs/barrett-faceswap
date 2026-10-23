"""Expected failures raised by the pipeline and the compliance gate."""

from __future__ import annotations


class BarrettFaceswapError(Exception):
    """Base class for errors the CLI can report without a traceback."""


class ConfigError(BarrettFaceswapError):
    """The YAML config is missing or invalid."""


class ModelNotFoundError(BarrettFaceswapError):
    """One or more ONNX weights are missing under the configured paths."""


class ModelHashError(BarrettFaceswapError):
    """A local weight file does not match the pinned sha256."""


class DetectorError(BarrettFaceswapError):
    """A detector graph returned tensors this build cannot decode."""


class AlignError(BarrettFaceswapError):
    """Five-point landmarks could not be mapped onto the model template."""


class SwapError(BarrettFaceswapError):
    """The swap graph rejected an input or is missing a required tensor."""


class NoFaceError(BarrettFaceswapError):
    """No face passed the score threshold, or the requested index is missing."""


class UnsupportedMediaError(BarrettFaceswapError):
    """The file type is not valid for the reader or pipeline that received it."""


class MediaToolError(BarrettFaceswapError):
    """ffmpeg or ffprobe is missing, or a media command failed."""


class VoiceError(BarrettFaceswapError):
    """Voice conversion was requested and could not run."""


class ProcessorError(BarrettFaceswapError):
    """The named processor is unknown or not implemented in this phase."""


class ComplianceError(BarrettFaceswapError):
    """Base class for consent, filter, watermark, and audit failures."""


class ConsentError(ComplianceError):
    """The run has no usable authorization record."""


class CompliancePermissionError(ComplianceError):
    """A safety control was weakened without the permission switch."""


class ContentBlockedError(ComplianceError):
    """The content filter refused a source, target, or output frame."""


class ContentFilterError(ComplianceError):
    """The selected filter provider could not be constructed."""


class WatermarkError(ComplianceError):
    """Visible or metadata marking could not be written."""
