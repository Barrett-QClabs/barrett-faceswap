"""Shared records passed between the CLI, the pipeline, and processors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict


@dataclass
class Face:
    """A face in the coordinate frame of the image passed to the detector.

    ``bbox`` is ``x1, y1, x2, y2``. ``landmarks`` holds five points in ArcFace
    order: left eye, right eye, nose, left mouth corner, right mouth corner.
    """

    bbox: np.ndarray
    score: float
    landmarks: np.ndarray


@dataclass
class ProcessContext:
    """Mutable per-run state visible to every processor."""

    source_bgr: np.ndarray
    dry_run: bool
    target_face_index: int = 0
    notes: list[str] = field(default_factory=list)


class SwapRequest(BaseModel):
    """One still-image invocation. The CLI and the pipeline share this schema."""

    model_config = ConfigDict(extra="forbid")

    source_path: Path
    target_path: Path
    output_path: Path
    dry_run: bool = False
    enable_compliance: bool = False
    consent_file: Path | None = None
    acknowledge_authorized_use: bool = False
    consent_subject: str | None = None
    consent_statement: str | None = None
    no_watermark: bool = False
    disable_content_filter: bool = False
    content_filter_provider: str | None = None
    content_filter_threshold: float | None = None
    no_block_on_minor: bool = False
    processors: list[str] | None = None
    execution_providers: list[str] | None = None
    target_face_index: int = 0


@dataclass
class SwapResult:
    """A finished run. ``notes`` may include the development compliance warning."""

    output_path: Path
    dry_run: bool
    compliance_enabled: bool
    notes: list[str] = field(default_factory=list)
