"""Shared records passed between the CLI, the pipeline, and processors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict


@dataclass
class Face:
    """A face in the coordinate frame of the image passed to the detector.

    ``bbox`` is ``x1, y1, x2, y2``. ``landmarks`` holds at least five points.
    The first five stay in ArcFace order: left eye, right eye, nose, left
    mouth corner, right mouth corner. A 106-point set may follow them.
    ``landmarks_68`` is the optional 2DFAN4 set. It is not appended to
    ``landmarks``, because eye openness treats a long array as 106 points.
    """

    bbox: np.ndarray
    score: float
    landmarks: np.ndarray
    landmarks_68: np.ndarray | None = None


@dataclass
class ProcessContext:
    """Mutable per-run state visible to every processor."""

    source_bgr: np.ndarray
    dry_run: bool
    target_face_index: int = 0
    notes: list[str] = field(default_factory=list)
    # Video runs turn this off so a long clip does not append one note per frame.
    log_every_frame: bool = True
    # When set, the swapper uses these faces instead of running the detector.
    known_faces: list | None = None
    # Video sets this so a frame with no face is returned unchanged.
    allow_missing_face: bool = False
    # Video and live reuse occlusion and parsing masks between frames.
    reuse_masks: bool = False
    # Set by the swapper so the enhancer can keep open eyes.
    eye_openness: float | None = None
    # When set, the source embedding uses this face and skips source detection.
    source_face: Face | None = None


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
    trim_start: float | None = None
    trim_end: float | None = None
    video_encoder: str | None = None
    video_threads: int | None = None
    keep_audio: bool | None = None
    filter_every_n_frames: int | None = None
    enhancer_model: str | None = None
    enhancer_blend: float | None = None
    enhancer_weight: float | None = None
    swapper_model: str | None = None
    pixel_boost: int | None = None
    many_faces: bool = False
    selector: str | None = None
    reference_path: Path | None = None
    face_map: Path | None = None
    blend_mode: str | None = None
    color_transfer: str | None = None
    video_mode: str | None = None
    eye_mode: str | None = None
    with_enhancer: bool = False
    job_id: str | None = None
    resume: bool = False
    commercial: bool = False
    occluder_model: str | None = None
    detect_score: float | None = None
    live_session: bool = False
    live_frames: int = 0


@dataclass
class SwapResult:
    """A finished run. ``notes`` may include the development compliance warning."""

    output_path: Path
    dry_run: bool
    compliance_enabled: bool
    notes: list[str] = field(default_factory=list)
