"""Video pipeline: probe, extract, process frames, encode, restore audio.

``CompliancePolicy.before_process`` runs once. While enforcement is on, a
sample of target frames is filtered before the swap, a sample of output frames
is filtered after it, and every output frame is watermarked. One audit line is
written for the finished file. Enforcement off skips those steps. The default
config leaves enforcement off.
"""

from __future__ import annotations

import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from barrett_faceswap.compliance.policy import CompliancePolicy
from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import ConfigError, MediaToolError, UnsupportedMediaError
from barrett_faceswap.i18n import t
from barrett_faceswap.media.ffmpeg import (
    VIDEO_OUTPUT_SUFFIXES,
    encode_video,
    extract_frames,
    probe,
    sample_indexes,
)
from barrett_faceswap.processors import create_processor
from barrett_faceswap.types import ProcessContext, SwapRequest, SwapResult
from barrett_faceswap.vision import is_video, read_image, write_image


class _LockedNotes(list):
    """A list whose ``append`` is safe to call from frame workers."""

    def __init__(self, items: list[str] | None = None) -> None:
        super().__init__(items or [])
        self._lock = threading.Lock()

    def append(self, item: object) -> None:
        with self._lock:
            super().append(item)


class VideoPipeline:
    """Run the processor chain on every frame of a target video."""

    def __init__(self, config: AppConfig, admin_path: Path | None = None) -> None:
        self.config = config
        self.policy = CompliancePolicy(config, admin_path=admin_path)

    def run(self, request: SwapRequest) -> SwapResult:
        context = None
        workspace: tempfile.TemporaryDirectory[str] | None = None
        try:
            self._check_paths(request)
            info = probe(request.target_path)
            trim_start, trim_end = _trim_bounds(request, info.duration)
            workspace = tempfile.TemporaryDirectory(prefix="barrett-faceswap-")
            root = Path(workspace.name)
            raw_dir = root / "raw"
            frames = extract_frames(
                request.target_path,
                raw_dir,
                trim_start=trim_start,
                trim_end=trim_end,
            )
            every = self.config.video.filter_every_n_frames
            indexes = sample_indexes(len(frames), every)
            samples = None
            if self.policy.enforcement_enabled(request):
                samples = [read_image(frames[index]) for index in indexes]
            context = self.policy.before_process(request, target_samples=samples)
            processed_dir = root / "processed"
            processed_dir.mkdir()
            notes = _LockedNotes(list(context.notes))
            self._process_frames(request, frames, processed_dir, notes)
            context.notes = list(notes)
            if context.enabled:
                for index in indexes:
                    self.policy.screen_frame(
                        context,
                        read_image(processed_dir / frames[index].name),
                        role=f"output[{index}]",
                    )
                if context.watermark_enabled:
                    for frame_path in frames:
                        marked_path = processed_dir / frame_path.name
                        marked = self.policy.mark_frame(context, read_image(marked_path))
                        write_image(marked_path, marked)
            audio_mode = encode_video(
                processed_dir,
                request.output_path,
                fps=info.fps,
                encoder=self.config.video.encoder,
                crf=self.config.video.crf,
                audio_source=request.target_path,
                keep_audio=self.config.video.keep_audio,
                has_audio=info.has_audio,
                trim_start=trim_start,
                trim_end=trim_end,
            )
            context.notes.append(t("video.audio_mode", mode=audio_mode))
            context.notes.append(t("video.frames_done", count=len(frames)))
            return self.policy.complete_output(context, request)
        except Exception as exc:
            if context is not None:
                self.policy.abort(context, exc)
            raise
        finally:
            if workspace is not None:
                workspace.cleanup()

    def _check_paths(self, request: SwapRequest) -> None:
        if is_video(request.source_path):
            raise UnsupportedMediaError(t("video.source_still", name=request.source_path.name))
        if not is_video(request.target_path):
            raise UnsupportedMediaError(t("video.target_required", name=request.target_path.name))
        if request.output_path.suffix.lower() not in VIDEO_OUTPUT_SUFFIXES:
            raise MediaToolError(
                t(
                    "video.output_suffix",
                    suffix=request.output_path.suffix or "-",
                    allowed=", ".join(sorted(VIDEO_OUTPUT_SUFFIXES)),
                )
            )
        if not request.source_path.is_file():
            raise MediaToolError(t("vision.missing", path=request.source_path))

    def _process_frames(
        self,
        request: SwapRequest,
        frames: list[Path],
        processed_dir: Path,
        notes: _LockedNotes,
    ) -> None:
        source = read_image(request.source_path)
        names = list(request.processors) if request.processors else list(self.config.processors)
        providers = list(request.execution_providers) if request.execution_providers else list(
            self.config.execution.providers
        )
        processors = [create_processor(name, self.config, providers) for name in names]
        shared = ProcessContext(
            source_bgr=source,
            dry_run=request.dry_run,
            target_face_index=request.target_face_index,
            notes=notes,
            log_every_frame=False,
        )
        for processor in processors:
            processor.pre_check(shared)
        workers = max(1, min(int(self.config.video.threads), len(frames)))

        def _one(frame_path: Path) -> None:
            frame = read_image(frame_path)
            if request.dry_run:
                frame = _dry_run_banner(frame, t("processor.dry_run_banner"))
            for processor in processors:
                frame = processor.process_frame(frame, shared)
            write_image(processed_dir / frame_path.name, frame)

        if workers == 1:
            for frame_path in tqdm(frames, desc=t("video.progress"), disable=None):
                _one(frame_path)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_one, frame_path) for frame_path in frames]
                for future in tqdm(as_completed(futures), total=len(futures), desc=t("video.progress"), disable=None):
                    future.result()
        for processor in processors:
            processor.post_process()


def _trim_bounds(request: SwapRequest, duration: float) -> tuple[float | None, float | None]:
    start = request.trim_start
    end = request.trim_end
    if start is None and end is None:
        return None, None
    if start is None:
        start = 0.0
    if start < 0 or (end is not None and end <= start) or (duration > 0 and start >= duration):
        raise ConfigError(t("video.trim", start=start, end="-" if end is None else end))
    if end is not None and duration > 0 and end > duration:
        end = duration
    return start, end


def _dry_run_banner(image: np.ndarray, text: str) -> np.ndarray:
    canvas = image.copy()
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas
