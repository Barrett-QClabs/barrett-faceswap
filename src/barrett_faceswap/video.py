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
from barrett_faceswap.jobs import JobState, JobStore
from barrett_faceswap.media.ffmpeg import (
    VIDEO_OUTPUT_SUFFIXES,
    RawVideoPipe,
    encode_video,
    iter_raw_frames,
    even_size,
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

    def run(
        self,
        request: SwapRequest,
        job: JobState | None = None,
        store: JobStore | None = None,
    ) -> SwapResult:
        if self.config.video.mode == "pipe":
            return self._run_pipe(request, job, store)
        return self._run_disk(request, job, store)

    def _run_disk(
        self,
        request: SwapRequest,
        job: JobState | None,
        store: JobStore | None,
    ) -> SwapResult:
        context = None
        workspace: tempfile.TemporaryDirectory[str] | None = None
        try:
            self._check_paths(request)
            info = probe(request.target_path)
            trim_start, trim_end = _trim_bounds(request, info.duration)
            if job is not None and store is not None:
                root = store.directory(job.id)
                root.mkdir(parents=True, exist_ok=True)
            else:
                workspace = tempfile.TemporaryDirectory(prefix="barrett-faceswap-")
                root = Path(workspace.name)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            start = int(job.next_frame) if job is not None and request.resume else 0
            existing = sorted(raw_dir.glob("frame_*.png")) if raw_dir.is_dir() else []
            if start > 0 and existing:
                frames = existing
            else:
                frames = extract_frames(
                    request.target_path,
                    raw_dir,
                    trim_start=trim_start,
                    trim_end=trim_end,
                )
            if job is not None and store is not None:
                job.total_frames = len(frames)
                store.save(job)
                context_note = t("jobs.resume_from", frame=start, job=job.id)
            else:
                context_note = None
            every = self.config.video.filter_every_n_frames
            indexes = sample_indexes(len(frames), every)
            samples = None
            if self.policy.enforcement_enabled(request):
                samples = [read_image(frames[index]) for index in indexes]
            context = self.policy.before_process(request, target_samples=samples)
            if context_note:
                context.notes.append(context_note)
            notes = _LockedNotes(list(context.notes))
            self._process_frames(request, frames, processed_dir, notes, start_index=start, job=job, store=store)
            context.notes = list(notes)
            if context.enabled:
                for index in indexes:
                    self.policy.screen_frame(
                        context,
                        read_image(processed_dir / frames[index].name),
                        role=f"output[{index}]",
                    )
                if context.watermark_enabled:
                    watermark_frames(
                        self.policy,
                        context,
                        [processed_dir / frame_path.name for frame_path in frames],
                    )
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
            result = self.policy.complete_output(context, request)
            if job is not None and store is not None:
                job.next_frame = len(frames)
                job.state = "done"
                store.save(job)
            return result
        except Exception as exc:
            if context is not None:
                self.policy.abort(context, exc)
            raise
        finally:
            if workspace is not None:
                workspace.cleanup()

    def _run_pipe(
        self,
        request: SwapRequest,
        job: JobState | None,
        store: JobStore | None,
    ) -> SwapResult:
        """Decode, process, and encode without writing a PNG for every frame."""

        context = None
        pipe: RawVideoPipe | None = None
        try:
            self._check_paths(request)
            info = probe(request.target_path)
            trim_start, trim_end = _trim_bounds(request, info.duration)
            width, height = even_size(info.width, info.height)
            start = int(job.next_frame) if job is not None and request.resume else 0
            estimated = max(1, int(round(info.duration * info.fps)))
            every = self.config.video.filter_every_n_frames
            samples = None
            if self.policy.enforcement_enabled(request):
                samples = _collect_pipe_samples(
                    request.target_path,
                    width=width,
                    height=height,
                    fps=info.fps,
                    trim_start=trim_start,
                    trim_end=trim_end,
                    every=every,
                    estimated=estimated,
                )
            context = self.policy.before_process(request, target_samples=samples)
            if start:
                context.notes.append(t("jobs.resume_from", frame=start, job=request.job_id or "-"))
            context.notes.append(t("video.pipe_mode"))
            trimmed = trim_start is not None or trim_end is not None
            if self.config.video.keep_audio and info.has_audio:
                audio_mode = "aac" if trimmed or start else "copy"
            else:
                audio_mode = "none"
            seek = (start / info.fps) if start else None
            pipe = RawVideoPipe(
                request.target_path,
                request.output_path,
                width=width,
                height=height,
                fps=info.fps,
                encoder=self.config.video.encoder,
                crf=self.config.video.crf,
                audio_source=request.target_path if audio_mode != "none" else None,
                audio_mode=audio_mode,
                trim_start=trim_start,
                trim_end=trim_end,
                seek=seek,
            )
            notes = _LockedNotes(list(context.notes))
            processors, shared = self._processors(request, notes)
            sample_set = set(sample_indexes(estimated, every))
            index = start
            produced = 0
            last_clean: np.ndarray | None = None
            while True:
                frame = pipe.read()
                if frame is None:
                    break
                if request.dry_run:
                    frame = _dry_run_banner(frame, t("processor.dry_run_banner"))
                for processor in processors:
                    frame = processor.process_frame(frame, shared)
                last_clean = frame.copy()
                if index in sample_set:
                    self.policy.screen_frame(context, frame, role=f"output[{index}]")
                frame = self.policy.mark_frame(context, frame)
                pipe.write(frame)
                produced += 1
                index += 1
                if job is not None and store is not None:
                    job.next_frame = index
                    job.total_frames = max(estimated, index)
                    store.save(job)
            for processor in processors:
                processor.post_process()
            if produced and context.enabled and last_clean is not None and (index - 1) not in sample_set:
                self.policy.screen_frame(context, last_clean, role=f"output[{index - 1}]")
            pipe.close()
            pipe = None
            if produced == 0:
                raise MediaToolError(t("video.no_frames", path=request.target_path))
            context.notes = list(notes)
            context.notes.append(t("video.audio_mode", mode=audio_mode))
            context.notes.append(t("video.frames_done", count=produced))
            result = self.policy.complete_output(context, request)
            if job is not None and store is not None:
                job.next_frame = index
                job.total_frames = index
                job.state = "done"
                store.save(job)
            return result
        except Exception as exc:
            if pipe is not None:
                pipe.abort()
            request.output_path.unlink(missing_ok=True)
            if context is not None:
                self.policy.abort(context, exc)
            raise

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

    def _processors(self, request: SwapRequest, notes: _LockedNotes):
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
        if self.config.swap.selector != "index" and self.config.video.threads > 1:
            notes.append(t("video.track_serial"))
        return processors, shared

    def _process_frames(
        self,
        request: SwapRequest,
        frames: list[Path],
        processed_dir: Path,
        notes: _LockedNotes,
        start_index: int = 0,
        job: JobState | None = None,
        store: JobStore | None = None,
    ) -> None:
        processors, shared = self._processors(request, notes)
        pending = [
            (index, frame_path)
            for index, frame_path in enumerate(frames)
            if index >= start_index or not (processed_dir / frame_path.name).is_file()
        ]
        workers = max(1, min(int(self.config.video.threads), len(pending) or 1))
        if self.config.swap.selector != "index" or job is not None:
            workers = 1

        def _one(item: tuple[int, Path]) -> None:
            index, frame_path = item
            frame = read_image(frame_path)
            if request.dry_run:
                frame = _dry_run_banner(frame, t("processor.dry_run_banner"))
            for processor in processors:
                frame = processor.process_frame(frame, shared)
            write_image(processed_dir / frame_path.name, frame)
            if job is not None and store is not None:
                job.next_frame = index + 1
                store.save(job)

        if workers == 1:
            for item in tqdm(pending, desc=t("video.progress"), disable=None):
                _one(item)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_one, item) for item in pending]
                for future in tqdm(as_completed(futures), total=len(futures), desc=t("video.progress"), disable=None):
                    future.result()
        for processor in processors:
            processor.post_process()


def watermark_frames(policy, context, frame_paths: list[Path]) -> int:
    """Watermark each frame that does not already have a ``.wm`` stamp.

    A resumed disk job skips frames stamped by the earlier attempt, so the
    bar is not drawn twice.
    """

    from barrett_faceswap.vision import read_image, write_image

    written = 0
    for path in frame_paths:
        stamp = Path(str(path) + ".wm")
        if stamp.is_file():
            continue
        marked = policy.mark_frame(context, read_image(path))
        write_image(path, marked)
        stamp.write_text("1\n", encoding="utf-8")
        written += 1
    return written


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


def _collect_pipe_samples(
    source: Path,
    *,
    width: int,
    height: int,
    fps: float,
    trim_start: float | None,
    trim_end: float | None,
    every: int,
    estimated: int,
) -> list[np.ndarray]:
    """First decode pass. Keeps sampled frames in memory, not on disk."""

    del fps
    chosen = set(sample_indexes(max(estimated, 1), every))
    samples: list[np.ndarray] = []
    last: np.ndarray | None = None
    index = 0
    for frame in iter_raw_frames(
        source,
        width=width,
        height=height,
        trim_start=trim_start,
        trim_end=trim_end,
        seek=None,
    ):
        if index in chosen:
            samples.append(frame)
        last = frame
        index += 1
    if last is not None and (not samples or samples[-1] is not last):
        samples.append(last.copy())
    return samples


def _dry_run_banner(image: np.ndarray, text: str) -> np.ndarray:
    canvas = image.copy()
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas
