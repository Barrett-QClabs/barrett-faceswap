"""Realtime capture, detect, swap, and output.

Three threads share bounded queues. Capture keeps only the newest frames.
Detection runs every few frames; the cache carries the landmarks forward.
The source embedding is prepared once. Compliance runs at the start of the
session, watermarks each output frame when enforcement is on, and writes one
audit line when the session ends.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from barrett_faceswap.commercial import guard_commercial
from barrett_faceswap.compliance.policy import CompliancePolicy
from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import MediaToolError, NoFaceError
from barrett_faceswap.face_cache import DetectionCache
from barrett_faceswap.i18n import t
from barrett_faceswap.processors import create_processor
from barrett_faceswap.queues import LatestQueue
from barrett_faceswap.recover import LandmarkSmoother
from barrett_faceswap.types import ProcessContext, SwapRequest, SwapResult

logger = logging.getLogger(__name__)


@dataclass
class LiveStats:
    """Counters a benchmark or a test can read after the session."""

    frames_in: int = 0
    frames_out: int = 0
    dropped: int = 0
    stage_s: dict[str, float] = field(default_factory=lambda: {"capture": 0.0, "detect": 0.0, "swap": 0.0, "output": 0.0})
    prepared: int = 0


class LoopbackWriter:
    """Write raw BGR frames to a v4l2 device through ffmpeg.

    A missing device is logged once. The preview path still runs.
    """

    def __init__(self, device: str, fps: float = 30.0) -> None:
        self.device = device
        self.fps = float(fps)
        self._proc: subprocess.Popen | None | bool = None

    def write(self, frame: np.ndarray) -> None:
        if self._proc is False:
            return
        if self._proc is None:
            if not Path(self.device).exists():
                logger.warning(t("live.v4l2_missing", device=self.device))
                self._proc = False
                return
            height, width = frame.shape[:2]
            command = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{self.fps:.3f}",
                "-i",
                "pipe:0",
                "-f",
                "v4l2",
                self.device,
            ]
            self._proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert isinstance(self._proc, subprocess.Popen)
        if self._proc.stdin is None:
            return
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if not isinstance(self._proc, subprocess.Popen):
            return
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait(timeout=5)


class LiveSession:
    """One webcam or synthetic-frame session."""

    def __init__(
        self,
        config: AppConfig,
        request: SwapRequest,
        *,
        processors: list | None = None,
        policy: CompliancePolicy | None = None,
        hint_source=None,
        hint_target=None,
    ) -> None:
        self.config = config
        self.request = request
        self.policy = policy or CompliancePolicy(config)
        self.stats = LiveStats()
        self._context = None
        self._processors = processors
        self._hint_source = hint_source
        self._hint_target = hint_target
        self._shared: ProcessContext | None = None
        self._cache = DetectionCache(config.live.detect_every, iou_threshold=config.swap.track_iou)
        self._smoother = LandmarkSmoother(str(config.video.smooth))
        self._screen_every = max(1, int(config.video.filter_every_n_frames))
        self._started = False

    def start(self, first_frame: np.ndarray) -> None:
        """Consent, filter the source and the first frame, prepare the embedding once."""

        if self._started:
            return
        guard_commercial(self.config, self.request.processors)
        self._context = self.policy.before_process(self.request, target_samples=[first_frame])
        names = list(self.request.processors) if self.request.processors else list(self.config.processors)
        providers = list(self.request.execution_providers) if self.request.execution_providers else list(
            self.config.execution.providers
        )
        if self._processors is None:
            self._processors = [create_processor(name, self.config, providers) for name in names]
        notes = self._context.notes
        self._shared = ProcessContext(
            source_bgr=_source_image(self.request),
            dry_run=self.request.dry_run,
            target_face_index=self.request.target_face_index,
            notes=notes,
            log_every_frame=False,
            source_face=self._hint_source,
            reuse_masks=True,
            allow_missing_face=True,
        )
        for processor in self._processors:
            processor.pre_check(self._shared)
        for processor in self._processors:
            prepare = getattr(processor, "prepare_source", None)
            if prepare is not None:
                prepare(self._shared)
                self.stats.prepared += 1
        self._started = True
        notes.append(t("live.started"))

    def step(self, frame: np.ndarray) -> np.ndarray:
        """Detect or interpolate, swap, filter on a stride, and watermark."""

        if not self._started:
            self.start(frame)
        assert self._shared is not None and self._context is not None
        self.stats.frames_in += 1
        started = time.perf_counter()
        if self.request.dry_run:
            output = frame
            self.stats.stage_s["detect"] += 0.0
        else:
            detect_started = time.perf_counter()
            if self._cache.wants_detection():
                try:
                    detected = _detect_with(self._processors, frame)
                except NoFaceError:
                    detected = []
                if not detected and self._hint_target is not None:
                    detected = [self._hint_target]
                faces = self._cache.observe(detected if detected else [])
            else:
                faces = self._cache.observe(None)
            self.stats.stage_s["detect"] += time.perf_counter() - detect_started
            faces = self._smoother.apply(faces)
            self._shared.known_faces = faces
            output = frame
            for processor in self._processors or []:
                output = processor.process_frame(output, self._shared)
        self.stats.stage_s["swap"] += time.perf_counter() - started
        if self.stats.frames_out % self._screen_every == 0:
            self.policy.screen_frame(self._context, output, role=f"output[{self.stats.frames_out}]")
        output = self.policy.mark_frame(self._context, output)
        self.stats.frames_out += 1
        return output

    def finish(self) -> SwapResult:
        """One audit line for the session. Safe to call once."""

        for processor in self._processors or []:
            processor.post_process()
        if self._context is None:
            self._context = self.policy.before_process(self.request, target_samples=[np.zeros((16, 16, 3), np.uint8)])
        self._context.notes.append(t("live.frames_done", count=self.stats.frames_out, dropped=self.stats.dropped))
        return self.policy.complete_output(self._context, self.request)


class LivePipeline:
    """Thread the capture, the swap, and the output."""

    def __init__(self, config: AppConfig, admin_path: Path | None = None) -> None:
        self.config = config
        self.admin_path = admin_path

    def run(
        self,
        request: SwapRequest,
        frames,
        *,
        writer=None,
        preview: bool | None = None,
        processors: list | None = None,
        hint_source=None,
        hint_target=None,
    ) -> tuple[SwapResult, LiveStats]:
        session = LiveSession(
            self.config,
            request,
            processors=processors,
            policy=CompliancePolicy(self.config, admin_path=self.admin_path),
            hint_source=hint_source,
            hint_target=hint_target,
        )
        show = self.config.live.preview if preview is None else preview
        incoming: LatestQueue = LatestQueue(self.config.live.queue_size)
        outgoing: LatestQueue = LatestQueue(self.config.live.queue_size)
        capture_done = threading.Event()
        process_done = threading.Event()
        failed: list[BaseException] = []
        max_frames = int(request.live_frames or 0)

        def _capture() -> None:
            started = time.perf_counter()
            try:
                for frame in frames:
                    incoming.put_latest(frame)
            except BaseException as exc:  # noqa: BLE001 - surfaced after the threads join
                failed.append(exc)
            finally:
                session.stats.stage_s["capture"] += time.perf_counter() - started
                capture_done.set()

        def _process() -> None:
            try:
                while not failed:
                    item = incoming.get(0.05)
                    if item is None:
                        if capture_done.is_set() and incoming.empty():
                            return
                        continue
                    produced = session.step(np.asarray(item))
                    outgoing.put_latest(produced)
                    if max_frames and session.stats.frames_out >= max_frames:
                        return
            except BaseException as exc:  # noqa: BLE001
                failed.append(exc)
            finally:
                process_done.set()

        def _output() -> None:
            try:
                while not failed:
                    item = outgoing.get(0.05)
                    if item is None:
                        if process_done.is_set() and outgoing.empty():
                            return
                        continue
                    started = time.perf_counter()
                    frame = np.asarray(item)
                    if writer is not None:
                        writer(frame)
                    _preview(frame, enabled=show)
                    session.stats.stage_s["output"] += time.perf_counter() - started
            except BaseException as exc:  # noqa: BLE001
                failed.append(exc)

        workers = [
            threading.Thread(target=_capture, name="barrett-capture"),
            threading.Thread(target=_process, name="barrett-swap"),
            threading.Thread(target=_output, name="barrett-output"),
        ]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            if failed:
                if session._context is not None:
                    session.policy.abort(session._context, failed[0])
                raise failed[0]
            if session.stats.frames_out < 1:
                raise MediaToolError(t("live.no_frames"))
            session.stats.dropped = incoming.dropped + outgoing.dropped
            if session._context is not None:
                session._context.notes.append(t("live.dropped", count=session.stats.dropped))
            return session.finish(), session.stats
        finally:
            if writer is not None and hasattr(writer, "close"):
                writer.close()


def iter_camera(index: int, max_frames: int):
    """Yield frames from a webcam. Tests do not call this."""

    import cv2

    capture = cv2.VideoCapture(int(index))
    if not capture.isOpened():
        raise MediaToolError(t("live.camera_unavailable", index=index))
    count = 0
    try:
        while max_frames <= 0 or count < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            yield frame
            count += 1
    finally:
        capture.release()


def _source_image(request: SwapRequest) -> np.ndarray:
    from barrett_faceswap.vision import read_image

    return read_image(request.source_path)


def _detect_with(processors: list | None, frame: np.ndarray):
    for processor in processors or []:
        detect = getattr(processor, "detect_faces", None)
        if detect is not None:
            return detect(frame)
    return []


def _preview(frame: np.ndarray, *, enabled: bool) -> None:
    if not enabled:
        return
    if not os.environ.get("DISPLAY"):
        logger.info(t("live.preview_headless"))
        return
    import cv2

    cv2.imshow(t("live.window"), frame)
    cv2.waitKey(1)
