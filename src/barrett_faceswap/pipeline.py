"""Still-image pipeline. Compliance wraps every public run."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from tqdm import tqdm

from barrett_faceswap.compliance.policy import CompliancePolicy
from barrett_faceswap.config import AppConfig, apply_request_overrides
from barrett_faceswap.errors import ConfigError
from barrett_faceswap.i18n import t
from barrett_faceswap.jobs import JobState, JobStore, open_for_run
from barrett_faceswap.processors import available_processors, create_processor, resolve_processor_names
from barrett_faceswap.types import ProcessContext, SwapRequest, SwapResult
from barrett_faceswap.vision import is_video, read_image


class FaceSwapPipeline:
    """Detect, align, swap, and paste, with the compliance policy around it."""

    def __init__(self, config: AppConfig, admin_path=None, voice_converter=None) -> None:
        self.config = config
        self.policy = CompliancePolicy(config, admin_path=admin_path)
        # Tests pass a stand-in. None loads OpenVoice when a real conversion runs.
        self.voice_converter = voice_converter

    def run(
        self,
        request: SwapRequest,
        on_progress: Callable[[int, int], None] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> SwapResult:
        self.config = apply_request_overrides(self.config, request)
        from barrett_faceswap.commercial import guard_commercial

        guard_commercial(self.config, request.processors)
        store = JobStore(self.config.jobs.root)
        job = open_for_run(store, request)
        try:
            if is_video(request.target_path):
                from barrett_faceswap.video import VideoPipeline
                from barrett_faceswap.voice import apply_voice, check_voice

                check_voice(self.config, request, converter=self.voice_converter)
                result = VideoPipeline(self.config, admin_path=self.policy.admin_path).run(
                    request,
                    job=job,
                    store=store,
                    on_progress=on_progress,
                )
                return apply_voice(
                    self.config,
                    request,
                    result,
                    converter=self.voice_converter,
                    on_stage=on_stage,
                )
            if request.trim_start is not None or request.trim_end is not None or request.video_encoder is not None:
                raise ConfigError(t("video.flags_need_video"))
            if request.video_mode == "pipe":
                raise ConfigError(t("video.pipe_needs_video"))
            result = self._run_image(request, job, store)
            from barrett_faceswap.voice import apply_voice

            return apply_voice(self.config, request, result, converter=self.voice_converter)
        except Exception as exc:
            if job is not None and job.state != "done":
                job.state = "interrupted"
                job.message = str(exc)
                store.save(job)
            raise

    def _run_image(self, request: SwapRequest, job: JobState | None, store: JobStore) -> SwapResult:
        context = None
        try:
            context = self.policy.before_process(request)
            frame = self._execute(request, context.notes)
            result = self.policy.after_process(context, frame, request)
            if job is not None:
                job.next_frame = 1
                job.total_frames = 1
                job.state = "done"
                store.save(job)
            return result
        except Exception as exc:
            if context is not None:
                self.policy.abort(context, exc)
            raise

    def _execute(self, request: SwapRequest, notes: list[str]) -> np.ndarray:
        source = read_image(request.source_path)
        frame = read_image(request.target_path)
        if request.dry_run:
            frame = _dry_run_banner(frame, t("processor.dry_run_banner"))
        names = resolve_processor_names(self.config, request)
        providers = list(request.execution_providers) if request.execution_providers else list(
            self.config.execution.providers
        )
        processors = [create_processor(name, self.config, providers) for name in names]
        context = ProcessContext(
            source_bgr=source,
            dry_run=request.dry_run,
            target_face_index=request.target_face_index,
            notes=notes,
        )
        for processor in processors:
            processor.pre_check(context)
        for processor in tqdm(processors, desc=t("pipeline.progress"), disable=None):
            frame = processor.process_frame(frame, context)
        for processor in processors:
            processor.post_process()
        return frame


def list_processor_names() -> list[str]:
    return available_processors()


def _dry_run_banner(image: np.ndarray, text: str) -> np.ndarray:
    canvas = image.copy()
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas
