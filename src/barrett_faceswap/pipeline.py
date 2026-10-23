"""Still-image pipeline. Compliance wraps every public run."""

from __future__ import annotations

import cv2
import numpy as np
from tqdm import tqdm

from barrett_faceswap.compliance.policy import CompliancePolicy
from barrett_faceswap.config import AppConfig, apply_request_overrides
from barrett_faceswap.errors import ConfigError
from barrett_faceswap.i18n import t
from barrett_faceswap.jobs import JobState, JobStore, open_for_run
from barrett_faceswap.processors import available_processors, create_processor
from barrett_faceswap.types import ProcessContext, SwapRequest, SwapResult
from barrett_faceswap.vision import is_video, read_image


class FaceSwapPipeline:
    """Detect, align, swap, and paste, with the compliance policy around it."""

    def __init__(self, config: AppConfig, admin_path=None) -> None:
        self.config = config
        self.policy = CompliancePolicy(config, admin_path=admin_path)

    def run(self, request: SwapRequest) -> SwapResult:
        self.config = apply_request_overrides(self.config, request)
        store = JobStore(self.config.jobs.root)
        job = open_for_run(store, request)
        try:
            if is_video(request.target_path):
                from barrett_faceswap.video import VideoPipeline

                return VideoPipeline(self.config, admin_path=self.policy.admin_path).run(request, job=job, store=store)
            if request.trim_start is not None or request.trim_end is not None or request.video_encoder is not None:
                raise ConfigError(t("video.flags_need_video"))
            if request.video_mode == "pipe":
                raise ConfigError(t("video.pipe_needs_video"))
            return self._run_image(request, job, store)
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
        names = list(request.processors) if request.processors else list(self.config.processors)
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
