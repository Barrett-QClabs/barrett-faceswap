"""Still-image pipeline. Compliance wraps every public run."""

from __future__ import annotations

import cv2
import numpy as np
from tqdm import tqdm

from barrett_faceswap.compliance.policy import CompliancePolicy
from barrett_faceswap.config import AppConfig
from barrett_faceswap.i18n import t
from barrett_faceswap.processors import available_processors, create_processor
from barrett_faceswap.types import ProcessContext, SwapRequest, SwapResult
from barrett_faceswap.vision import read_image


class FaceSwapPipeline:
    """Detect, align, swap, and paste, with the compliance policy around it."""

    def __init__(self, config: AppConfig, admin_path=None) -> None:
        self.config = config
        self.policy = CompliancePolicy(config, admin_path=admin_path)

    def run(self, request: SwapRequest) -> SwapResult:
        context = None
        try:
            context = self.policy.before_process(request)
            frame = self._execute(request, context.notes)
            return self.policy.after_process(context, frame, request)
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
