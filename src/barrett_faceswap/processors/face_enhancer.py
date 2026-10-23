"""GFPGAN 1.4 or CodeFormer on each detected face.

The processor is registered as ``face_enhancer``. A dry-run does not open
weights. A real run requires the detector and the selected enhancer file.
Missing weights raise ``ModelNotFoundError``; nothing is downloaded. Faces
are aligned to the FFHQ template, restored, blended, and pasted back. A frame
with no face is returned unchanged.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from barrett_faceswap.align import align_face, ffhq_template
from barrett_faceswap.blend import paste_back
from barrett_faceswap.config import AppConfig
from barrett_faceswap.detect import FaceDetector
from barrett_faceswap.enhance import blend_images, run_enhancer
from barrett_faceswap.eyes import eye_mask, eye_openness
from barrett_faceswap.errors import NoFaceError, ProcessorError
from barrett_faceswap.i18n import t
from barrett_faceswap.masks import region_mask
from barrett_faceswap.model_registry import registry_record
from barrett_faceswap.models import ModelStore
from barrett_faceswap.onnx_io import session_hw
from barrett_faceswap.processors.registry import register_processor
from barrett_faceswap.types import ProcessContext

ENHANCER_ROLES = {
    "gfpgan_1.4": "gfpgan",
    "codeformer": "codeformer",
}


class FaceEnhancer:
    """Restore aligned faces with the configured ONNX enhancer."""

    name = "face_enhancer"

    def __init__(self, config: AppConfig, providers: list[str]) -> None:
        self.config = config
        self.providers = providers
        self.store: ModelStore | None = None
        self.detector_session = None
        self.enhancer_session = None
        self.role = ENHANCER_ROLES.get(config.enhancer.model, "")
        self._lock = threading.Lock()
        self._miss_noted = False
        self._detector_obj: FaceDetector | None = None
        self._previous: list[tuple[np.ndarray, np.ndarray]] = []

    def pre_check(self, ctx: ProcessContext) -> None:
        if ctx.dry_run:
            ctx.notes.append(t("processor.enhancer_dry_run", model=self.config.enhancer.model))
            return
        if self.config.enhancer.model not in ENHANCER_ROLES:
            raise ProcessorError(t("processor.enhancer_unknown", model=self.config.enhancer.model))
        self.role = ENHANCER_ROLES[self.config.enhancer.model]
        self.store = ModelStore(self.config, self.providers)
        self.store.ensure_present(("detector", self.role))
        self.detector_session = self.store.session("detector")
        self.enhancer_session = self.store.session(self.role)

    def process_frame(self, frame: np.ndarray, ctx: ProcessContext) -> np.ndarray:
        if ctx.dry_run:
            return frame
        if self.store is None or self.enhancer_session is None or self.detector_session is None:
            self.pre_check(ctx)
        assert self.enhancer_session is not None
        assert self.detector_session is not None
        if ctx.known_faces is not None:
            faces = list(ctx.known_faces)
        else:
            try:
                faces = self._detector().detect(frame)
            except NoFaceError:
                self._note_miss(ctx)
                return frame
        if not faces:
            self._note_miss(ctx)
            return frame
        record = registry_record(self.config, self.role)
        network_h, network_w = session_hw(self.enhancer_session, record.input_size)
        side = max(network_h, network_w)
        output = frame
        carried: list[tuple[np.ndarray, np.ndarray]] = []
        for index, face in enumerate(faces):
            crop, matrix = align_face(output, face.landmarks[:5], side, template=ffhq_template(side))
            restored = run_enhancer(self.enhancer_session, crop, weight=self.config.enhancer.weight)
            eyes = eye_mask(side, ffhq_template(side), self.config.blend.mask_blur)
            openness = ctx.eye_openness if ctx.eye_openness is not None else eye_openness(face.landmarks)
            # Open eyes keep more of the target crop. eye_blend 1 still means full restoration.
            eye_amount = float(self.config.enhancer.eye_blend) * (1.0 - 0.7 * float(openness))
            blended = blend_images(
                restored,
                crop,
                self.config.enhancer.blend,
                eye_mask=eyes,
                eye_amount=eye_amount,
            )
            blended = self._stabilize(index, face, blended)
            carried.append((blended, np.asarray(face.landmarks[:5], dtype=np.float32).copy()))
            mask = region_mask(side, self.config.blend.mask_blur)
            output = paste_back(output, blended, matrix, mask)
        self._previous = carried
        if ctx.log_every_frame:
            ctx.notes.append(
                t(
                    "processor.enhancer_done",
                    model=self.config.enhancer.model,
                    faces=len(faces),
                    blend=f"{self.config.enhancer.blend:.2f}",
                )
            )
        return output

    def post_process(self) -> None:
        self._previous = []

    def _stabilize(self, index: int, face, blended: np.ndarray) -> np.ndarray:
        """Mix the previous enhanced crop when the landmarks barely moved."""

        amount = float(self.config.enhancer.temporal)
        if amount <= 0 or index >= len(self._previous):
            return blended
        previous, previous_landmarks = self._previous[index]
        if _motion(previous_landmarks, face) > float(self.config.enhancer.temporal_motion):
            return blended
        if previous.shape[:2] != blended.shape[:2]:
            previous = cv2.resize(previous, (blended.shape[1], blended.shape[0]), interpolation=cv2.INTER_LINEAR)
        return blend_images(previous, blended, amount)

    def _detector(self) -> FaceDetector:
        if self._detector_obj is None:
            assert self.detector_session is not None
            self._detector_obj = FaceDetector(
                self.detector_session,
                det_size=self.config.detect.det_size,
                score_threshold=self.config.detect.score_threshold,
                nms_threshold=self.config.detect.nms_threshold,
            )
        return self._detector_obj

    def _note_miss(self, ctx: ProcessContext) -> None:
        with self._lock:
            if ctx.log_every_frame or not self._miss_noted:
                ctx.notes.append(t("processor.enhancer_no_face"))
                self._miss_noted = True


def _motion(previous: np.ndarray, face) -> float:
    current = np.asarray(face.landmarks[:5], dtype=np.float32)
    earlier = np.asarray(previous, dtype=np.float32).reshape(-1, 2)[:5]
    if current.shape != earlier.shape:
        return 1.0
    shift = float(np.linalg.norm(current - earlier, axis=1).mean())
    width = max(float(face.bbox[2]) - float(face.bbox[0]), 1.0)
    return shift / width


@register_processor("face_enhancer")
def _build(config: AppConfig, providers: list[str]) -> FaceEnhancer:
    return FaceEnhancer(config, providers)
