"""Compose box, parsing, and occlusion masks for one aligned crop.

Model files are optional. A missing occlusion graph keeps the all-ones stub.
A missing parser uses a feathered landmark hull when the caller passes crop
landmarks, and the ellipse otherwise. Both still go through :func:`merge_min`.

Video frames reuse the previous crop mask when the face barely moves.
Occlusion and parsing, when both are loaded, run on two threads. The crop is
already aligned, so the reuse warp is the change of alignment.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from barrett_faceswap.config import AppConfig
from barrett_faceswap.face_parsing import eye_mask_from_labels, mask_from_labels, parser_labels
from barrett_faceswap.masks import composite_mask, occlusion_mask, parser_stub_mask
from barrett_faceswap.models import ModelStore
from barrett_faceswap.occlusion import refine_occlusion, run_occlusion

_OCCLUDER_ROLE = {
    "xseg_1": "occluder",
    "xseg_2": "xseg_2",
    "xseg_3": "xseg_3",
    "sam2_refine": "occluder",
}


class MaskStack:
    """Lazy ONNX masks for the swap paste. ``load`` runs on the calling thread."""

    def __init__(self, config: AppConfig, store: ModelStore | None) -> None:
        self.config = config
        self.store = store
        self._occlusion = None
        self._parser = None
        self._sam = None
        self._ready = False
        self.eye_mask: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self._cached: np.ndarray | None = None
        self._cached_eyes: np.ndarray | None = None
        self._cached_points: np.ndarray | None = None
        self._cached_matrix: np.ndarray | None = None
        self._since = 0

    def load(self) -> None:
        """Open mask sessions when the files exist. Missing files stay on the stub."""

        if self._ready:
            return
        self._ready = True
        if self.store is None:
            return
        settings = self.config.masks
        role = _OCCLUDER_ROLE.get(settings.occluder_model, "occluder")
        if settings.occlusion and self.store.paths()[role].is_file():
            self._occlusion = self.store.session(role)
        if settings.occluder_model == "sam2_refine" and self.store.paths()["sam2_refine"].is_file():
            self._sam = self.store.session("sam2_refine")
        if settings.parser and self.store.paths()["parser"].is_file():
            self._parser = self.store.session("parser")

    def reset(self) -> None:
        """Drop the video mask cache at the end of a run."""

        self._cached = None
        self._cached_eyes = None
        self._cached_points = None
        self._cached_matrix = None
        self._since = 0
        self.eye_mask = None
        self.labels = None

    def compose(
        self,
        crop_bgr,
        size: int,
        blur: int,
        landmarks=None,
        *,
        frame_points=None,
        matrix=None,
        sequential: bool = False,
        face_width: float | None = None,
    ):
        """Box, region, and occlusion, merged by the per-pixel minimum.

        ``landmarks`` are the five template points in the aligned crop. They
        are used only when the parsing file is absent. ``sequential`` is set
        for video and live so a small landmark move reuses the last mask.
        """

        self.load()
        if self._reuse(frame_points, sequential, face_width):
            assert self._cached is not None
            warped = _warp_crop_mask(self._cached, self._cached_matrix, matrix, size)
            if self._cached_eyes is None:
                self.eye_mask = None
            else:
                self.eye_mask = _warp_crop_mask(self._cached_eyes, self._cached_matrix, matrix, size)
            return warped
        occlusion, region, eyes = self._infer(crop_bgr, size, blur, landmarks)
        merged = composite_mask(size, blur, occlusion=occlusion, region=region)
        self.eye_mask = eyes
        self._cached = merged
        self._cached_eyes = None if eyes is None else eyes.copy()
        self._cached_points = None if frame_points is None else np.asarray(frame_points, dtype=np.float32).reshape(-1, 2)[:5].copy()
        self._cached_matrix = None if matrix is None else np.asarray(matrix, dtype=np.float32).reshape(2, 3).copy()
        self._since = 0
        return merged

    def _reuse(self, frame_points, sequential: bool, face_width: float | None) -> bool:
        stride = int(self.config.masks.stride)
        if not sequential or stride <= 1 or self._cached is None:
            return False
        self._since += 1
        if self._since >= stride:
            return False
        if frame_points is None or self._cached_points is None:
            return True
        if _motion(self._cached_points, frame_points, face_width) > float(self.config.masks.stride_motion):
            return False
        return True

    def _infer(self, crop_bgr, size: int, blur: int, landmarks):
        settings = self.config.masks
        if self._occlusion is not None and self._parser is not None:
            with ThreadPoolExecutor(max_workers=2) as pool:
                occlusion_job = pool.submit(
                    run_occlusion,
                    self._occlusion,
                    crop_bgr,
                    size,
                    blur,
                    invert=settings.occlusion_invert,
                )
                parser_job = pool.submit(self._parse, crop_bgr, size, blur)
                occlusion = occlusion_job.result()
                region, eyes = parser_job.result()
            return self._maybe_refine(crop_bgr, size, occlusion), region, eyes
        if self._occlusion is not None:
            occlusion = run_occlusion(
                self._occlusion,
                crop_bgr,
                size,
                blur,
                invert=settings.occlusion_invert,
            )
        else:
            path = None if self.store is None else str(self.store.paths()["occluder"])
            occlusion = occlusion_mask(size, path)
        if self._parser is not None:
            region, eyes = self._parse(crop_bgr, size, blur)
        else:
            path = None if self.store is None else str(self.store.paths()["parser"])
            region = parser_stub_mask(size, blur, path, landmarks=landmarks)
            eyes = None
            self.labels = None
        return self._maybe_refine(crop_bgr, size, occlusion), region, eyes

    def _maybe_refine(self, crop_bgr, size: int, occlusion: np.ndarray) -> np.ndarray:
        if self._sam is None:
            return occlusion
        return refine_occlusion(self._sam, crop_bgr, occlusion, size)

    def _parse(self, crop_bgr, size: int, blur: int):
        labels = parser_labels(self._parser, crop_bgr, size)
        self.labels = labels
        region = mask_from_labels(labels, list(self.config.masks.regions), blur)
        return region, eye_mask_from_labels(labels, blur)


def _motion(previous, current, face_width: float | None) -> float:
    earlier = np.asarray(previous, dtype=np.float32).reshape(-1, 2)[:5]
    points = np.asarray(current, dtype=np.float32).reshape(-1, 2)[:5]
    if earlier.shape != points.shape:
        return 1.0
    shift = float(np.linalg.norm(points - earlier, axis=1).mean())
    width = float(face_width) if face_width else float(np.linalg.norm(points[1] - points[0]))
    return shift / max(width, 1.0)


def _warp_crop_mask(mask: np.ndarray, previous_matrix, matrix, size: int) -> np.ndarray:
    """Carry a crop mask onto the current alignment.

    ``matrix`` maps frame coordinates onto the crop. The composed affine
    samples the previous mask at the current crop pixel.
    """

    current = np.asarray(mask, dtype=np.float32)
    if current.shape[:2] != (size, size):
        current = cv2.resize(current, (size, size), interpolation=cv2.INTER_LINEAR)
    if previous_matrix is None or matrix is None:
        return current
    previous = np.eye(3, dtype=np.float64)
    following = np.eye(3, dtype=np.float64)
    previous[:2] = np.asarray(previous_matrix, dtype=np.float64).reshape(2, 3)
    following[:2] = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    try:
        composed = previous @ np.linalg.inv(following)
    except np.linalg.LinAlgError:
        return current
    affine = composed[:2].astype(np.float32)
    return cv2.warpAffine(
        current,
        affine,
        (size, size),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
