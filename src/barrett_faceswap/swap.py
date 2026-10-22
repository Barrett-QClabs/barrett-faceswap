"""ONNX identity swap.

The recognizer consumes a 112 pixel aligned crop and emits a 512-d embedding.
The swapper consumes that embedding, after the graph's ``emap`` matrix, plus
a 128 pixel aligned target crop. Input names are read from the session so a
renamed graph still binds as long as the ranks stay the same.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from barrett_faceswap.align import align_face
from barrett_faceswap.errors import SwapError
from barrett_faceswap.i18n import t
from barrett_faceswap.models import ModelStore

logger = logging.getLogger(__name__)


class IdentitySwapper:
    """Recognizer plus inswapper-style swapper, both loaded from :class:`ModelStore`."""

    def __init__(self, store: ModelStore, swap_resolution: int) -> None:
        self.store = store
        self.swap_resolution = int(swap_resolution)
        self._emap: np.ndarray | None = None

    def embed(self, source_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        session = self.store.session("recognizer")
        size = _spatial_size(session, fallback=112)
        crop, _matrix = align_face(source_bgr, landmarks, size)
        blob = _recognizer_blob(crop)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: blob})[0]
        vector = np.asarray(output, dtype=np.float32).reshape(-1)
        if vector.shape[0] != 512:
            raise SwapError(t("swap.embedding_length", length=vector.shape[0]))
        norm = float(np.linalg.norm(vector)) + 1e-8
        return (vector / norm).astype(np.float32)

    def forward(self, target_crop_bgr: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        session = self.store.session("swapper")
        latent = self._map_embedding(embedding)
        image_name, latent_name = _bind_swap_inputs(session)
        blob = _swapper_blob(target_crop_bgr)
        prediction = session.run(None, {latent_name: latent, image_name: blob})[0]
        array = np.asarray(prediction, dtype=np.float32)
        if array.ndim != 4:
            raise SwapError(t("swap.output_rank", rank=array.ndim))
        image = np.transpose(array[0], (1, 2, 0))
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def _map_embedding(self, embedding: np.ndarray) -> np.ndarray:
        emap = self._load_emap()
        latent = np.asarray(embedding, dtype=np.float32).reshape(1, 512)
        mapped = latent @ emap
        mapped = mapped / (np.linalg.norm(mapped) + 1e-8)
        return mapped.astype(np.float32)

    def _load_emap(self) -> np.ndarray:
        if self._emap is not None:
            return self._emap
        path = self.store.paths()["swapper"]
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError as exc:
            raise SwapError(t("swap.need_onnx")) from exc
        model = onnx.load(str(path))
        for initializer in model.graph.initializer:
            if initializer.name != "emap":
                continue
            array = numpy_helper.to_array(initializer).astype(np.float32)
            if array.size != 512 * 512:
                raise SwapError(t("swap.emap_size", size=array.size, expected=512 * 512))
            self._emap = array.reshape(512, 512)
            return self._emap
        raise SwapError(t("swap.emap_missing", name=path.name))


def _bind_swap_inputs(session) -> tuple[str, str]:
    image_name = None
    latent_name = None
    for item in session.get_inputs():
        shape = list(item.shape)
        if len(shape) == 4:
            image_name = item.name
        elif len(shape) == 2:
            latent_name = item.name
    if image_name is None or latent_name is None:
        found = [(item.name, list(item.shape)) for item in session.get_inputs()]
        raise SwapError(t("swap.inputs", found=found))
    return image_name, latent_name


def _spatial_size(session, fallback: int) -> int:
    shape = session.get_inputs()[0].shape
    if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        return int(shape[2])
    return fallback


def _recognizer_blob(crop_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 128.0
    return np.transpose(rgb, (2, 0, 1))[None, ...]


def _swapper_blob(crop_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = rgb / 255.0
    return np.transpose(rgb, (2, 0, 1))[None, ...]
