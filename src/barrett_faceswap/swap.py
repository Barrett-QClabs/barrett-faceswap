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
        return embed_arcface(self.store, source_bgr, landmarks)

    def forward(self, target_crop_bgr: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        session = self.store.session("swapper")
        latent = self._map_embedding(embedding)
        image_name, latent_name = _bind_swap_inputs(session)
        blob = _swapper_blob(target_crop_bgr)
        from barrett_faceswap.binding import run_feeds

        prediction = run_feeds(
            session,
            {latent_name: latent, image_name: blob},
            cuda_graph=bool(self.store.config.execution.cuda_graph),
        )[0]
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
        self._emap = read_emap(self.store.paths()["swapper"])
        return self._emap


def read_emap(path) -> np.ndarray:
    """Read the 512×512 inswapper matrix from an ONNX file.

    Official ``inswapper_128.onnx`` stores it as the last initializer, named
    ``buff2fs``. A graph that names the tensor ``emap`` still wins.
    """

    from pathlib import Path

    file_path = Path(path)
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:
        raise SwapError(t("swap.need_onnx")) from exc
    model = onnx.load(str(file_path))
    chosen = None
    for initializer in model.graph.initializer:
        if initializer.name == "emap":
            chosen = initializer
            break
    if chosen is None and len(model.graph.initializer):
        chosen = model.graph.initializer[-1]
    if chosen is None:
        raise SwapError(t("swap.emap_missing", name=file_path.name))
    array = numpy_helper.to_array(chosen).astype(np.float32)
    if array.size != 512 * 512:
        raise SwapError(t("swap.emap_size", size=array.size, expected=512 * 512))
    return array.reshape(512, 512)


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
    return _normalize_blob(crop_bgr, "zero_one")


def embed_arcface(store: ModelStore, image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """L2-normalized 512-d ArcFace embedding for one set of landmarks."""

    session = store.session("recognizer")
    size = _spatial_size(session, fallback=112)
    crop, _matrix = align_face(image_bgr, landmarks, size)
    blob = _recognizer_blob(crop)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: blob})[0]
    vector = np.asarray(output, dtype=np.float32).reshape(-1)
    if vector.shape[0] != 512:
        raise SwapError(t("swap.embedding_length", length=vector.shape[0]))
    norm = float(np.linalg.norm(vector)) + 1e-8
    return (vector / norm).astype(np.float32)


class GraphSwapper:
    """Swapper that takes a 4-D image and a 512-d latent, plus an optional converter.

    Inswapper stays on :class:`IdentitySwapper` because its latent is
    ``embedding @ emap`` from an initializer, not a second ONNX file.
    HyperSwap, SimSwap, and Ghost use this class. A crossface graph, when the
    spec names one, maps ArcFace into that swapper's embedding space.
    """

    def __init__(self, store: ModelStore, spec) -> None:
        self.store = store
        self.spec = spec
        self.native_size = int(spec.native_size)

    def project(self, arcface: np.ndarray) -> np.ndarray:
        """Return the latent the swap graph consumes."""

        if self.spec.embedding != "crossface" or not self.spec.crossface_role:
            vector = np.asarray(arcface, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector)) + 1e-8
            return (vector / norm).astype(np.float32)
        session = self.store.session(self.spec.crossface_role)
        return _run_vector(session, arcface)

    def forward(self, target_crop_bgr: np.ndarray, latent: np.ndarray) -> np.ndarray:
        session = self.store.session(self.spec.role)
        image_name, latent_name = _bind_swap_inputs(session)
        blob = _normalize_blob(target_crop_bgr, self.spec.normalize)
        vector = np.asarray(latent, dtype=np.float32).reshape(1, -1)
        from barrett_faceswap.binding import run_feeds

        prediction = run_feeds(
            session,
            {latent_name: vector, image_name: blob},
            cuda_graph=bool(self.store.config.execution.cuda_graph),
        )[0]
        array = np.asarray(prediction, dtype=np.float32)
        if array.ndim != 4:
            raise SwapError(t("swap.output_rank", rank=array.ndim))
        return _bgr_from_nchw(array)


def _normalize_blob(crop_bgr: np.ndarray, mode: str) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    if mode == "minus_one_one":
        rgb = (rgb / 255.0 - 0.5) / 0.5
    elif mode == "zero_one":
        rgb = rgb / 255.0
    else:
        raise SwapError(t("swap.normalize", mode=mode))
    return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)


def _bgr_from_nchw(array: np.ndarray) -> np.ndarray:
    """Decode a swapper image. Ranges match the enhancer helper."""

    from barrett_faceswap.onnx_io import nchw_to_bgr

    return nchw_to_bgr(array)


def _run_vector(session, vector: np.ndarray) -> np.ndarray:
    incoming = np.asarray(vector, dtype=np.float32).reshape(-1)
    if incoming.shape[0] != 512:
        raise SwapError(t("swap.embedding_length", length=int(incoming.shape[0])))
    item = session.get_inputs()[0]
    shape = list(item.shape)
    if len(shape) <= 1:
        payload = incoming
    else:
        payload = incoming.reshape(1, 512)
    output = session.run(None, {item.name: payload.astype(np.float32)})[0]
    converted = np.asarray(output, dtype=np.float32).reshape(-1)
    if converted.shape[0] != 512:
        raise SwapError(t("swap.embedding_length", length=int(converted.shape[0])))
    norm = float(np.linalg.norm(converted)) + 1e-8
    return (converted / norm).astype(np.float32)
