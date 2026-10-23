"""Occlusion and parsing masks, including the stub fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from barrett_faceswap.config import load_config
from barrett_faceswap.face_parsing import labels_from_output, run_parser
from barrett_faceswap.masking import MaskStack
from barrett_faceswap.models import ModelStore
from barrett_faceswap.occlusion import run_occlusion
from barrett_faceswap.processors.face_swapper import FaceSwapper
from barrett_faceswap.types import Face, ProcessContext
from tests.conftest import write_config
from tests.onnx_graphs import write_constant_onnx


def _config(tmp_path: Path, **mask_updates: object):
    config = load_config(write_config(tmp_path, enabled=False))
    masks = config.masks.model_copy(update=mask_updates or {"parser": False})
    return config.model_copy(
        update={
            "masks": masks,
            "blend": config.blend.model_copy(update={"mask_blur": 0}),
            "swap": config.swap.model_copy(update={"resolution": 32}),
        }
    )


def test_parser_keeps_only_the_selected_class(tmp_path: Path) -> None:
    scores = np.full((1, 19, 8, 8), -5, dtype=np.float32)
    scores[:, 1] = 5
    path = tmp_path / "parser.onnx"
    write_constant_onnx(path, [1, 3, 8, 8], scores)
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    skin = run_parser(session, crop, ["skin"], 8, 0)
    hair = run_parser(session, crop, ["hair"], 8, 0)
    assert skin.shape == (8, 8)
    assert float(skin.min()) == 1.0
    assert float(hair.max()) == 0.0


def test_label_map_rank_is_accepted() -> None:
    labels = np.array([[1, 17], [0, 1]], dtype=np.int32)
    assert labels_from_output(labels)[0, 1] == 17


def test_occlusion_invert_and_min_merge(tmp_path: Path) -> None:
    visible = np.full((1, 1, 8, 8), 0.25, dtype=np.float32)
    path = tmp_path / "occlusion_xseg.onnx"
    write_constant_onnx(path, [1, 3, 8, 8], visible)
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    crop = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = run_occlusion(session, crop, 16, 0, invert=False)
    flipped = run_occlusion(session, crop, 16, 0, invert=True)
    assert mask.shape == (16, 16)
    assert float(mask[8, 8]) == pytest.approx(0.25, abs=0.02)
    assert float(flipped[8, 8]) == pytest.approx(0.75, abs=0.02)

    config = _config(tmp_path, parser=False, occlusion=True, occlusion_invert=False)
    # Point the occluder path at the graph we just wrote.
    config = config.model_copy(
        update={"models": config.models.model_copy(update={"occluder": config.models.occluder.model_copy(update={"path": path})})}
    )
    stack = MaskStack(config, ModelStore(config, ["cpu"]))
    composed = stack.compose(crop, 16, 0)
    assert composed.shape == (16, 16)
    assert float(composed[8, 8]) == pytest.approx(0.25, abs=0.02)
    assert float(composed[0, 0]) == 0.0


def test_missing_mask_models_fall_back_to_stubs(tmp_path: Path) -> None:
    config = _config(tmp_path, parser=True, occlusion=True)
    stack = MaskStack(config, ModelStore(config, ["cpu"]))
    crop = np.zeros((32, 32, 3), dtype=np.uint8)
    composed = stack.compose(crop, 32, 0)
    assert composed.shape == (32, 32)
    assert float(composed[16, 16]) > 0.0
    assert float(composed[0, 0]) == 0.0


def test_swapper_uses_an_occlusion_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "w600k_r50.onnx").write_bytes(b"recognizer")
    (weights / "inswapper_128.onnx").write_bytes(b"swapper")
    write_constant_onnx(weights / "det_10g.onnx", [1, 3, 8, 8], np.zeros((1, 3, 8, 8), np.float32))
    zeros = np.zeros((1, 1, 8, 8), dtype=np.float32)
    write_constant_onnx(weights / "occlusion_xseg.onnx", [1, 3, 8, 8], zeros)
    config = _config(tmp_path, parser=False, occlusion=True)
    occluder = config.models.occluder.model_copy(update={"path": weights / "occlusion_xseg.onnx"})
    config = config.model_copy(update={"models": config.models.model_copy(update={"occluder": occluder})})

    class _Detector:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def detect(self, image: np.ndarray) -> list[Face]:
            height, width = image.shape[:2]
            landmarks = np.array(
                [
                    [width * 0.3, height * 0.35],
                    [width * 0.7, height * 0.35],
                    [width * 0.5, height * 0.55],
                    [width * 0.35, height * 0.75],
                    [width * 0.65, height * 0.75],
                ],
                dtype=np.float32,
            )
            return [Face(bbox=np.array([0, 0, width, height], np.float32), score=0.9, landmarks=landmarks)]

    class _Swapper:
        def __init__(self, store, resolution: int) -> None:
            del store
            self.resolution = resolution

        def embed(self, source_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
            del source_bgr, landmarks
            return np.ones(512, dtype=np.float32)

        def forward(self, target_crop_bgr: np.ndarray, embedding: np.ndarray) -> np.ndarray:
            del embedding
            painted = np.zeros_like(target_crop_bgr)
            painted[:, :] = (0, 255, 0)
            return painted

    monkeypatch.setattr("barrett_faceswap.processors.face_swapper.FaceDetector", _Detector)
    monkeypatch.setattr("barrett_faceswap.processors.face_swapper.IdentitySwapper", _Swapper)
    swapper = FaceSwapper(config, ["cpu"])
    frame = np.full((64, 64, 3), (255, 0, 0), dtype=np.uint8)
    context = ProcessContext(source_bgr=frame.copy(), dry_run=False)
    swapper.pre_check(context)
    output = swapper.process_frame(frame, context)
    assert np.array_equal(output, frame)
