"""GFPGAN / CodeFormer runner without real weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from barrett_faceswap.config import load_config
from barrett_faceswap.enhance import blend_images, run_enhancer
from barrett_faceswap.errors import ModelNotFoundError
from barrett_faceswap.processors.face_enhancer import FaceEnhancer
from barrett_faceswap.types import Face, ProcessContext
from tests.conftest import write_config
from tests.onnx_graphs import write_constant_onnx, write_identity_with_weight


def test_blend_respects_the_amount() -> None:
    original = np.full((4, 4, 3), 100, dtype=np.uint8)
    restored = np.zeros((4, 4, 3), dtype=np.uint8)
    assert int(blend_images(restored, original, 0).mean()) == 100
    assert int(blend_images(restored, original, 1).mean()) == 0


def test_codeformer_weight_input_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "codeformer.onnx"
    write_identity_with_weight(path, 16)
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    crop = np.full((16, 16, 3), (10, 40, 220), dtype=np.uint8)
    restored = run_enhancer(session, crop, weight=0.35)
    assert restored.shape == crop.shape
    assert np.max(np.abs(restored.astype(np.int16) - crop.astype(np.int16))) <= 1


def test_enhancer_pastes_a_synthetic_restoration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    write_constant_onnx(weights / "det_10g.onnx", [1, 3, 8, 8], np.zeros((1, 3, 8, 8), np.float32))
    write_constant_onnx(weights / "gfpgan_1.4.onnx", [1, 3, 32, 32], np.zeros((1, 3, 32, 32), np.float32))
    config = load_config(write_config(tmp_path, enabled=False))
    gfpgan = config.models.gfpgan.model_copy(update={"path": weights / "gfpgan_1.4.onnx"})
    config = config.model_copy(
        update={
            "models": config.models.model_copy(update={"gfpgan": gfpgan}),
            "blend": config.blend.model_copy(update={"mask_blur": 0}),
            "enhancer": config.enhancer.model_copy(update={"blend": 1.0, "model": "gfpgan_1.4"}),
        }
    )

    class _Detector:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def detect(self, image: np.ndarray) -> list[Face]:
            height, width = image.shape[:2]
            landmarks = np.array(
                [
                    [width * 0.35, height * 0.38],
                    [width * 0.65, height * 0.38],
                    [width * 0.50, height * 0.55],
                    [width * 0.40, height * 0.70],
                    [width * 0.60, height * 0.70],
                ],
                dtype=np.float32,
            )
            return [Face(bbox=np.array([0, 0, width, height], np.float32), score=0.99, landmarks=landmarks)]

    monkeypatch.setattr("barrett_faceswap.processors.face_enhancer.FaceDetector", _Detector)
    enhancer = FaceEnhancer(config, ["cpu"])
    frame = np.full((96, 96, 3), 255, dtype=np.uint8)
    context = ProcessContext(source_bgr=frame, dry_run=False)
    enhancer.pre_check(context)
    restored = enhancer.process_frame(frame, context)
    assert restored.shape == frame.shape
    assert not np.array_equal(restored, frame)
    assert any("gfpgan_1.4" in note for note in context.notes)


def test_missing_enhancer_weight_is_reported(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    write_constant_onnx(weights / "det_10g.onnx", [1, 3, 8, 8], np.zeros((1, 3, 8, 8), np.float32))
    config = load_config(write_config(tmp_path, enabled=False))
    enhancer = FaceEnhancer(config, ["cpu"])
    with pytest.raises(ModelNotFoundError) as caught:
        enhancer.pre_check(ProcessContext(source_bgr=np.zeros((8, 8, 3), np.uint8), dry_run=False))
    assert "gfpgan_1.4.onnx" in str(caught.value)


def test_enhancer_dry_run_does_not_require_weights(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, enabled=False))
    enhancer = FaceEnhancer(config, ["cpu"])
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    context = ProcessContext(source_bgr=frame, dry_run=True)
    enhancer.pre_check(context)
    assert np.array_equal(enhancer.process_frame(frame, context), frame)
    assert any("gfpgan_1.4" in note for note in context.notes)
