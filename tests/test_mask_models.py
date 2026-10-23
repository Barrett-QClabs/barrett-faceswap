"""Occlusion and parsing masks, including the stub fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from barrett_faceswap.config import load_config
from barrett_faceswap.face_parsing import eye_mask_from_labels, labels_from_output, run_parser
from barrett_faceswap.masking import MaskStack, _warp_crop_mask
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


def test_occlusion_accepts_nchw_and_nhwc(tmp_path: Path) -> None:
    import onnxruntime

    crop = np.zeros((16, 16, 3), dtype=np.uint8)
    nchw = tmp_path / "nchw.onnx"
    write_constant_onnx(nchw, [1, 3, 8, 8], np.full((1, 1, 8, 8), 0.4, np.float32))
    nchw_session = onnxruntime.InferenceSession(str(nchw), providers=["CPUExecutionProvider"])
    channels_first = run_occlusion(nchw_session, crop, 16, 0, invert=False)
    nhwc = tmp_path / "nhwc.onnx"
    write_constant_onnx(nhwc, [1, 8, 8, 3], np.full((1, 8, 8, 1), 0.7, np.float32))
    nhwc_session = onnxruntime.InferenceSession(str(nhwc), providers=["CPUExecutionProvider"])
    channels_last = run_occlusion(nhwc_session, crop, 16, 0, invert=False)
    assert channels_first.shape == channels_last.shape == (16, 16)
    assert float(channels_first[8, 8]) == pytest.approx(0.4, abs=0.02)
    assert float(channels_last[8, 8]) == pytest.approx(0.7, abs=0.02)


def test_parser_eye_classes_replace_the_ellipse() -> None:
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[4:8, 4:8] = 4
    labels[4:8, 10:14] = 5
    mask = eye_mask_from_labels(labels, 0)
    assert mask is not None
    assert float(mask[5, 5]) == 1.0
    assert float(mask[5, 11]) == 1.0
    assert float(mask[0, 0]) == 0.0
    assert eye_mask_from_labels(np.zeros((16, 16), dtype=np.int32), 0) is None


def test_video_mask_reuses_until_the_face_moves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "occlusion_xseg.onnx"
    write_constant_onnx(path, [1, 3, 8, 8], np.full((1, 1, 8, 8), 0.5, np.float32))
    config = _config(tmp_path, parser=False, occlusion=True, stride=3, stride_motion=0.04)
    config = config.model_copy(
        update={
            "models": config.models.model_copy(
                update={"occluder": config.models.occluder.model_copy(update={"path": path})}
            )
        }
    )
    calls: list[int] = []
    real = run_occlusion

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr("barrett_faceswap.masking.run_occlusion", counted)
    stack = MaskStack(config, ModelStore(config, ["cpu"]))
    crop = np.zeros((16, 16, 3), dtype=np.uint8)
    points = np.array([[8, 8], [24, 8], [16, 16], [10, 24], [22, 24]], dtype=np.float32)
    stack.compose(crop, 16, 0, frame_points=points, sequential=True, face_width=40)
    stack.compose(crop, 16, 0, frame_points=points + 0.2, sequential=True, face_width=40)
    assert calls == [1]
    moved = points + np.array([20, 0], dtype=np.float32)
    stack.compose(crop, 16, 0, frame_points=moved, sequential=True, face_width=40)
    assert calls == [1, 1]
    stack.compose(crop, 16, 0, frame_points=points, sequential=False, face_width=40)
    assert calls == [1, 1, 1]


def test_alignment_warp_moves_a_reused_mask() -> None:
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[:, :2] = 1.0
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    same = _warp_crop_mask(mask, identity, identity, 8)
    assert np.allclose(same[:, :2], 1.0)
    shifted = np.array([[1, 0, 2], [0, 1, 0]], dtype=np.float32)
    moved = _warp_crop_mask(mask, identity, shifted, 8)
    assert float(moved[:, 0].max()) < float(moved[:, 2].max())


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
