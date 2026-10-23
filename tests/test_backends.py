"""CPU checks for selectable backends. No weights are downloaded."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime
import pytest

from barrett_faceswap.align import ARCFACE_112_V1, swap_template, template_for_size
from barrett_faceswap.cli import main
from barrett_faceswap.config import ModelFile, TensorRTConfig, load_config
from barrett_faceswap.enhance import blend_region
from barrett_faceswap.execution import prefer_cuda, tensorrt_options
from barrett_faceswap.face_parsing import enhancer_skin_mask
from barrett_faceswap.fetch import KNOWN_SHA256
from barrett_faceswap.models import ModelStore, clear_session_cache
from barrett_faceswap.occlusion import refine_occlusion
from barrett_faceswap.preprocess import denormalize_nchw, normalize_crop
from barrett_faceswap.recover import LandmarkSmoother
from barrett_faceswap.swap import GraphSwapper
from barrett_faceswap.swap_catalog import SWAPPERS
from barrett_faceswap.types import Face
from tests.conftest import write_config, write_png
from tests.onnx_graphs import write_vector_identity


def _config(tmp_path: Path):
    return load_config(write_config(tmp_path, enabled=False))


def _write_named_identity(path: Path, source_size: int, target_size: int) -> None:
    import onnx
    from onnx import TensorProto, helper

    source = helper.make_tensor_value_info("source", TensorProto.FLOAT, [1, 3, source_size, source_size])
    target = helper.make_tensor_value_info("target", TensorProto.FLOAT, [1, 3, target_size, target_size])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, target_size, target_size])
    kept = helper.make_tensor_value_info("kept", TensorProto.FLOAT, [1, 3, source_size, source_size])
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["target"], ["output"]),
            helper.make_node("Identity", ["source"], ["kept"]),
        ],
        "blend",
        [source, target],
        [output, kept],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def _write_refine(path: Path, size: int) -> None:
    import onnx
    from onnx import TensorProto, helper

    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, size, size])
    mask = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, size, size])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, size, size])
    kept = helper.make_tensor_value_info("kept", TensorProto.FLOAT, [1, 3, size, size])
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["mask"], ["output"]),
            helper.make_node("Identity", ["image"], ["kept"]),
        ],
        "refine",
        [image, mask],
        [output, kept],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def test_normalize_matches_each_swapper_contract() -> None:
    crop = np.zeros((4, 4, 3), dtype=np.uint8)
    crop[:, :] = (10, 20, 30)
    rgb = np.array([30, 20, 10], dtype=np.float32) / 255.0
    for name, spec in SWAPPERS.items():
        blob = normalize_crop(crop, spec)
        mean = np.asarray(spec.mean, dtype=np.float32)
        std = np.asarray(spec.std, dtype=np.float32)
        expected = (rgb - mean) / std
        assert blob.shape == (1, 3, 4, 4), name
        assert blob[0, :, 0, 0] == pytest.approx(expected, abs=1e-5), name
        assert denormalize_nchw(blob, spec)[0, :, 0, 0] == pytest.approx(rgb, abs=1e-5)
    assert SWAPPERS["simswap_256"].normalize == "imagenet"
    assert SWAPPERS["hyperswap_1a_256"].normalize == "minus_one_one"
    assert SWAPPERS["simswap_unofficial_512"].normalize == "zero_one"
    assert SWAPPERS["blendswap_256"].embedding == "source_image"
    assert all(spec.commercial_use is False for spec in SWAPPERS.values())


def test_arcface_v1_template_is_not_the_shifted_inswapper_layout() -> None:
    v1 = swap_template("arcface_v1", 256)
    shifted = swap_template("arcface_128", 256)
    assert np.allclose(v1, ARCFACE_112_V1 * 256)
    assert np.allclose(shifted, template_for_size(256))
    assert not np.allclose(v1, shifted)


def test_ghost_keeps_the_raw_converter_vector(tmp_path: Path) -> None:
    clear_session_cache()
    cross = tmp_path / "crossface_ghost.onnx"
    write_vector_identity(cross)
    config = _config(tmp_path)
    models = config.models.model_copy(update={"crossface_ghost": ModelFile(path=cross)})
    config = config.model_copy(update={"models": models})
    backend = GraphSwapper(ModelStore(config, ["cpu"]), SWAPPERS["ghost_1_256"])
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 4.0
    projected = backend.project(vector)
    assert projected[0] == pytest.approx(4.0)
    assert float(np.linalg.norm(projected)) == pytest.approx(4.0)


def test_blendswap_binds_named_source_and_target(tmp_path: Path) -> None:
    clear_session_cache()
    graph = tmp_path / "blendswap_256.onnx"
    _write_named_identity(graph, 4, 8)
    config = _config(tmp_path)
    models = config.models.model_copy(update={"blendswap_256": ModelFile(path=graph)})
    config = config.model_copy(update={"models": models})
    backend = GraphSwapper(ModelStore(config, ["cpu"]), SWAPPERS["blendswap_256"])
    crop = np.full((8, 8, 3), (40, 90, 160), dtype=np.uint8)
    source = np.zeros((1, 3, 4, 4), dtype=np.float32)
    restored = backend.forward(crop, source)
    assert restored.shape == crop.shape
    assert int(np.abs(restored.astype(np.int16) - crop.astype(np.int16)).max()) <= 2


def test_skin_mask_keeps_eyes_and_brows_on_the_plain_swap() -> None:
    labels = np.ones((8, 8), dtype=np.int32)
    labels[0, 0] = 4
    labels[0, 1] = 2
    mask = enhancer_skin_mask(labels, 0)
    assert mask[0, 0] == 0.0
    assert mask[0, 1] == 0.0
    assert mask[2, 2] == 1.0
    plain = np.zeros((8, 8, 3), dtype=np.uint8)
    plain[:, :] = (1, 2, 3)
    enhanced = np.zeros_like(plain)
    enhanced[:, :] = (200, 210, 220)
    mixed = blend_region(enhanced, plain, 1.0, mask)
    assert np.array_equal(mixed[0, 0], plain[0, 0])
    assert np.array_equal(mixed[0, 1], plain[0, 1])
    assert np.array_equal(mixed[2, 2], enhanced[2, 2])


def test_one_euro_second_sample_sits_between_the_inputs() -> None:
    smoother = LandmarkSmoother("one_euro")
    start = Face(
        bbox=np.array([0, 0, 10, 10], dtype=np.float32),
        score=1.0,
        landmarks=np.zeros((5, 2), dtype=np.float32),
    )
    end = Face(
        bbox=np.array([1, 1, 12, 12], dtype=np.float32),
        score=1.0,
        landmarks=np.ones((5, 2), dtype=np.float32),
    )
    first = smoother.apply([start])[0]
    assert np.allclose(first.landmarks, 0.0)
    assert np.allclose(first.bbox, start.bbox)
    second = smoother.apply([end])[0]
    assert np.all(second.landmarks > 0.0)
    assert np.all(second.landmarks < 1.0)
    untouched = LandmarkSmoother("none").apply([end])[0]
    assert np.allclose(untouched.landmarks, end.landmarks)


def test_compare_dry_run_writes_sheet_and_timing(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    output = tmp_path / "grid.png"
    write_png(source, (20, 40, 60), (32, 32))
    write_png(target, (80, 90, 100), (32, 32))
    config = write_config(tmp_path, enabled=False)
    code = main(
        [
            "compare",
            "--source",
            str(source),
            "--target",
            str(target),
            "--output",
            str(output),
            "--config",
            str(config),
            "--dry-run",
            "--face-swapper-model",
            "inswapper_128",
            "--face-swapper-model",
            "blendswap_256",
            "--face-enhancer-model",
            "gfpgan_1.4",
            "--face-enhancer-model",
            "codeformer",
        ]
    )
    assert code == 0
    assert output.is_file()
    table = output.with_suffix(".timing.txt").read_text(encoding="utf-8")
    assert "blendswap_256" in table
    assert "codeformer" in table
    assert "dry-run" in table


def test_refine_occlusion_returns_the_coarse_mask(tmp_path: Path) -> None:
    path = tmp_path / "sam2_refine.onnx"
    _write_refine(path, 8)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    coarse = np.zeros((8, 8), dtype=np.float32)
    coarse[2:5, 3:6] = 1.0
    refined = refine_occlusion(session, crop, coarse, 8)
    assert refined.shape == coarse.shape
    assert np.allclose(refined, coarse)


def test_tensorrt_role_prefix_and_cuda_order(tmp_path: Path) -> None:
    settings = TensorRTConfig(
        engine_cache=tmp_path / "engines",
        timing_cache=tmp_path / "timing",
        fp16=True,
        builder_optimization_level=3,
    )
    occluder = tensorrt_options(0, settings, role="occluder")
    swapper = tensorrt_options(0, settings, role="swapper")
    plain = tensorrt_options(0, settings)
    assert occluder["trt_fp16_enable"] is True
    assert occluder["trt_engine_cache_prefix"] == "occluder"
    assert swapper["trt_engine_cache_prefix"] == "swapper"
    assert "trt_engine_cache_prefix" not in plain
    assert prefer_cuda(["cpu", "tensorrt", "cuda"]) == ["cuda", "tensorrt", "cpu"]


def test_download_pins_do_not_fetch() -> None:
    assert KNOWN_SHA256["blendswap_256.onnx"] == (
        "20ac5e3c002288e7abd4ea6b1af74103892a6628382d1fa905d13c0270c2667b"
    )
    assert KNOWN_SHA256["codeformer.onnx"].startswith("21710e7a")
    assert KNOWN_SHA256["xseg_2.onnx"].startswith("cd9a0879")
    assert KNOWN_SHA256["gpen_bfr_512.onnx"].startswith("d5f066b9")
    assert KNOWN_SHA256["inswapper_128.onnx"].startswith("e4a3f08c")
