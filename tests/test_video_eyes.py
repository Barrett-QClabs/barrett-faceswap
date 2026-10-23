"""Video face recovery, eye restore, and the download/packaging edges around them."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from barrett_faceswap.align import swap_template
from barrett_faceswap.blend import paste_back
from barrett_faceswap.config import load_config
from barrett_faceswap.doctor import installed_ort_names, run_doctor
from barrett_faceswap.errors import BarrettFaceswapError, NoFaceError
from barrett_faceswap.eyes import eye_mask, restore_eyes
from barrett_faceswap.fetch import CONNECT_TIMEOUT_S, download_models
from barrett_faceswap.processors import resolve_processor_names
from barrett_faceswap.processors.face_swapper import FaceSwapper
from barrett_faceswap.recover import FaceStream, map_points, rotate_bgr, search_faces
from barrett_faceswap.types import Face, ProcessContext, SwapRequest
from tests.conftest import write_config


def _face(origin: float) -> Face:
    return Face(
        bbox=np.array([origin, 0, origin + 10, 10], dtype=np.float32),
        score=0.9,
        landmarks=np.array(
            [[origin, 1], [origin + 4, 1], [origin + 2, 3], [origin + 1, 6], [origin + 3, 6]],
            dtype=np.float32,
        ),
    )


def test_face_stream_interpolates_inside_the_gap() -> None:
    stream = FaceStream(max_gap=10, smooth="none")
    blank = np.zeros((4, 4, 3), dtype=np.uint8)
    ready = []
    ready.extend(stream.push(0, blank, [_face(0)], "hit"))
    for index in range(1, 4):
        ready.extend(stream.push(index, blank, [], "miss"))
    ready.extend(stream.push(4, blank, [_face(40)], "hit"))
    filled = [item for item in ready if item.tag == "filled"]
    assert [item.index for item in filled] == [1, 2, 3]
    assert filled[1].faces[0].landmarks[0, 0] == pytest.approx(20.0)
    assert stream.counts["detected"] == 2
    assert stream.counts["filled"] == 3
    assert stream.counts["passthrough"] == 0


def test_face_stream_passthrough_beyond_the_gap_and_at_the_end() -> None:
    stream = FaceStream(max_gap=2, smooth="none")
    blank = np.zeros((4, 4, 3), dtype=np.uint8)
    ready = []
    ready.extend(stream.push(0, blank, [_face(0)], "hit"))
    for index in range(1, 6):
        ready.extend(stream.push(index, blank, [], "miss"))
    ready.extend(stream.finish())
    by_index = {item.index: item for item in ready}
    assert by_index[1].tag == "filled"
    assert by_index[2].tag == "filled"
    assert by_index[3].tag == "passthrough"
    assert by_index[5].tag == "passthrough"
    assert stream.counts["passthrough"] >= 2

    tight = FaceStream(max_gap=0, smooth="none")
    missed = tight.push(1, blank, [], "miss")
    assert missed[0].tag == "passthrough"
    assert missed[0].faces == []


def test_smoothing_holds_a_constant_and_moves_partway() -> None:
    blank = np.zeros((4, 4, 3), dtype=np.uint8)
    euro = FaceStream(max_gap=4, smooth="one_euro")
    first = euro.push(0, blank, [_face(8)], "hit")[0]
    second = euro.push(1, blank, [_face(8)], "hit")[0]
    assert second.faces[0].bbox[0] == pytest.approx(first.faces[0].bbox[0])

    ema = FaceStream(max_gap=4, smooth="ema")
    ema.push(0, blank, [_face(0)], "hit")
    moved = ema.push(1, blank, [_face(100)], "hit")[0]
    assert 0 < float(moved.faces[0].bbox[0]) < 100


def test_rotation_round_trip_and_retry_tag() -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    rotated, matrix = rotate_bgr(image, 20)
    point = np.array([[6, 9]], dtype=np.float32)
    inverse = cv2.invertAffineTransform(matrix)
    back = map_points(map_points(point, matrix), inverse)
    assert np.allclose(back, point, atol=1.0)

    class _Hit:
        score_threshold = 0.5

        def try_detect(self, image_bgr, threshold=None):
            del image_bgr, threshold
            return [_face(4)]

    faces, tag = search_faces(_Hit(), image)
    assert tag == "hit"
    assert faces

    class _Later:
        score_threshold = 0.5

        def __init__(self) -> None:
            self.calls = 0

        def try_detect(self, image_bgr, threshold=None):
            self.calls += 1
            if self.calls == 1:
                return []
            if image_bgr.shape[0] > 32:
                height, width = image_bgr.shape[:2]
                center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)
                landmarks = np.repeat(center.reshape(1, 2), 5, axis=0)
                return [
                    Face(
                        bbox=np.array([center[0] - 2, center[1] - 2, center[0] + 2, center[1] + 2], np.float32),
                        score=0.4,
                        landmarks=landmarks,
                    )
                ]
            return []

    found, retry = search_faces(_Later(), np.zeros((32, 32, 3), dtype=np.uint8))
    assert retry == "retry"
    assert found
    assert float(found[0].landmarks[0, 0]) == pytest.approx(15.5, abs=2.0)


def test_eye_mask_and_restore() -> None:
    template = swap_template("inswapper", 128)
    mask = eye_mask(128, template, blur=1)
    eye = template[0]
    assert mask[int(round(float(eye[1]))), int(round(float(eye[0])))] > 0.5
    assert mask[0, 0] == 0
    soft = eye_mask(128, template, blur=9)
    assert np.any((soft > 0.05) & (soft < 0.95))

    swapped = np.zeros((128, 128, 3), dtype=np.uint8)
    target = np.full((128, 128, 3), 255, dtype=np.uint8)
    kept = restore_eyes("target", swapped, target, template, blur=1, sharpen=0)
    assert int(kept[int(round(float(eye[1]))), int(round(float(eye[0]))), 0]) == 255
    assert int(kept[0, 0, 0]) == 0
    untouched = restore_eyes("swap", swapped, target, template, blur=1, sharpen=0)
    assert int(untouched[int(round(float(eye[1]))), int(round(float(eye[0]))), 0]) == 0

    points = np.zeros((106, 2), dtype=np.float32)
    for index in range(25):
        points[index] = (40 + (index % 5) - 2, 30 + (index // 5) - 2)
    for index in range(28):
        points[25 + index] = (90 + (index % 5) - 2, 32 + (index // 5) - 2)
    points[53:, 0] = 64
    points[53:, 1] = 100
    dense = eye_mask(128, points, blur=1)
    assert dense[30, 40] > 0.5
    assert dense[32, 90] > 0.5
    assert dense[120, 4] == 0


def test_pixel_boost_uses_four_native_tiles(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, enabled=False))
    config = config.model_copy(
        update={
            "swap": config.swap.model_copy(update={"pixel_boost": 256}),
            "blend": config.blend.model_copy(update={"color": "none", "eye_mode": "swap", "eye_sharpen": 0}),
        }
    )
    swapper = FaceSwapper(config, ["cpu"])

    class _Tiles:
        def __init__(self) -> None:
            self.shapes: list[tuple[int, ...]] = []

        def forward(self, tile, latent):
            del latent
            self.shapes.append(tuple(int(dim) for dim in tile.shape))
            return tile

    backend = _Tiles()
    swapper.store = object()
    swapper.masks = SimpleNamespace(compose=lambda crop, side, blur, landmarks=None: np.ones((side, side), np.float32))
    swapper._backend = lambda: backend  # type: ignore[method-assign]
    swapper._slots = [SimpleNamespace(latent=np.zeros((1, 512), np.float32))]
    template = swap_template("inswapper", 256)
    face = Face(
        bbox=np.array([10, 10, 300, 300], dtype=np.float32),
        score=0.99,
        landmarks=template + np.array([20, 20], dtype=np.float32),
    )
    frame = np.zeros((360, 360, 3), dtype=np.uint8)
    swapper.process_frame(
        frame,
        ProcessContext(source_bgr=frame, dry_run=False, known_faces=[face], log_every_frame=False),
    )
    assert backend.shapes == [(128, 128, 3)] * 4


def test_still_image_raises_and_video_can_pass_a_miss(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, enabled=False))
    swapper = FaceSwapper(config, ["cpu"])
    swapper.store = object()
    swapper._detect = lambda image: []  # type: ignore[method-assign]
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(NoFaceError):
        swapper.process_frame(frame, ProcessContext(source_bgr=frame, dry_run=False))
    passed = swapper.process_frame(
        frame,
        ProcessContext(source_bgr=frame, dry_run=False, known_faces=[], allow_missing_face=True),
    )
    assert np.array_equal(passed, frame)


def test_paste_back_matches_a_full_frame_composite() -> None:
    base = np.zeros((48, 64, 3), dtype=np.uint8)
    base[:, :] = (12, 40, 90)
    crop = np.zeros((20, 20, 3), dtype=np.uint8)
    crop[:, :] = (200, 30, 10)
    mask = np.zeros((20, 20), dtype=np.float32)
    cv2.circle(mask, (10, 10), 7, 1.0, thickness=-1)
    matrix = np.array([[1.1, 0.0, 18], [0.0, 1.1, 12]], dtype=np.float32)
    inverse = cv2.invertAffineTransform(matrix)
    warped = cv2.warpAffine(crop, inverse, (64, 48), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpAffine(mask, inverse, (64, 48), flags=cv2.INTER_LINEAR)
    alpha = warped_mask[..., None]
    reference = np.clip(warped.astype(np.float32) * alpha + base.astype(np.float32) * (1.0 - alpha), 0, 255).astype(
        np.uint8
    )
    pasted = paste_back(base, crop, matrix, mask)
    assert int(np.max(np.abs(pasted.astype(np.int16) - reference.astype(np.int16)))) <= 1
    empty = paste_back(base, crop, matrix, np.zeros_like(mask))
    assert np.array_equal(empty, base)


def test_with_enhancer_appends_the_processor(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, enabled=False))
    request = SwapRequest(
        source_path=tmp_path / "a.png",
        target_path=tmp_path / "b.png",
        output_path=tmp_path / "c.png",
        with_enhancer=True,
    )
    assert resolve_processor_names(config, request) == ["face_swapper", "face_enhancer"]


def test_doctor_warns_when_both_wheels_are_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(write_config(tmp_path, enabled=False))
    text = "\n".join(run_doctor(config))
    if set(installed_ort_names()) != {"onnxruntime", "onnxruntime-gpu"}:
        assert "hide the GPU" not in text
    monkeypatch.setattr(
        "barrett_faceswap.doctor.installed_ort_names",
        lambda: ["onnxruntime", "onnxruntime-gpu"],
    )
    warned = "\n".join(run_doctor(config))
    assert "hide the GPU" in warned
    assert config.compliance.enabled is False


def test_download_imports_a_directory_and_still_fails_over(tmp_path: Path) -> None:
    assert CONNECT_TIMEOUT_S == 15
    det = b"det-bytes"
    rec = b"rec-bytes"
    swap = b"swap-bytes"
    pins = {
        "det_10g.onnx": hashlib.sha256(det).hexdigest(),
        "w600k_r50.onnx": hashlib.sha256(rec).hexdigest(),
        "inswapper_128.onnx": hashlib.sha256(swap).hexdigest(),
    }
    folder = tmp_path / "drop"
    folder.mkdir()
    (folder / "det_10g.onnx").write_bytes(det)
    (folder / "w600k_r50.onnx").write_bytes(rec)
    (folder / "inswapper_128.onnx").write_bytes(swap)
    config = load_config(write_config(tmp_path, enabled=False))

    def opener(url: str, headers: dict[str, str]):
        del url, headers
        raise AssertionError("network")

    notes = download_models(config, opener=opener, pins=pins, from_dir=folder)
    text = "\n".join(notes)
    assert pins["inswapper_128.onnx"] in text
    assert (tmp_path / "weights" / "inswapper_128.onnx").read_bytes() == swap

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "det_10g.onnx").write_bytes(b"nope")
    with pytest.raises(BarrettFaceswapError):
        download_models(config, opener=opener, pins=pins, from_dir=bad, force=True)

    called: list[str] = []

    def missing(url: str, headers: dict[str, str]):
        del headers
        called.append(url)
        raise OSError("stop")

    (tmp_path / "weights" / "inswapper_128.onnx").unlink()
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "nested").mkdir()
    (partial / "nested" / "det_10g.onnx").write_bytes(det)
    (partial / "nested" / "w600k_r50.onnx").write_bytes(rec)
    with pytest.raises(BarrettFaceswapError):
        download_models(config, opener=missing, pins=pins, from_dir=partial, attempts=1)
    assert any(url.endswith("inswapper_128.onnx") for url in called)
