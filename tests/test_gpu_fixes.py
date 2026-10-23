"""emap fallback, score log, hull mask, CUDA preload, and offline downloads."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from barrett_faceswap.align import swap_template
from barrett_faceswap.cli import main
from barrett_faceswap.config import load_config
from barrett_faceswap.errors import SwapError
from barrett_faceswap.execution import prepare_runtime, reset_runtime_prepare
from barrett_faceswap.fetch import download_models
from barrett_faceswap.masks import landmark_hull_mask, region_mask
from barrett_faceswap.processors.face_swapper import FaceSwapper, _Slot
from barrett_faceswap.swap import read_emap
from barrett_faceswap.types import Face, ProcessContext
from tests.conftest import REPO, write_config


def _write_initializers(path: Path, arrays: list[tuple[str, np.ndarray]]) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    initializers = [
        numpy_helper.from_array(np.asarray(values, dtype=np.float32), name) for name, values in arrays
    ]
    entered = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    left = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "emap-fixture", [entered], [left], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def test_emap_name_wins_over_the_last_initializer(tmp_path: Path) -> None:
    path = tmp_path / "named.onnx"
    named = np.full((512, 512), 3.0, dtype=np.float32)
    last = np.full((512, 512), 9.0, dtype=np.float32)
    _write_initializers(path, [("buff2fs", last), ("emap", named)])
    matrix = read_emap(path)
    assert matrix.shape == (512, 512)
    assert float(matrix[0, 0]) == pytest.approx(3.0)


def test_last_initializer_is_buff2fs_when_emap_is_absent(tmp_path: Path) -> None:
    path = tmp_path / "buff2fs.onnx"
    last = np.full((512, 512), 4.0, dtype=np.float32)
    _write_initializers(path, [("other", np.zeros((4,), dtype=np.float32)), ("buff2fs", last)])
    matrix = read_emap(path)
    assert matrix.shape == (512, 512)
    assert float(matrix[0, 0]) == pytest.approx(4.0)


def test_wrong_size_initializer_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "short.onnx"
    _write_initializers(path, [("buff2fs", np.zeros((8,), dtype=np.float32))])
    with pytest.raises(SwapError):
        read_emap(path)


def test_named_emap_with_the_wrong_size_is_not_replaced(tmp_path: Path) -> None:
    path = tmp_path / "bad-name.onnx"
    good = np.full((512, 512), 1.0, dtype=np.float32)
    _write_initializers(path, [("emap", np.zeros((4,), dtype=np.float32)), ("buff2fs", good)])
    with pytest.raises(SwapError):
        read_emap(path)


def test_source_and_target_scores_are_logged_separately() -> None:
    config = load_config(REPO / "configs" / "default.yaml")
    swapper = FaceSwapper(config, ["cpu"])
    swapper.store = object()
    swapper._slots = [_Slot(latent=np.zeros(512, dtype=np.float32), match=np.zeros(512), source_score=0.111)]
    face = Face(
        bbox=np.array([0, 0, 10, 10], dtype=np.float32),
        score=0.722,
        landmarks=np.zeros((5, 2), dtype=np.float32),
    )
    ctx = ProcessContext(
        source_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
        dry_run=False,
        known_faces=[face],
        log_every_frame=True,
    )
    swapper._swap_one = lambda frame, found, slot, context: frame  # type: ignore[method-assign]
    swapper.process_frame(np.zeros((8, 8, 3), dtype=np.uint8), ctx)
    assert ctx.notes
    assert "0.111" in ctx.notes[0]
    assert "0.722" in ctx.notes[0]
    assert ctx.notes[0].index("0.111") < ctx.notes[0].index("0.722")


def test_landmark_hull_is_feathered_and_not_the_ellipse() -> None:
    size = 128
    template = swap_template("inswapper", size)
    hull = landmark_hull_mask(size, template, blur=15)
    ellipse = region_mask(size, 15)
    nose_x, nose_y = (int(template[2, 0]), int(template[2, 1]))
    assert hull.shape == (size, size)
    assert float(hull[nose_y, nose_x]) > 0.5
    assert float(hull[0, 0]) == 0.0
    assert np.any((hull > 0.0) & (hull < 1.0))
    assert not np.allclose(hull, ellipse)


def test_default_color_is_lab_and_compliance_stays_off() -> None:
    config = load_config(REPO / "configs" / "default.yaml")
    assert config.blend.color == "lab"
    assert config.compliance.enabled is False


def test_prepare_runtime_does_not_raise_on_the_cpu_wheel() -> None:
    reset_runtime_prepare()
    status = prepare_runtime()
    assert status["preload"] in {"ok", "unavailable", "failed"}
    assert prepare_runtime()["preload"] == status["preload"]


def test_doctor_reports_providers_without_a_gpu(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["doctor", "--config", str(write_config(tmp_path, enabled=False))])
    captured = capsys.readouterr()
    assert code == 0
    assert "cpu" in captured.out
    assert "detector" in captured.out
    assert "missing" in captured.out


class _Body:
    def __init__(self, status: int, data: bytes) -> None:
        self.status = status
        self._data = data
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._data) - self._pos
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def close(self) -> None:
        return None


class _ShortThenReset(_Body):
    def __init__(self, data: bytes) -> None:
        super().__init__(200, data)
        self._failed = False

    def read(self, size: int = -1) -> bytes:
        if self._failed:
            raise OSError("connection reset")
        self._failed = True
        return super().read(size)


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in members.items():
            archive.writestr(f"buffalo_l/{name}", payload)
    return buffer.getvalue()


def test_download_falls_back_resumes_and_checks_zip_members(tmp_path: Path) -> None:
    det = b"det-bytes"
    rec = b"rec-bytes"
    swap = b"swap-bytes-0123456789"
    pins = {
        "det_10g.onnx": hashlib.sha256(det).hexdigest(),
        "w600k_r50.onnx": hashlib.sha256(rec).hexdigest(),
        "inswapper_128.onnx": hashlib.sha256(swap).hexdigest(),
    }
    archive = _zip_bytes({"det_10g.onnx": det, "w600k_r50.onnx": rec})
    ranges: list[str] = []

    def opener(url: str, headers: dict[str, str]):
        if url.endswith("buffalo_l.zip") and "/v0.7/" in url:
            raise OSError("timed out")
        if url.endswith("buffalo_l.zip"):
            return _Body(200, archive)
        if url.endswith("inswapper_128.onnx") and "insightface" in url:
            if "Range" not in headers:
                return _ShortThenReset(swap[:4])
            ranges.append(headers["Range"])
            return _Body(206, swap[4:])
        raise OSError(url)

    config = load_config(write_config(tmp_path, enabled=False))
    notes = download_models(config, opener=opener, pins=pins, attempts=3)
    text = "\n".join(notes)
    assert pins["det_10g.onnx"] in text
    assert pins["inswapper_128.onnx"] in text
    assert ranges == ["bytes=4-"]
    assert (tmp_path / "weights" / "det_10g.onnx").read_bytes() == det
    assert (tmp_path / "weights" / "inswapper_128.onnx").read_bytes() == swap
    assert config.compliance.enabled is False

    again = download_models(config, opener=opener, pins=pins, attempts=1)
    assert all("sha256=" in line for line in again)
    assert not any("Wrote" in line or "已写入" in line for line in again)
