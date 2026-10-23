"""Benchmark command without weights or a GPU."""

from __future__ import annotations

from pathlib import Path

import pytest

from barrett_faceswap.benchmark import run_benchmark, synthetic_face
from barrett_faceswap.cli import main
from barrett_faceswap.config import load_config
from barrett_faceswap.errors import ConfigError, NoFaceError
from barrett_faceswap.processors.face_swapper import FaceSwapper
from barrett_faceswap.types import Face
from tests.conftest import write_config, write_png


def test_dry_run_benchmark_reports_fps(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, enabled=False))
    report = run_benchmark(config, frames=4, width=32, height=24, dry_run=True, providers=["cpu"])
    assert report.frames == 4
    assert report.fps > 0
    assert report.dry_run is True
    assert report.vram_mb is None or report.vram_mb >= 0
    text = report.line()
    assert "fps=" in text
    assert "vram_mb=" in text


def test_benchmark_cli(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "benchmark",
            "--config",
            str(write_config(tmp_path, enabled=False)),
            "--dry-run",
            "--frames",
            "3",
            "--width",
            "32",
            "--height",
            "32",
            "--execution-provider",
            "cpu",
        ]
    )
    assert code == 0
    assert "fps=" in capsys.readouterr().out


def test_synthetic_face_keeps_landmarks_on_the_drawing() -> None:
    image, face = synthetic_face(64, 48, (40, 80, 160))
    assert image.shape == (48, 64, 3)
    for x, y in face.landmarks:
        assert 0 <= float(x) < 64
        assert 0 <= float(y) < 48
    eye = image[int(face.landmarks[0][1]), int(face.landmarks[0][0])]
    nose = image[int(face.landmarks[2][1]), int(face.landmarks[2][0])]
    assert int(eye.sum()) < int(nose.sum())


def test_benchmark_source_and_target_must_be_paired(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, enabled=False))
    source = tmp_path / "source.png"
    write_png(source, (20, 40, 60))
    with pytest.raises(ConfigError):
        run_benchmark(
            config,
            frames=1,
            width=32,
            height=32,
            dry_run=True,
            source_path=source,
        )


def test_source_hint_skips_detection(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, enabled=False))
    swapper = FaceSwapper(config, ["cpu"])
    called: list[int] = []

    def _detect(image):
        del image
        called.append(1)
        raise NoFaceError("detector should not run")

    swapper._detect = _detect  # type: ignore[method-assign]

    class _Backend:
        def embed(self, image, landmarks):
            del image, landmarks
            import numpy as np

            return np.ones(512, dtype=np.float32)

    _image, face = synthetic_face(32, 32, (30, 60, 90))
    vector, score = swapper._embed_image(_Backend(), _image, face)
    assert called == []
    assert score == face.score
    assert vector.shape == (512,)


def test_known_face_type_is_a_face() -> None:
    _image, face = synthetic_face(32, 32, (1, 2, 3))
    assert isinstance(face, Face)
