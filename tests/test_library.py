"""Saved source profiles: fusion, voice cap, disk reload, and the library CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from barrett_faceswap.cli import main
from barrett_faceswap.errors import BarrettFaceswapError
from barrett_faceswap.library import FaceLibrary, SampleParts
from barrett_faceswap.sourceprep import _read_wav
from barrett_faceswap.vision import read_image
from tests.conftest import write_config


def _unit(index: int, tilt: float = 0.0) -> np.ndarray:
    vector = np.zeros(8, dtype=np.float32)
    vector[0] = 1.0
    if tilt:
        vector[index] = tilt
    return vector


def _still(blue: int) -> np.ndarray:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :] = (blue, 0, 0)
    return image


def _photo(embedding: np.ndarray, quality: float, blue: int, filename: str) -> SampleParts:
    return SampleParts(
        kind="photo",
        embedding=embedding,
        quality=quality,
        still=_still(blue),
        speech=None,
        filename=filename,
    )


def test_outlier_does_not_move_the_identity(tmp_path: Path) -> None:
    library = FaceLibrary(tmp_path / "library")
    library.create("Ada")
    library.add_sample("Ada", _photo(np.array([1, 0, 0, 0], dtype=np.float32), 0.99, 10, "a.png"))
    library.add_sample("Ada", _photo(np.array([0, 1, 0, 0], dtype=np.float32), 0.1, 200, "b.png"))
    resolved = library.resolve("Ada")
    assert resolved.fusion == "single"
    assert resolved.embedding[0] == pytest.approx(1.0)
    assert abs(float(resolved.embedding[1])) < 1e-6
    assert int(read_image(resolved.still)[0, 0, 0]) == 10


def test_similar_faces_fuse_to_a_mean(tmp_path: Path) -> None:
    library = FaceLibrary(tmp_path / "library")
    library.create("Ada")
    vectors = [
        _unit(1, 0.4),
        _unit(1, -0.4),
        _unit(2, 0.4),
        _unit(0),
        _unit(3, 0.05),
    ]
    qualities = [0.9, 0.8, 0.7, 0.6, 0.5]
    for index, (vector, quality) in enumerate(zip(vectors, qualities, strict=True)):
        library.add_sample("Ada", _photo(vector, quality, 20 + index, f"{index}.png"))
    resolved = library.resolve("Ada")
    assert resolved.fusion == "mean"
    best = vectors[0] / np.linalg.norm(vectors[0])
    assert not np.allclose(resolved.embedding, best, atol=1e-3)
    reloaded = FaceLibrary(tmp_path / "library").resolve("ada")
    assert np.allclose(reloaded.embedding, resolved.embedding)


def test_voice_keeps_the_newest_tail_and_drops_when_the_clip_is_removed(tmp_path: Path) -> None:
    library = FaceLibrary(tmp_path / "library", voice_seconds=0.5)
    library.create("Ada")
    first = np.full(10000, 0.5, dtype=np.float32)
    second = np.full(2000, 0.25, dtype=np.float32)
    library.add_sample(
        "Ada",
        SampleParts(kind="audio", embedding=None, quality=0.0, still=None, speech=first, filename="a.wav"),
    )
    added = library.add_sample(
        "Ada",
        SampleParts(kind="audio", embedding=None, quality=0.0, still=None, speech=second, filename="b.wav"),
    )
    assert added.has_voice is True
    assert added.has_face is False
    with pytest.raises(BarrettFaceswapError, match="no face"):
        library.resolve("Ada")
    audio, rate = _read_wav(tmp_path / "library" / "ada" / "voice.wav")
    assert rate == 16000
    assert audio.size == 8000
    assert audio[0] == pytest.approx(0.5, abs=0.002)
    assert audio[5999] == pytest.approx(0.5, abs=0.002)
    assert audio[6000] == pytest.approx(0.25, abs=0.002)
    choices = library.sample_choices("Ada")
    assert len(choices) == 2
    summary = library.remove_sample("Ada", choices[0][1])
    summary = library.remove_sample("Ada", library.sample_choices("Ada")[0][1])
    assert summary.has_voice is False
    assert not (tmp_path / "library" / "ada" / "voice.wav").exists()


def test_names_are_unique_and_delete_removes_the_directory(tmp_path: Path) -> None:
    library = FaceLibrary(tmp_path / "library")
    created = library.create("Ada")
    with pytest.raises(BarrettFaceswapError, match="already exists"):
        library.create(" ada ")
    with pytest.raises(BarrettFaceswapError, match="profile name"):
        library.create("   ")
    library.delete(created.id)
    assert not (tmp_path / "library" / created.id).exists()
    with pytest.raises(BarrettFaceswapError, match="No profile"):
        library.delete("Ada")


def test_cli_create_list_and_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    config = write_config(tmp_path, enabled=False)
    assert main(["--lang", "en", "library", "create", "--name", "Ada", "--config", str(config)]) == 0
    assert main(["--lang", "en", "library", "list", "--config", str(config)]) == 0
    listed = capsys.readouterr().out
    assert "Ada" in listed
    assert (tmp_path / "library" / "ada" / "profile.json").is_file()
    assert main(["--lang", "en", "library", "delete", "--name", "Ada", "--config", str(config)]) == 0
    assert not (tmp_path / "library" / "ada").exists()
    assert main(["--lang", "en", "swap", "--target", "t.png", "--output", "o.png", "--config", str(config)]) == 2
    assert (
        main(
            [
                "--lang",
                "en",
                "swap",
                "--source",
                "a.png",
                "--library",
                "Ada",
                "--target",
                "t.png",
                "--output",
                "o.png",
                "--config",
                str(config),
            ]
        )
        == 2
    )
    assert main(["--lang", "en", "swap", "--library", "Nope", "--target", "t.png", "--output", "o.png", "--config", str(config)]) == 5
