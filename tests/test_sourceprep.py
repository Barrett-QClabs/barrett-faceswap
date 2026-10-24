"""Source-video identity choice, speech trim, and the face-only silent target."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from barrett_faceswap.config import load_config
from barrett_faceswap.i18n import set_language, t
from barrett_faceswap.pipeline import FaceSwapPipeline
from barrett_faceswap.processors.face_swapper import FaceSwapper
from barrett_faceswap.sourceprep import (
    face_quality,
    prepare_source,
    select_identity,
    source_mode_visibility,
    trim_speech,
)
from barrett_faceswap.types import Face, ProcessContext, SwapRequest
from barrett_faceswap.ui.app import _voice_reference_for_mode, source_files_ready
from tests.conftest import write_config, write_png


def _ffmpeg_or_skip() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg is not installed")


def _config(tmp_path: Path):
    config = load_config(write_config(tmp_path, enabled=False))
    return config.model_copy(update={"video": config.video.model_copy(update={"threads": 2})})


def _face(width: int = 64, height: int = 64) -> Face:
    return Face(
        bbox=np.array([8, 8, width - 8, height - 8], dtype=np.float32),
        score=0.9,
        landmarks=np.array([[20, 24], [44, 24], [32, 36], [22, 48], [42, 48]], dtype=np.float32),
    )


def _unit(index: int, tilt: float = 0.0) -> np.ndarray:
    vector = np.zeros(8, dtype=np.float64)
    vector[0] = 1.0
    if tilt:
        vector[index] = tilt
    return vector


def test_average_wins_when_held_out_faces_agree_with_it() -> None:
    vectors = [
        _unit(1, 0.4),
        _unit(1, -0.4),
        _unit(2, 0.4),
        _unit(0),
        _unit(3, 0.05),
    ]
    qualities = [0.9, 0.8, 0.7, 0.6, 0.5]
    choice = select_identity(vectors, qualities)
    assert choice.mode == "mean"
    assert choice.mean_score > choice.best_score
    assert abs(float(choice.vector[0])) > abs(float(choice.vector[1]))


def test_best_face_wins_when_the_average_drifts() -> None:
    vectors = [
        _unit(0),
        _unit(1, 0.9),
        _unit(1, 0.9),
        _unit(0),
    ]
    qualities = [0.99, 0.7, 0.6, 0.5]
    choice = select_identity(vectors, qualities)
    assert choice.mode == "best"
    assert choice.best_score > choice.mean_score
    assert choice.vector[0] == pytest.approx(1.0)


def test_one_face_stays_that_face() -> None:
    choice = select_identity([_unit(0)], [0.8])
    assert choice.mode == "single"
    assert choice.count == 1


def test_sharp_frontal_crop_outranks_a_blur() -> None:
    sharp = np.zeros((80, 80, 3), dtype=np.uint8)
    sharp[::2, :, :] = 255
    blur = np.full((80, 80, 3), 40, dtype=np.uint8)
    face = _face(80, 80)
    assert face_quality(sharp, face) > face_quality(blur, face)


def test_trim_speech_keeps_the_harmonic_stretch() -> None:
    rate = 16000
    silence = np.zeros(int(0.3 * rate))
    times = np.arange(int(0.5 * rate)) / rate
    harmonic = sum(np.sin(2 * np.pi * freq * times) for freq in (180, 360, 540, 720, 1080))
    harmonic = 0.4 * harmonic / np.max(np.abs(harmonic))
    noise = np.random.default_rng(1).normal(0, 0.3, int(0.4 * rate))
    audio = np.concatenate([silence, harmonic, silence, noise])
    trimmed = trim_speech(audio, rate)
    assert trimmed is not None
    assert trimmed.size < int(0.7 * audio.size)
    assert trimmed.size > int(0.25 * rate)


def test_trim_speech_rejects_noise_only() -> None:
    noise = np.random.default_rng(2).normal(0, 0.2, 16000)
    assert trim_speech(noise, 16000) is None


def test_source_modes_show_one_upload_set() -> None:
    assert source_mode_visibility("video") == {
        "video": True,
        "photo": False,
        "audio": False,
        "voice_override": True,
    }
    assert source_mode_visibility("photo")["photo"] is True
    assert source_mode_visibility("photo")["audio"] is False
    assert source_mode_visibility("photo")["voice_override"] is False
    assert source_mode_visibility("photo_audio")["audio"] is True
    assert source_mode_visibility("photo_audio")["video"] is False
    assert source_files_ready("video", "clip.mp4", None, None, "target.mp4") is True
    assert source_files_ready("photo", "clip.mp4", None, None, "target.mp4") is False
    assert source_files_ready("photo", None, "face.png", None, "target.mp4") is True
    assert source_files_ready("photo_audio", None, "face.png", None, "target.mp4") is False
    assert source_files_ready("photo_audio", None, "face.png", "voice.wav", "target.mp4") is True
    assert _voice_reference_for_mode("video", None, Path("override.wav")) == Path("override.wav")
    assert _voice_reference_for_mode("video", None, None) is None
    assert _voice_reference_for_mode("photo", Path("clip.wav"), Path("override.wav")) is None
    assert _voice_reference_for_mode("photo_audio", Path("clip.wav"), Path("override.wav")) == Path("clip.wav")


def test_prepared_embedding_is_what_the_swapper_uses(tmp_path: Path) -> None:
    swapper = FaceSwapper(_config(tmp_path), ["CPUExecutionProvider"])
    swapper._backend = lambda: object()
    vector = np.zeros(8, dtype=np.float32)
    vector[0] = 4.0
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    context = ProcessContext(source_bgr=image, dry_run=False, source_embedding=vector)
    slots = swapper._identities(context)
    assert slots[0].latent[0] == pytest.approx(1.0)
    assert abs(float(slots[0].latent[1])) < 1e-6


def test_source_video_dry_run_extracts_speech_only_when_the_target_has_audio(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    set_language("en")
    source = tmp_path / "source.mp4"
    talking = tmp_path / "talking.mp4"
    silent = tmp_path / "silent.mp4"
    _clip(source, audio=True)
    _clip(talking, audio=True)
    _clip(silent, audio=False)
    called: list[str] = []

    def converter(*_args: object) -> None:
        called.append("ran")

    pipeline = FaceSwapPipeline(_config(tmp_path), voice_converter=converter)
    with_audio = pipeline.run(
        SwapRequest(
            source_path=source,
            target_path=talking,
            output_path=tmp_path / "out-voice.mp4",
            dry_run=True,
            voice_enabled=True,
        )
    )
    assert called == []
    assert with_audio.voice_status == "dry_run"
    assert t("identity.speech") in with_audio.notes
    assert t("identity.dry_frame") in with_audio.notes

    silent_result = pipeline.run(
        SwapRequest(
            source_path=source,
            target_path=silent,
            output_path=tmp_path / "out-silent.mp4",
            dry_run=True,
            voice_enabled=True,
        )
    )
    assert called == []
    assert silent_result.voice_status == "no_audio"
    assert t("identity.speech") not in silent_result.notes
    assert t("voice.no_audio") in silent_result.notes


def test_injected_detector_stores_the_chosen_embedding(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    set_language("en")
    source = tmp_path / "source.mp4"
    target = tmp_path / "target.png"
    _clip(source, audio=False)
    write_png(target, (0, 180, 0), size=(64, 64))
    face = _face()

    def detect(_image: np.ndarray) -> list[Face]:
        return [face]

    def embed(_image: np.ndarray, _face: Face) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    request = SwapRequest(
        source_path=source,
        target_path=target,
        output_path=tmp_path / "out.png",
        dry_run=False,
        voice_enabled=True,
    )
    updated, notes = prepare_source(
        _config(tmp_path),
        request,
        tmp_path / "work",
        detector=detect,
        embedder=embed,
    )
    assert updated.source_path.suffix == ".png"
    assert updated.source_embedding is not None
    assert updated.source_embedding[0] == pytest.approx(1.0)
    assert updated.voice_reference is None
    assert t("identity.speech") not in notes
    assert any("held-out" in note or "usable face" in note for note in notes)


def _clip(path: Path, *, audio: bool) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=64x64:r=10:d=0.6",
    ]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=220:sample_rate=16000:d=0.6"]
    command += ["-shortest", str(path)]
    subprocess.run(command, check=True, capture_output=True)
