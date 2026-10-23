"""Voice plan, length fit, and mux. The neural converter is a stand-in."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from barrett_faceswap.config import load_config
from barrett_faceswap.errors import VoiceError
from barrett_faceswap.i18n import t
from barrett_faceswap.media.ffmpeg import build_replace_audio_command, probe
from barrett_faceswap.pipeline import FaceSwapPipeline
from barrett_faceswap.types import SwapRequest
from barrett_faceswap.voice import apply_voice, atempo_chain, length_filter, plan_voice, voice_toggle_for_target
from tests.conftest import write_config, write_png


def _ffmpeg_or_skip() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg is not installed")


def _make_clip(path: Path, *, seconds: float = 0.4, audio: bool = False, frequency: int = 440) -> None:
    _ffmpeg_or_skip()
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=red:s=160x120:r=10:d={seconds}",
    ]
    if audio:
        command += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:duration={seconds}",
            "-shortest",
            "-c:a",
            "aac",
        ]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _make_wav(path: Path, *, seconds: float, frequency: int) -> None:
    _ffmpeg_or_skip()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:duration={seconds}",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _dominant_hz(path: Path) -> float:
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
    )
    samples = np.frombuffer(raw, dtype=np.float32)
    if samples.size < 80:
        return 0.0
    signs = np.sign(samples)
    signs[signs == 0] = 1
    crossings = int(np.count_nonzero(signs[1:] * signs[:-1] < 0))
    return (crossings / 2) / (samples.size / 8000)


def _moov_precedes_mdat(path: Path) -> None:
    data = path.read_bytes()
    moov = data.find(b"moov")
    mdat = data.find(b"mdat")
    assert moov >= 0 and mdat >= 0
    assert moov < mdat


def _request(target: Path, output: Path, **kwargs: object) -> SwapRequest:
    source = target.parent / "source.png"
    if not source.is_file():
        write_png(source, (0, 0, 255), size=(160, 120))
    payload: dict[str, object] = {
        "source_path": source,
        "target_path": target,
        "output_path": output,
        "dry_run": True,
    }
    payload.update(kwargs)
    return SwapRequest(**payload)


def _config(tmp_path: Path):
    config = load_config(write_config(tmp_path, enabled=False))
    return config.model_copy(
        update={
            "video": config.video.model_copy(update={"threads": 2}),
            "voice": config.voice.model_copy(update={"checkpoint_dir": tmp_path / "no-voice-weights"}),
        }
    )


def test_plan_skips_when_the_target_has_no_audio(tmp_path: Path) -> None:
    config = _config(tmp_path)
    reference = tmp_path / "ref.wav"
    request = SwapRequest(
        source_path=tmp_path / "source.png",
        target_path=tmp_path / "clip.mp4",
        output_path=tmp_path / "out.mp4",
        voice_enabled=True,
        voice_reference=reference,
    )
    assert plan_voice(config, request, is_video_target=True, has_audio=False) == "no_audio"
    assert plan_voice(config, request, is_video_target=False, has_audio=False) == "off"
    silent = request.model_copy(update={"voice_reference": None})
    assert plan_voice(config, silent, is_video_target=True, has_audio=False) == "no_audio"
    preview = request.model_copy(update={"dry_run": True})
    assert plan_voice(config, preview, is_video_target=True, has_audio=True) == "dry_run"
    talking = request.model_copy(update={"dry_run": False})
    assert plan_voice(config, talking, is_video_target=True, has_audio=True) == "run"
    missing_ref = talking.model_copy(update={"voice_reference": None})
    assert plan_voice(config, missing_ref, is_video_target=True, has_audio=True) == "no_reference"
    disabled = talking.model_copy(update={"voice_enabled": False})
    assert plan_voice(config, disabled, is_video_target=True, has_audio=True) == "off"
    implied = talking.model_copy(update={"voice_enabled": None})
    assert plan_voice(config, implied, is_video_target=True, has_audio=True) == "run"
    config_off = config.model_copy(update={"voice": config.voice.model_copy(update={"enabled": False})})
    assert plan_voice(config_off, implied, is_video_target=True, has_audio=True) == "off"


def test_length_filter_stretches_a_small_gap_and_pads_a_large_one() -> None:
    exact, exact_mode = length_filter(1.0, 1.0)
    assert exact_mode == "pad"
    assert "atempo" not in exact
    stretch, stretch_mode = length_filter(1.1, 1.0)
    assert stretch_mode == "stretch"
    assert "atempo=1.100000" in stretch
    long_audio, long_mode = length_filter(3.0, 1.0)
    assert long_mode == "pad"
    assert "atempo" not in long_audio
    assert atempo_chain(4.0) == "atempo=2.0,atempo=2.000000"
    assert atempo_chain(0.25) == "atempo=0.5,atempo=0.500000"


def test_toggle_follows_target_audio(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    silent = tmp_path / "silent.mp4"
    talking = tmp_path / "talking.mp4"
    still = tmp_path / "still.png"
    _make_clip(silent, audio=False)
    _make_clip(talking, audio=True)
    write_png(still, size=(32, 32))
    assert voice_toggle_for_target(None, auto_enable=True)[0] is False
    assert voice_toggle_for_target(still, auto_enable=True)[0] is False
    assert voice_toggle_for_target(silent, auto_enable=True) == (False, t("ui.voice_no_audio"))
    assert voice_toggle_for_target(talking, auto_enable=True) == (True, t("ui.voice_ready"))
    assert voice_toggle_for_target(talking, auto_enable=False) == (False, t("ui.voice_auto_off"))


def test_replace_audio_command_keeps_faststart(tmp_path: Path) -> None:
    command = build_replace_audio_command(tmp_path / "in.mp4", tmp_path / "voice.wav", tmp_path / "out.mp4")
    assert command[-3:] == ["-movflags", "+faststart", str(tmp_path / "out.mp4")]
    assert "-c:v" in command and "copy" in command
    assert "-shortest" not in command


def test_silent_target_does_not_call_the_converter(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    target = tmp_path / "silent.mp4"
    output = tmp_path / "out.mp4"
    reference = tmp_path / "ref.wav"
    _make_clip(target, audio=False)
    _make_wav(reference, seconds=0.3, frequency=880)
    called: list[float] = []

    def converter(source: Path, ref: Path, dest: Path, tau: float) -> None:
        called.append(tau)
        dest.write_bytes(b"should-not-run")

    result = FaceSwapPipeline(_config(tmp_path), voice_converter=converter).run(
        _request(target, output, voice_enabled=True, voice_reference=reference)
    )
    assert called == []
    assert result.voice_status == "no_audio"
    assert t("voice.no_audio") in result.notes
    assert probe(output).has_audio is False


def test_voice_off_keeps_the_original_track(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    target = tmp_path / "talking.mp4"
    output = tmp_path / "out.mp4"
    reference = tmp_path / "ref.wav"
    _make_clip(target, audio=True, frequency=440)
    _make_wav(reference, seconds=0.3, frequency=880)
    called: list[float] = []

    def converter(*_args: object) -> None:
        called.append(1.0)

    result = FaceSwapPipeline(_config(tmp_path), voice_converter=converter).run(
        _request(target, output, voice_enabled=False, voice_reference=reference)
    )
    assert called == []
    assert result.voice_status == "off"
    assert probe(output).has_audio is True
    assert _dominant_hz(output) == pytest.approx(440, abs=30)


def test_converter_audio_is_muxed_and_trimmed_to_the_video(tmp_path: Path) -> None:
    """Face-swap dry-run first, then the voice step the pipeline runs after it.

    A real OpenVoice weight is not in CI. The stand-in writes a longer 880 Hz
    clip so the mux has to trim it back to the picture.
    """

    _ffmpeg_or_skip()
    target = tmp_path / "talking.mp4"
    output = tmp_path / "out.mp4"
    reference = tmp_path / "ref.wav"
    _make_clip(target, seconds=0.4, audio=True, frequency=440)
    _make_wav(reference, seconds=0.3, frequency=880)
    config = _config(tmp_path)
    face = FaceSwapPipeline(config).run(_request(target, output))
    picture = probe(output)
    assert picture.has_audio is True
    seen: list[float] = []

    def converter(source: Path, ref: Path, dest: Path, tau: float) -> None:
        seen.append(tau)
        assert source.is_file()
        assert ref == reference
        _make_wav(dest, seconds=1.5, frequency=880)

    request = _request(
        target,
        output,
        voice_enabled=True,
        voice_reference=reference,
        dry_run=False,
        voice_tau=0.3,
    )
    result = apply_voice(config, request, face, converter=converter)
    assert seen == [0.3]
    assert result.voice_status == "converted"
    assert any("pad" in note for note in result.notes)
    info = probe(output)
    assert info.has_audio is True
    assert info.duration == pytest.approx(picture.duration, abs=0.05)
    assert _dominant_hz(output) == pytest.approx(880, abs=40)
    _moov_precedes_mdat(output)


def test_dry_run_pipeline_keeps_audio_when_a_reference_is_set(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    target = tmp_path / "talking.mp4"
    output = tmp_path / "out.mp4"
    reference = tmp_path / "ref.wav"
    _make_clip(target, audio=True, frequency=440)
    _make_wav(reference, seconds=0.2, frequency=880)
    called: list[str] = []

    def converter(*_args: object) -> None:
        called.append("ran")

    result = FaceSwapPipeline(_config(tmp_path), voice_converter=converter).run(
        _request(target, output, voice_enabled=True, voice_reference=reference)
    )
    assert called == []
    assert result.voice_status == "dry_run"
    assert t("voice.dry_run") in result.notes
    assert _dominant_hz(output) == pytest.approx(440, abs=30)


def test_missing_checkpoint_fails_before_a_face_swap(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    target = tmp_path / "talking.mp4"
    output = tmp_path / "out.mp4"
    reference = tmp_path / "ref.wav"
    _make_clip(target, audio=True)
    _make_wav(reference, seconds=0.2, frequency=550)
    with pytest.raises(VoiceError) as caught:
        FaceSwapPipeline(_config(tmp_path)).run(
            _request(target, output, voice_enabled=True, voice_reference=reference, dry_run=False)
        )
    assert "OpenVoice" in str(caught.value) or "openvoice" in str(caught.value).lower() or "converter" in str(caught.value)
    assert not output.is_file()


def test_bad_reference_suffix_is_rejected(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    target = tmp_path / "talking.mp4"
    reference = tmp_path / "ref.txt"
    _make_clip(target, audio=True)
    reference.write_text("not audio", encoding="utf-8")
    with pytest.raises(VoiceError):
        FaceSwapPipeline(_config(tmp_path)).run(
            _request(
                target,
                tmp_path / "out.mp4",
                voice_enabled=True,
                voice_reference=reference,
                dry_run=False,
            )
        )


def test_strength_zero_does_not_load_a_model(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    target = tmp_path / "talking.mp4"
    output = tmp_path / "out.mp4"
    reference = tmp_path / "ref.wav"
    _make_clip(target, audio=True, frequency=440)
    _make_wav(reference, seconds=0.2, frequency=880)
    config = _config(tmp_path)
    face = FaceSwapPipeline(config).run(_request(target, output))

    def converter(*_args: object) -> None:
        raise AssertionError("converter should not run")

    result = apply_voice(
        config,
        _request(
            target,
            output,
            voice_enabled=True,
            voice_reference=reference,
            dry_run=False,
            voice_strength=0,
        ),
        face,
        converter=converter,
    )
    assert result.voice_status == "strength_zero"
    assert probe(output).has_audio is True
    assert _dominant_hz(output) == pytest.approx(440, abs=30)
