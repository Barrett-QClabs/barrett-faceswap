"""Live camera voice: reference order, and conversion only with camera, mic, and a clip."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from barrett_faceswap.config import load_config
from barrett_faceswap.errors import VoiceError
from barrett_faceswap.live import LiveSession
from barrett_faceswap.live_voice import apply_live_chunk, live_voice_plan
from barrett_faceswap.sourceprep import _write_wav, resolve_live_reference
from barrett_faceswap.types import SwapRequest
from barrett_faceswap.ui.app import _live_voice_key, _mic_chunk
from tests.conftest import write_config, write_png


def _ffmpeg_or_skip() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")


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


def test_plan_needs_camera_mic_and_reference(tmp_path: Path) -> None:
    reference = tmp_path / "ref.wav"
    assert live_voice_plan(camera_on=True, mic_on=True, reference=reference, dry_run=False) == "run"
    assert live_voice_plan(camera_on=True, mic_on=True, reference=reference, dry_run=True) == "dry_run"
    assert live_voice_plan(camera_on=False, mic_on=True, reference=reference, dry_run=False) == "face_only"
    assert live_voice_plan(camera_on=True, mic_on=False, reference=reference, dry_run=False) == "face_only"
    assert live_voice_plan(camera_on=True, mic_on=True, reference=None, dry_run=False) == "face_only"


def test_status_keys_cover_each_plan() -> None:
    assert _live_voice_key("run") == "ui.live_voice_ready"
    assert _live_voice_key("dry_run") == "ui.live_voice_dry"
    assert _live_voice_key("wait") == "ui.live_voice_wait"
    assert _live_voice_key("unavailable") == "ui.live_voice_unavailable"
    assert _live_voice_key("face_only") == "ui.live_face_only"


def test_mic_chunk_reads_gradio_tuple() -> None:
    samples, rate = _mic_chunk((16000, np.ones(4, dtype=np.float32)))
    assert rate == 16000
    assert samples is not None and samples.shape == (4,)
    assert _mic_chunk(None) == (None, 16000)


def test_face_only_and_dry_run_do_not_call_the_converter(tmp_path: Path) -> None:
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"unused")

    def converter(*_args: object) -> None:
        raise AssertionError("converter should stay idle")

    audio = np.ones(800, dtype=np.float32)
    silent, silent_plan = apply_live_chunk(
        audio,
        1600,
        camera_on=False,
        reference=reference,
        dry_run=False,
        tau=0.3,
        work=tmp_path,
        converter=converter,
    )
    missing, missing_plan = apply_live_chunk(
        audio,
        1600,
        camera_on=True,
        reference=None,
        dry_run=False,
        tau=0.3,
        work=tmp_path,
        converter=converter,
    )
    empty, empty_plan = apply_live_chunk(
        np.zeros(0, dtype=np.float32),
        1600,
        camera_on=True,
        reference=reference,
        dry_run=False,
        tau=0.3,
        work=tmp_path,
        converter=converter,
    )
    dry, dry_plan = apply_live_chunk(
        audio,
        1600,
        camera_on=True,
        reference=reference,
        dry_run=True,
        tau=0.3,
        work=tmp_path,
        converter=converter,
    )
    assert silent is None and silent_plan == "face_only"
    assert missing is None and missing_plan == "face_only"
    assert empty is None and empty_plan == "face_only"
    assert dry is None and dry_plan == "dry_run"


def test_short_chunk_waits_and_a_real_chunk_converts(tmp_path: Path) -> None:
    reference = tmp_path / "ref.wav"
    _write_wav(reference, np.full(400, 0.2, dtype=np.float32), 1600)
    calls: list[float] = []

    def converter(source: Path, ref: Path, dest: Path, tau: float) -> None:
        calls.append(tau)
        assert source.is_file()
        assert ref == reference
        _write_wav(dest, np.full(400, 0.5, dtype=np.float32), 1600)

    waiting, wait_plan = apply_live_chunk(
        np.ones(10, dtype=np.float32),
        1600,
        camera_on=True,
        reference=reference,
        dry_run=False,
        tau=0.3,
        work=tmp_path,
        converter=converter,
    )
    converted, plan = apply_live_chunk(
        np.full(800, 0.1, dtype=np.float32),
        1600,
        camera_on=True,
        reference=reference,
        dry_run=False,
        tau=0.3,
        work=tmp_path,
        converter=converter,
    )
    assert waiting is None and wait_plan == "wait"
    assert plan == "run"
    assert converted is not None
    assert float(np.max(np.abs(converted))) > 0.2
    assert calls == [0.3]


def test_converter_is_cached_and_failures_stay_face_only(tmp_path: Path) -> None:
    reference = tmp_path / "ref.wav"
    _write_wav(reference, np.full(200, 0.2, dtype=np.float32), 1600)
    cache: dict[str, object] = {}
    calls = {"n": 0}

    def converter(_source: Path, _ref: Path, dest: Path, _tau: float) -> None:
        calls["n"] += 1
        _write_wav(dest, np.full(200, 0.4, dtype=np.float32), 1600)

    audio = np.full(800, 0.1, dtype=np.float32)
    apply_live_chunk(
        audio,
        1600,
        camera_on=True,
        reference=reference,
        dry_run=False,
        tau=0.2,
        work=tmp_path,
        converter=converter,
        cache=cache,
    )
    apply_live_chunk(
        audio,
        1600,
        camera_on=True,
        reference=reference,
        dry_run=False,
        tau=0.2,
        work=tmp_path,
        converter=None,
        cache=cache,
    )
    assert calls["n"] == 2
    assert cache["converter"] is converter

    def broken(*_args: object) -> None:
        raise VoiceError("missing")

    failed, plan = apply_live_chunk(
        audio,
        1600,
        camera_on=True,
        reference=reference,
        dry_run=False,
        tau=0.2,
        work=tmp_path,
        converter=broken,
    )
    missing, missing_plan = apply_live_chunk(
        audio,
        1600,
        camera_on=True,
        reference=reference,
        dry_run=False,
        tau=0.2,
        work=tmp_path,
        converter=None,
        config=None,
    )
    assert failed is None and plan == "unavailable"
    assert missing is None and missing_plan == "unavailable"


def test_reference_order_is_upload_then_source_video_then_none(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    source = tmp_path / "source.mp4"
    silent = tmp_path / "silent.mp4"
    _clip(source, audio=True)
    _clip(silent, audio=False)
    override = tmp_path / "override.wav"
    override.write_bytes(b"clip")
    photo_audio = tmp_path / "beside.wav"
    photo_audio.write_bytes(b"clip")
    work = tmp_path / "work"

    uploaded, upload_kind = resolve_live_reference("video", source, override, None, work)
    extracted, source_kind = resolve_live_reference("video", source, None, None, work)
    quiet, quiet_kind = resolve_live_reference("video", silent, None, None, work)
    photo, photo_kind = resolve_live_reference("photo", tmp_path / "face.png", override, photo_audio, work)
    paired, paired_kind = resolve_live_reference("photo_audio", tmp_path / "face.png", override, photo_audio, work)
    missing, missing_kind = resolve_live_reference(
        "photo_audio", tmp_path / "face.png", None, tmp_path / "missing.wav", work
    )

    assert uploaded == override and upload_kind == "upload"
    assert extracted is not None and extracted.is_file() and source_kind == "source"
    assert quiet is None and quiet_kind == "none"
    assert photo is None and photo_kind == "none"
    assert paired == photo_audio and paired_kind == "upload"
    assert missing is None and missing_kind == "none"


def test_live_session_keeps_the_prepared_embedding(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    write_png(source, (30, 60, 90), (64, 48))
    config = load_config(write_config(tmp_path, enabled=False))
    request = SwapRequest(
        source_path=source,
        target_path=Path("camera-ui"),
        output_path=tmp_path / "live-session.txt",
        dry_run=True,
        live_session=True,
        source_embedding=(1.0, 0.0, 0.0, 0.0),
        processors=[],
    )
    session = LiveSession(config, request, processors=[])
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    session.start(frame)
    assert session._shared is not None
    assert session._shared.source_embedding is not None
    assert float(session._shared.source_embedding[0]) == pytest.approx(1.0)
    assert session.step(frame).shape == frame.shape
