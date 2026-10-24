"""Microphone chunks for a live session.

Voice conversion runs only when the camera is on, a microphone chunk arrived,
and a reference clip was resolved. Anything else stays face-only: the chunk is
not converted and nothing is played back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import VoiceError
from barrett_faceswap.sourceprep import _read_wav, _write_wav


def live_voice_plan(
    *,
    camera_on: bool,
    mic_on: bool,
    reference: Path | None,
    dry_run: bool,
) -> str:
    """``run``, ``dry_run``, or ``face_only``.

    ``face_only`` covers a stopped camera, a silent microphone, and a missing
    reference. Those cases do not call the converter.
    """

    if not camera_on or not mic_on or reference is None:
        return "face_only"
    if dry_run:
        return "dry_run"
    return "run"


def apply_live_chunk(
    samples: np.ndarray | None,
    sample_rate: int,
    *,
    camera_on: bool,
    reference: Path | None,
    dry_run: bool,
    tau: float,
    work: Path,
    converter=None,
    config: AppConfig | None = None,
    cache: dict | None = None,
) -> tuple[np.ndarray | None, str]:
    """Convert one microphone chunk, or return no audio when the plan is face-only.

    A missing voice install also stays face-only so the camera preview keeps
    running. The returned array is mono float32.
    """

    audio = _mono_float(samples)
    mic_on = audio is not None and int(audio.size) > 0
    plan = live_voice_plan(
        camera_on=camera_on,
        mic_on=mic_on,
        reference=reference,
        dry_run=dry_run,
    )
    if plan != "run" or audio is None or reference is None:
        return None, plan
    if audio.size < max(1, int(sample_rate) // 5):
        return None, "wait"
    work.mkdir(parents=True, exist_ok=True)
    source = work / "mic-chunk.wav"
    dest = work / "mic-converted.wav"
    _write_wav(source, audio, int(sample_rate))
    runner = converter if callable(converter) else None
    if runner is None and isinstance(cache, dict):
        stored = cache.get("converter")
        if callable(stored):
            runner = stored
    if runner is None:
        if config is None:
            return None, "unavailable"
        try:
            from barrett_faceswap.voice import _openvoice_converter

            runner = _openvoice_converter(config)
        except VoiceError:
            return None, "unavailable"
    try:
        runner(source, reference, dest, float(tau))
    except VoiceError:
        return None, "unavailable"
    if not dest.is_file():
        return None, "unavailable"
    if isinstance(cache, dict):
        cache["converter"] = runner
    converted, _rate = _read_wav(dest)
    return converted, "run"


def _mono_float(samples: np.ndarray | None) -> np.ndarray | None:
    if samples is None:
        return None
    raw = np.asarray(samples)
    if raw.size == 0:
        return np.zeros(0, dtype=np.float32)
    original_dtype = raw.dtype
    if raw.ndim == 2:
        raw = raw.mean(axis=1)
    raw = raw.reshape(-1)
    if np.issubdtype(original_dtype, np.integer):
        audio = raw.astype(np.float32) / float(np.iinfo(original_dtype).max)
    else:
        audio = raw.astype(np.float32)
    return np.nan_to_num(audio, copy=False)
