"""Zero-shot voice conversion after a video face swap.

The face pipeline is unchanged. This module runs only when voice is enabled,
a reference clip is present, and the target video has an audio stream. A
silent target, a still image, or a dry-run never loads the voice model.

The engine is OpenVoice V2's tone-color converter. It keeps the words in the
target track and replaces the speaker timbre from a short reference. Weights
stay on disk under ``voice.checkpoint_dir`` and are not downloaded here.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import BarrettFaceswapError, VoiceError
from barrett_faceswap.i18n import t
from barrett_faceswap.media.ffmpeg import (
    extract_audio,
    ffmpeg_available,
    media_duration,
    mix_wavs,
    probe,
    render_wav,
    replace_video_audio,
)
from barrett_faceswap.types import SwapRequest, SwapResult
from barrett_faceswap.vision import is_video

VOICE_REFERENCE_SUFFIXES = {".wav", ".mp3", ".m4a"}
VoiceConverter = Callable[[Path, Path, Path, float], None]

_CONVERTER_CACHE: dict[tuple[str, str, str, bool], object] = {}


def voice_is_enabled(config: AppConfig, request: SwapRequest) -> bool:
    """True when this run asked for voice conversion.

    An explicit request flag wins. Otherwise the config default applies only
    together with a reference clip, so a normal CLI swap stays face-only.
    """

    if request.voice_enabled is not None:
        return bool(request.voice_enabled)
    return bool(config.voice.enabled) and request.voice_reference is not None


def voice_strength(config: AppConfig, request: SwapRequest) -> float:
    if request.voice_strength is None:
        return float(config.voice.strength)
    return float(request.voice_strength)


def voice_tau(config: AppConfig, request: SwapRequest) -> float:
    if request.voice_tau is None:
        return float(config.voice.tau)
    return float(request.voice_tau)


def plan_voice(
    config: AppConfig,
    request: SwapRequest,
    *,
    is_video_target: bool,
    has_audio: bool,
) -> str:
    """Decide the voice step without loading a model.

    ``run`` is the only value that may call the converter. ``no_audio`` wins
    over a missing reference: a silent target never enters conversion.
    """

    if not is_video_target or not voice_is_enabled(config, request):
        return "off"
    if not has_audio:
        return "no_audio"
    if request.voice_reference is None:
        return "no_reference"
    if request.dry_run:
        return "dry_run"
    return "run"


def target_has_audio(path: Path | None) -> bool:
    """True when ``path`` is a video whose container has an audio stream."""

    if path is None or not is_video(path) or not ffmpeg_available():
        return False
    try:
        return bool(probe(path).has_audio)
    except BarrettFaceswapError:
        return False


def voice_toggle_for_target(path: Path | None, *, auto_enable: bool) -> tuple[bool, str]:
    """Checkbox value and status line for the Gradio voice panel.

    The toggle turns on only when auto-enable is set and the target video has
    audio. A silent video or a still stays off. The caller still has to supply
    a reference clip before conversion runs.
    """

    if path is None:
        return False, t("ui.voice_idle")
    if not is_video(path):
        return False, t("ui.voice_still")
    if not ffmpeg_available():
        return False, t("ui.voice_probe_failed")
    try:
        has_audio = bool(probe(path).has_audio)
    except BarrettFaceswapError:
        return False, t("ui.voice_probe_failed")
    if not has_audio:
        return False, t("ui.voice_no_audio")
    if not auto_enable:
        return False, t("ui.voice_auto_off")
    return True, t("ui.voice_ready")


def atempo_chain(factor: float) -> str:
    """ffmpeg ``atempo`` chain. Each stage stays inside 0.5–2.0."""

    if factor <= 0:
        raise VoiceError(t("voice.bad_duration"))
    parts: list[str] = []
    remaining = float(factor)
    while remaining > 2.0 + 1e-6:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5 - 1e-6:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def length_filter(audio_duration: float, video_duration: float) -> tuple[str, str]:
    """Fit speech to the picture.

    A small gap is time-stretched so the words stay near the mouth. A large
    gap is padded or trimmed instead, which avoids a chipmunk track and keeps
    the full video. The return value is ``(ffmpeg audio filter, mode)``.
    """

    if audio_duration <= 0 or video_duration <= 0:
        raise VoiceError(t("voice.bad_duration"))
    ratio = audio_duration / video_duration
    end = f"{video_duration:.6f}"
    tail = f"apad=whole_dur={end},atrim=0:{end},asetpts=PTS-STARTPTS"
    if 0.85 <= ratio <= 1.15 and abs(ratio - 1.0) > 0.02:
        return f"{atempo_chain(ratio)},{tail}", "stretch"
    return tail, "pad"


def checkpoint_paths(config: AppConfig) -> tuple[Path, Path]:
    root = config.voice.checkpoint_dir
    return root / "converter" / "config.json", root / "converter" / "checkpoint.pth"


def ensure_openvoice_ready(config: AppConfig) -> None:
    """Fail before the face swap when the optional engine cannot start."""

    config_path, checkpoint = checkpoint_paths(config)
    if not config_path.is_file() or not checkpoint.is_file():
        raise VoiceError(t("voice.checkpoint_missing", path=config.voice.checkpoint_dir))
    try:
        import librosa  # noqa: F401
        import soundfile  # noqa: F401
        import torch  # noqa: F401
        import openvoice.api  # noqa: F401
    except ImportError as exc:
        raise VoiceError(t("voice.deps_missing", detail=exc)) from exc


def check_voice(
    config: AppConfig,
    request: SwapRequest,
    *,
    converter: VoiceConverter | None = None,
) -> str:
    """Return the voice plan. Raise when a real conversion cannot start.

    Called before frame processing so a missing checkpoint does not wait until
    the face swap has finished.
    """

    if not is_video(request.target_path):
        if voice_is_enabled(config, request):
            return "no_audio"
        return "off"
    has_audio = bool(probe(request.target_path).has_audio)
    plan = plan_voice(config, request, is_video_target=True, has_audio=has_audio)
    if plan != "run":
        return plan
    reference = request.voice_reference
    if reference is None or not reference.is_file():
        raise VoiceError(t("voice.reference_missing", path=reference or "-"))
    suffix = reference.suffix.lower()
    if suffix not in VOICE_REFERENCE_SUFFIXES:
        raise VoiceError(
            t(
                "voice.reference_suffix",
                suffix=suffix or "-",
                allowed="wav, mp3, m4a",
            )
        )
    strength = voice_strength(config, request)
    tau = voice_tau(config, request)
    if not 0.0 <= strength <= 1.0 or not 0.0 <= tau <= 1.0:
        raise VoiceError(t("config.unit_interval"))
    if strength <= 0.0:
        return "strength_zero"
    if converter is None:
        ensure_openvoice_ready(config)
    return "run"


def apply_voice(
    config: AppConfig,
    request: SwapRequest,
    result: SwapResult,
    *,
    converter: VoiceConverter | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> SwapResult:
    """Convert and mux when ``check_voice`` says ``run``. Other plans only annotate."""

    plan = check_voice(config, request, converter=converter)
    if plan != "run":
        result.voice_status = plan
        if plan == "no_audio":
            result.notes.append(t("voice.no_audio"))
        elif plan == "no_reference":
            result.notes.append(t("voice.no_reference"))
        elif plan == "dry_run":
            result.notes.append(t("voice.dry_run"))
        elif plan == "strength_zero":
            result.notes.append(t("voice.strength_zero"))
        return result
    reference = request.voice_reference
    if reference is None:
        result.voice_status = "no_reference"
        result.notes.append(t("voice.no_reference"))
        return result
    if on_stage is not None:
        on_stage("voice")
    runner = converter or _openvoice_converter(config)
    strength = voice_strength(config, request)
    tau = voice_tau(config, request)
    with tempfile.TemporaryDirectory(prefix="barrett-voice-") as directory:
        root = Path(directory)
        source_wav = root / "source.wav"
        converted = root / "converted.wav"
        extract_audio(
            request.target_path,
            source_wav,
            trim_start=request.trim_start,
            trim_end=request.trim_end,
        )
        try:
            runner(source_wav, reference, converted, tau)
        except VoiceError:
            raise
        except Exception as exc:
            _raise_engine(exc)
        if not converted.is_file():
            raise VoiceError(t("voice.no_output"))
        if on_stage is not None:
            on_stage("mux")
        duration = media_duration(request.output_path)
        fitted = root / "fitted.wav"
        _filter, mode = _fit(converted, fitted, duration)
        audio = fitted
        if strength < 1.0:
            fitted_source = root / "fitted-source.wav"
            _fit(source_wav, fitted_source, duration)
            mixed = root / "mixed.wav"
            mix_wavs(
                fitted_source,
                fitted,
                mixed,
                original_weight=1.0 - strength,
                converted_weight=strength,
            )
            audio = mixed
            mode = "mix"
        replace_video_audio(request.output_path, audio)
    result.voice_status = "converted"
    result.notes.append(t("voice.converted", mode=mode))
    return result


def _fit(source: Path, dest: Path, duration: float) -> tuple[str, str]:
    audio_filter, mode = length_filter(media_duration(source), duration)
    render_wav(source, dest, audio_filter=audio_filter)
    return audio_filter, mode


def _openvoice_converter(config: AppConfig) -> VoiceConverter:
    def _run(source: Path, reference: Path, output: Path, tau: float) -> None:
        convert_openvoice(source, reference, output, tau=tau, config=config)

    return _run


def convert_openvoice(
    source_wav: Path,
    reference_wav: Path,
    output_wav: Path,
    *,
    tau: float,
    config: AppConfig,
) -> None:
    """Run the OpenVoice V2 tone-color converter. Imports torch lazily."""

    converter = _load_openvoice(config)
    try:
        source_embedding = converter.extract_se(str(source_wav))
        reference_embedding = converter.extract_se(str(reference_wav))
        converter.convert(
            audio_src_path=str(source_wav),
            src_se=source_embedding,
            tgt_se=reference_embedding,
            output_path=str(output_wav),
            tau=float(tau),
        )
    except VoiceError:
        raise
    except Exception as exc:
        _raise_engine(exc)


def _load_openvoice(config: AppConfig):
    ensure_openvoice_ready(config)
    import torch
    from openvoice.api import OpenVoiceBaseClass, ToneColorConverter

    config_path, checkpoint = checkpoint_paths(config)
    device = _resolve_device(config, torch)
    watermark = bool(config.voice.watermark)
    key = (str(config_path), str(checkpoint), device, watermark)
    cached = _CONVERTER_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        if watermark:
            converter = ToneColorConverter(str(config_path), device=device)
        else:
            # The upstream constructor always imports wavmark, and it rejects
            # enable_watermark as a keyword. Build the same object without that import.
            converter = ToneColorConverter.__new__(ToneColorConverter)
            OpenVoiceBaseClass.__init__(converter, str(config_path), device=device)
            converter.watermark_model = None
            converter.version = getattr(converter.hps, "_version_", "v1")
        converter.load_ckpt(str(checkpoint))
    except VoiceError:
        raise
    except Exception as exc:
        _raise_engine(exc)
    _CONVERTER_CACHE.clear()
    _CONVERTER_CACHE[key] = converter
    return converter


def _resolve_device(config: AppConfig, torch_module: object) -> str:
    choice = config.voice.device
    available = bool(torch_module.cuda.is_available())
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        if not available:
            raise VoiceError(t("voice.cuda_unavailable"))
        return "cuda:0"
    return "cuda:0" if available else "cpu"


def _raise_engine(exc: Exception) -> None:
    text = str(exc).strip() or exc.__class__.__name__
    lowered = text.lower()
    if (
        "no kernel image" in lowered
        or "sm_120" in lowered
        or "not compatible with the current pytorch" in lowered
    ):
        raise VoiceError(t("voice.cuda_arch", detail=text[-500:])) from exc
    if isinstance(exc, ImportError):
        raise VoiceError(t("voice.deps_missing", detail=text[-500:])) from exc
    raise VoiceError(t("voice.failed", detail=text[-500:])) from exc
