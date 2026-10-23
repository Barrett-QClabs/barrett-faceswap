"""Probe, extract, and encode video without copying another project's ffmpeg builder.

Frames are written to a directory so the compliance gate can sample them and
so a visible watermark can be burned in before the encoder runs. A pipe that
never touches disk is left for a later phase.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from barrett_faceswap.errors import MediaToolError
from barrett_faceswap.i18n import t
from barrett_faceswap.vision import read_image

VIDEO_OUTPUT_SUFFIXES = {".mp4", ".mkv", ".mov"}


@dataclass(frozen=True)
class MediaInfo:
    """Video stream facts the encoder needs."""

    path: Path
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool


def ffmpeg_available() -> bool:
    """True when both ffmpeg and ffprobe are on PATH."""

    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def sample_indexes(frame_count: int, every: int) -> list[int]:
    """Frame indexes visited by the content filter, always including the last."""

    if frame_count <= 0:
        return []
    step = max(1, int(every))
    chosen = set(range(0, frame_count, step))
    chosen.add(frame_count - 1)
    return sorted(chosen)


def probe(path: Path) -> MediaInfo:
    """Read width, height, fps, duration, and whether an audio stream exists."""

    if not path.is_file():
        raise MediaToolError(t("vision.missing", path=path))
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise MediaToolError(t("video.ffprobe_missing"))
    completed = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ]
    )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaToolError(t("video.probe_parse", path=path)) from exc
    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not isinstance(video, dict):
        raise MediaToolError(t("video.no_stream", path=path))
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    fps = _ratio(str(video.get("avg_frame_rate") or "")) or _ratio(str(video.get("r_frame_rate") or ""))
    if fps <= 0:
        fps = 25.0
    duration = _float(video.get("duration"))
    if duration <= 0:
        duration = _float((payload.get("format") or {}).get("duration"))
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise MediaToolError(t("video.no_stream", path=path))
    return MediaInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        has_audio=has_audio,
    )


def build_extract_command(
    source: Path,
    directory: Path,
    *,
    trim_start: float | None,
    trim_end: float | None,
) -> list[str]:
    """ffmpeg argv that writes ``frame_000001.png`` and the frames after it."""

    ffmpeg = _ffmpeg_bin()
    pattern = directory / "frame_%06d.png"
    command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if trim_start is not None:
        command += ["-ss", _seconds(trim_start)]
    if trim_end is not None:
        command += ["-to", _seconds(trim_end)]
    command += ["-i", str(source), "-map", "0:v:0", str(pattern)]
    return command


def extract_frames(
    source: Path,
    directory: Path,
    *,
    trim_start: float | None,
    trim_end: float | None,
) -> list[Path]:
    """Extract PNG frames. The returned list is sorted by filename."""

    directory.mkdir(parents=True, exist_ok=True)
    _run(build_extract_command(source, directory, trim_start=trim_start, trim_end=trim_end))
    frames = sorted(directory.glob("frame_*.png"))
    if not frames:
        raise MediaToolError(t("video.no_frames", path=source))
    return frames


def build_encode_command(
    frames_dir: Path,
    output: Path,
    *,
    fps: float,
    encoder: str,
    crf: int,
    audio_source: Path | None,
    audio_mode: str,
    trim_start: float | None,
    trim_end: float | None,
) -> list[str]:
    """ffmpeg argv for ``libx264`` or ``h264_nvenc``.

    ``audio_mode`` is ``none``, ``copy``, or ``aac``. Trim bounds apply to the
    audio input. Frames are already the trimmed range.
    """

    if encoder not in {"libx264", "h264_nvenc"}:
        raise MediaToolError(t("config.encoder", allowed="libx264, h264_nvenc"))
    if audio_mode not in {"none", "copy", "aac"}:
        raise MediaToolError(t("video.audio_mode", mode=audio_mode))
    ffmpeg = _ffmpeg_bin()
    pattern = frames_dir / "frame_%06d.png"
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{float(fps):.6f}",
        "-i",
        str(pattern),
    ]
    if audio_mode != "none" and audio_source is not None:
        if trim_start is not None:
            command += ["-ss", _seconds(trim_start)]
        if trim_end is not None:
            command += ["-to", _seconds(trim_end)]
        command += ["-i", str(audio_source)]
    command += ["-map", "0:v:0"]
    if audio_mode != "none" and audio_source is not None:
        command += ["-map", "1:a:0"]
    command += ["-c:v", encoder]
    if encoder == "libx264":
        command += ["-preset", "veryfast", "-crf", str(int(crf))]
    else:
        command += ["-preset", "p4", "-rc", "vbr", "-cq", str(int(crf)), "-b:v", "0"]
    command += ["-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    if audio_mode == "copy":
        command += ["-c:a", "copy"]
    elif audio_mode == "aac":
        command += ["-c:a", "aac", "-b:a", "192k"]
    if audio_mode != "none":
        command += ["-shortest"]
    command += [str(output)]
    return command


def encode_video(
    frames_dir: Path,
    output: Path,
    *,
    fps: float,
    encoder: str,
    crf: int,
    audio_source: Path | None,
    keep_audio: bool,
    has_audio: bool,
    trim_start: float | None,
    trim_end: float | None,
) -> str:
    """Encode frames and mux the source audio back when it exists.

    Returns the audio mode that succeeded (``none``, ``copy``, or ``aac``).
    A failed stream copy is retried as AAC. NVENC is not replaced with libx264.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    trimmed = trim_start is not None or trim_end is not None
    if keep_audio and has_audio and audio_source is not None:
        mode = "aac" if trimmed else "copy"
    else:
        mode = "none"
    command = build_encode_command(
        frames_dir,
        output,
        fps=fps,
        encoder=encoder,
        crf=crf,
        audio_source=audio_source,
        audio_mode=mode,
        trim_start=trim_start,
        trim_end=trim_end,
    )
    if mode == "copy":
        completed = _run(command, check=False)
        if completed.returncode == 0 and output.is_file():
            return "copy"
        output.unlink(missing_ok=True)
        mode = "aac"
        command = build_encode_command(
            frames_dir,
            output,
            fps=fps,
            encoder=encoder,
            crf=crf,
            audio_source=audio_source,
            audio_mode="aac",
            trim_start=trim_start,
            trim_end=trim_end,
        )
    _run(command)
    if not output.is_file():
        raise MediaToolError(t("video.ffmpeg_failed", detail="output missing"))
    return mode


def read_first_frame(path: Path):
    """Decode the first video frame as BGR. Used by the UI preview."""

    with tempfile.TemporaryDirectory(prefix="barrett-faceswap-") as directory:
        dest = Path(directory) / "frame.png"
        command = [
            _ffmpeg_bin(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            str(dest),
        ]
        _run(command)
        return read_image(dest)


def _ffmpeg_bin() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise MediaToolError(t("video.ffmpeg_missing"))
    return ffmpeg


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise MediaToolError(t("video.ffmpeg_failed", detail=exc)) from exc
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        tail = detail[-500:] if detail else str(completed.returncode)
        raise MediaToolError(t("video.ffmpeg_failed", detail=tail))
    return completed


def _ratio(text: str) -> float:
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        try:
            den = float(denominator)
            if den == 0:
                return 0.0
            return float(numerator) / den
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _float(value: object) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _seconds(value: float) -> str:
    return f"{float(value):.3f}"
