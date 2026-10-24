"""Probe, extract, and encode video without copying another project's ffmpeg builder.

Disk mode writes PNG frames so the compliance gate can sample them. Pipe mode
decodes raw BGR frames on stdout and encodes raw BGR on stdin. Disk stays the
default.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    # Output option: relocate the moov atom so browsers can play before the download finishes.
    command += ["-movflags", "+faststart", str(output)]
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


def media_duration(path: Path) -> float:
    """Container duration in seconds. Works for audio files as well as video."""

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
            "-show_format",
            str(path),
        ]
    )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaToolError(t("video.probe_parse", path=path)) from exc
    duration = _float((payload.get("format") or {}).get("duration"))
    if duration <= 0:
        raise MediaToolError(t("video.bad_duration", path=path))
    return duration


def extract_audio(
    source: Path,
    dest: Path,
    *,
    trim_start: float | None,
    trim_end: float | None,
    sample_rate: int | None = None,
) -> None:
    """Write the first audio stream as mono 16-bit PCM WAV."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [_ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error"]
    if trim_start is not None:
        command += ["-ss", _seconds(trim_start)]
    if trim_end is not None:
        command += ["-to", _seconds(trim_end)]
    command += ["-i", str(source), "-map", "0:a:0", "-ac", "1"]
    if sample_rate is not None:
        command += ["-ar", str(int(sample_rate))]
    command += ["-c:a", "pcm_s16le", str(dest)]
    _run(command)
    if not dest.is_file():
        raise MediaToolError(t("video.ffmpeg_failed", detail="audio missing"))


def sample_video_frames(source: Path, directory: Path, count: int) -> list[Path]:
    """Save up to ``count`` frames spread across the clip.

    The fps filter keeps a long video from writing every frame. The list is
    sorted by filename and is shorter when the clip itself is shorter.
    """

    if int(count) < 1:
        raise MediaToolError(t("video.no_frames", path=source))
    directory.mkdir(parents=True, exist_ok=True)
    duration = media_duration(source)
    rate = float(count) / max(duration, 1e-3)
    pattern = directory / "sample_%03d.png"
    _run(
        [
            _ffmpeg_bin(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            f"fps={rate:.6f}",
            "-frames:v",
            str(int(count)),
            str(pattern),
        ]
    )
    frames = sorted(directory.glob("sample_*.png"))
    if not frames:
        raise MediaToolError(t("video.no_frames", path=source))
    return frames


def save_middle_frame(source: Path, dest: Path) -> None:
    """Write one frame from the middle of ``source``. Dry-run uses this still."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = media_duration(source)
    _run(
        [
            _ffmpeg_bin(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-ss",
            _seconds(max(0.0, duration / 2.0)),
            "-frames:v",
            "1",
            str(dest),
        ]
    )
    if not dest.is_file():
        raise MediaToolError(t("video.no_frames", path=source))


def render_wav(source: Path, dest: Path, *, audio_filter: str) -> None:
    """Run one audio filter and write mono PCM WAV."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _ffmpeg_bin(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-af",
        audio_filter,
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    _run(command)
    if not dest.is_file():
        raise MediaToolError(t("video.ffmpeg_failed", detail="audio missing"))


def mix_wavs(
    original: Path,
    converted: Path,
    dest: Path,
    *,
    original_weight: float,
    converted_weight: float,
) -> None:
    """Mix two mono WAV files. Weights should already sum to about 1."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Scale each clip, then mix at equal gain. A space inside amix weights is easy to split wrong.
    command = [
        _ffmpeg_bin(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(original),
        "-i",
        str(converted),
        "-filter_complex",
        (
            f"[0:a]volume={float(original_weight):.6f}[a0];"
            f"[1:a]volume={float(converted_weight):.6f}[a1];"
            "[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0"
        ),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    _run(command)
    if not dest.is_file():
        raise MediaToolError(t("video.ffmpeg_failed", detail="audio missing"))


def build_replace_audio_command(video: Path, audio: Path, output: Path) -> list[str]:
    """ffmpeg argv that copies the video stream and muxes a new AAC track.

    The audio clip is fitted to the picture before this command runs, so the
    video stream is not trimmed with ``-shortest``. ``+faststart`` matches the
    frame encoder.
    """

    return [
        _ffmpeg_bin(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output),
    ]


def replace_video_audio(video: Path, audio: Path) -> None:
    """Replace ``video``'s audio in place and keep the picture bitstream."""

    if not video.is_file():
        raise MediaToolError(t("vision.missing", path=video))
    temp = video.with_name(f"{video.stem}.voice-tmp{video.suffix}")
    try:
        _run(build_replace_audio_command(video, audio, temp))
        if not temp.is_file():
            raise MediaToolError(t("video.ffmpeg_failed", detail="output missing"))
        temp.replace(video)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def even_size(width: int, height: int) -> tuple[int, int]:
    """yuv420p wants even dimensions."""

    return max(2, int(width) - int(width) % 2), max(2, int(height) - int(height) % 2)


def build_decode_pipe_command(
    source: Path,
    *,
    width: int,
    height: int,
    trim_start: float | None,
    trim_end: float | None,
    seek: float | None,
) -> list[str]:
    """ffmpeg argv that writes raw BGR frames to stdout."""

    command = [_ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-thread_queue_size", "32"]
    start = seek
    if trim_start is not None and seek:
        start = float(trim_start) + float(seek)
    elif trim_start is not None:
        start = float(trim_start)
    if start:
        command += ["-ss", _seconds(start)]
    if trim_end is not None:
        command += ["-to", _seconds(trim_end)]
    command += [
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-vf",
        f"scale={int(width)}:{int(height)}",
        "pipe:1",
    ]
    return command


def build_encode_pipe_command(
    output: Path,
    *,
    width: int,
    height: int,
    fps: float,
    encoder: str,
    crf: int,
    audio_source: Path | None,
    audio_mode: str,
    trim_start: float | None,
    trim_end: float | None,
) -> list[str]:
    """ffmpeg argv that reads raw BGR frames from stdin."""

    if encoder not in {"libx264", "h264_nvenc"}:
        raise MediaToolError(t("config.encoder", allowed="libx264, h264_nvenc"))
    command = [
        _ffmpeg_bin(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{int(width)}x{int(height)}",
        "-r",
        f"{float(fps):.6f}",
        "-i",
        "pipe:0",
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
    command += ["-pix_fmt", "yuv420p"]
    if audio_mode == "copy":
        command += ["-c:a", "copy"]
    elif audio_mode == "aac":
        command += ["-c:a", "aac", "-b:a", "192k"]
    if audio_mode != "none":
        command += ["-shortest"]
    command += ["-movflags", "+faststart", str(output)]
    return command


def iter_raw_frames(
    source: Path,
    *,
    width: int,
    height: int,
    trim_start: float | None,
    trim_end: float | None,
    seek: float | None,
):
    """Yield BGR frames from ffmpeg stdout. Does not encode."""

    process = subprocess.Popen(
        build_decode_pipe_command(
            source,
            width=width,
            height=height,
            trim_start=trim_start,
            trim_end=trim_end,
            seek=seek,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    need = int(width) * int(height) * 3
    try:
        while True:
            buffer = bytearray()
            while len(buffer) < need:
                chunk = process.stdout.read(need - len(buffer))
                if not chunk:
                    if buffer:
                        raise MediaToolError(t("video.pipe_short"))
                    return
                buffer.extend(chunk)
            frame = np.frombuffer(buffer, dtype=np.uint8).reshape(int(height), int(width), 3)
            yield frame.copy()
    finally:
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass


class RawVideoPipe:
    """Decode raw frames and encode raw frames without writing a PNG per frame."""

    def __init__(
        self,
        source: Path,
        output: Path,
        *,
        width: int,
        height: int,
        fps: float,
        encoder: str,
        crf: int,
        audio_source: Path | None,
        audio_mode: str,
        trim_start: float | None,
        trim_end: float | None,
        seek: float | None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self._errors: list[str] = []
        output.parent.mkdir(parents=True, exist_ok=True)
        self.decode = subprocess.Popen(
            build_decode_pipe_command(
                source,
                width=self.width,
                height=self.height,
                trim_start=trim_start,
                trim_end=trim_end,
                seek=seek,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.encode = subprocess.Popen(
            build_encode_pipe_command(
                output,
                width=self.width,
                height=self.height,
                fps=fps,
                encoder=encoder,
                crf=crf,
                audio_source=audio_source,
                audio_mode=audio_mode,
                trim_start=trim_start,
                trim_end=trim_end,
            ),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._threads = [
            threading.Thread(target=self._drain, args=(self.decode.stderr,), daemon=True),
            threading.Thread(target=self._drain, args=(self.encode.stderr,), daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def read(self) -> np.ndarray | None:
        assert self.decode.stdout is not None
        need = self.width * self.height * 3
        buffer = bytearray()
        while len(buffer) < need:
            chunk = self.decode.stdout.read(need - len(buffer))
            if not chunk:
                if not buffer:
                    return None
                raise MediaToolError(t("video.pipe_short"))
            buffer.extend(chunk)
        frame = np.frombuffer(buffer, dtype=np.uint8).reshape(self.height, self.width, 3)
        return frame.copy()

    def write(self, frame: np.ndarray) -> None:
        assert self.encode.stdin is not None
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            import cv2

            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        self.encode.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.encode.stdin is not None:
            self.encode.stdin.close()
        decode_code = self.decode.wait(timeout=120)
        encode_code = self.encode.wait(timeout=120)
        for thread in self._threads:
            thread.join(timeout=5)
        if decode_code != 0 or encode_code != 0:
            detail = " ".join(self._errors)[-500:] or f"decode={decode_code} encode={encode_code}"
            raise MediaToolError(t("video.ffmpeg_failed", detail=detail))

    def abort(self) -> None:
        for process in (self.decode, self.encode):
            if process.poll() is None:
                process.kill()
        for process in (self.decode, self.encode):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def _drain(self, pipe) -> None:
        if pipe is None:
            return
        data = pipe.read()
        if data:
            self._errors.append(data.decode("utf-8", errors="replace"))


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
