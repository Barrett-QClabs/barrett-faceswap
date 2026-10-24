"""Named source profiles stored on disk.

A profile collects photos, videos, and audio of one person. Each new sample
is fused into the identity: face embeddings go through the same held-out
average used for a source video, and trimmed speech is appended to the
reference clip. A swap can use a profile or a one-shot upload.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import BarrettFaceswapError, UnsupportedMediaError
from barrett_faceswap.i18n import t
from barrett_faceswap.media.ffmpeg import extract_audio
from barrett_faceswap.sourceprep import (
    SPEECH_RATE,
    _read_wav,
    _write_wav,
    embed_still,
    prepare_source,
    select_identity,
    trim_speech,
    write_speech_reference,
)
from barrett_faceswap.types import SwapRequest
from barrett_faceswap.vision import IMAGE_SUFFIXES, is_video, read_image, write_image

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


@dataclass
class SampleParts:
    """One contribution before it is written into the profile."""

    kind: str
    embedding: np.ndarray | None
    quality: float
    still: np.ndarray | None
    speech: np.ndarray | None
    speech_rate: int = SPEECH_RATE
    filename: str = ""


@dataclass
class ProfileSummary:
    id: str
    name: str
    samples: int
    has_voice: bool
    has_face: bool
    fusion: str


@dataclass
class ResolvedProfile:
    """Files and the fused embedding a swap can use immediately."""

    id: str
    name: str
    still: Path
    embedding: np.ndarray
    voice: Path | None
    samples: int
    fusion: str


class FaceLibrary:
    """One directory of profiles. Each profile is a folder of JSON and media."""

    def __init__(self, root: Path, *, voice_seconds: float = 24.0) -> None:
        self.root = Path(root)
        self.voice_seconds = float(voice_seconds)

    def list_profiles(self) -> list[ProfileSummary]:
        if not self.root.is_dir():
            return []
        found: list[ProfileSummary] = []
        for path in sorted(self.root.iterdir()):
            record = path / "profile.json"
            if record.is_file():
                found.append(_summary(_read_json(record), path))
        return sorted(found, key=lambda item: item.name.casefold())

    def create(self, name: str) -> ProfileSummary:
        clean = _clean_name(name)
        if self._find_dir(clean) is not None:
            raise BarrettFaceswapError(t("library.exists", name=clean))
        profile_id = _fresh_id(self.root, clean)
        directory = self.root / profile_id
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "samples").mkdir()
        record = {
            "id": profile_id,
            "name": clean,
            "created": _now(),
            "updated": _now(),
            "fusion": "none",
            "embedding": None,
            "samples": [],
        }
        _write_json(directory / "profile.json", record)
        return _summary(record, directory)

    def delete(self, key: str) -> None:
        directory = self._require_dir(key)
        import shutil

        shutil.rmtree(directory)

    def add_sample(self, key: str, parts: SampleParts) -> ProfileSummary:
        if parts.embedding is None and parts.speech is None:
            raise BarrettFaceswapError(t("library.empty_sample"))
        directory = self._require_dir(key)
        record = _read_json(directory / "profile.json")
        sample_id = uuid.uuid4().hex[:8]
        payload: dict[str, object] = {
            "id": sample_id,
            "kind": parts.kind,
            "filename": parts.filename or sample_id,
            "added": _now(),
            "quality": float(parts.quality),
            "embedding": None if parts.embedding is None else [float(value) for value in np.asarray(parts.embedding).reshape(-1)],
        }
        sample_dir = directory / "samples"
        sample_dir.mkdir(exist_ok=True)
        if parts.still is not None:
            write_image(sample_dir / f"{sample_id}.png", parts.still)
        if parts.speech is not None:
            _write_wav(sample_dir / f"{sample_id}.wav", parts.speech, int(parts.speech_rate))
        _write_json(sample_dir / f"{sample_id}.json", payload)
        samples = list(record.get("samples") or [])
        samples.append(sample_id)
        record["samples"] = samples
        self._fuse(directory, record)
        return _summary(record, directory)

    def remove_sample(self, key: str, sample_id: str) -> ProfileSummary:
        directory = self._require_dir(key)
        record = _read_json(directory / "profile.json")
        samples = [item for item in list(record.get("samples") or []) if item != sample_id]
        if len(samples) == len(list(record.get("samples") or [])):
            raise BarrettFaceswapError(t("library.missing_sample", sample=sample_id))
        record["samples"] = samples
        folder = directory / "samples"
        for path in folder.glob(f"{sample_id}.*"):
            path.unlink(missing_ok=True)
        self._fuse(directory, record)
        return _summary(record, directory)

    def resolve(self, key: str) -> ResolvedProfile:
        directory = self._require_dir(key)
        record = _read_json(directory / "profile.json")
        embedding = record.get("embedding")
        still = directory / "still.png"
        if not embedding or not still.is_file():
            raise BarrettFaceswapError(t("library.no_face", name=record.get("name", key)))
        voice = directory / "voice.wav"
        return ResolvedProfile(
            id=str(record["id"]),
            name=str(record["name"]),
            still=still,
            embedding=np.asarray(embedding, dtype=np.float32),
            voice=voice if voice.is_file() else None,
            samples=len(list(record.get("samples") or [])),
            fusion=str(record.get("fusion") or "none"),
        )

    def sample_choices(self, key: str) -> list[tuple[str, str]]:
        directory = self._find_dir(key)
        if directory is None:
            return []
        record = _read_json(directory / "profile.json")
        choices: list[tuple[str, str]] = []
        for sample_id in list(record.get("samples") or []):
            payload = _read_json(directory / "samples" / f"{sample_id}.json")
            label = f"{payload.get('kind', '')} · {payload.get('filename', sample_id)}"
            choices.append((label, sample_id))
        return choices

    def _fuse(self, directory: Path, record: dict) -> None:
        vectors: list[np.ndarray] = []
        qualities: list[float] = []
        best_still: tuple[float, Path] | None = None
        speeches: list[np.ndarray] = []
        for sample_id in list(record.get("samples") or []):
            payload = _read_json(directory / "samples" / f"{sample_id}.json")
            quality = float(payload.get("quality") or 0.0)
            raw = payload.get("embedding")
            if raw:
                vectors.append(np.asarray(raw, dtype=np.float32))
                qualities.append(quality)
            still = directory / "samples" / f"{sample_id}.png"
            if still.is_file() and (best_still is None or quality >= best_still[0]):
                best_still = (quality, still)
            speech = directory / "samples" / f"{sample_id}.wav"
            if speech.is_file():
                audio, _rate = _read_wav(speech)
                speeches.append(audio)
        if vectors:
            choice = select_identity(vectors, qualities)
            record["embedding"] = [float(value) for value in choice.vector.reshape(-1)]
            record["fusion"] = choice.mode
        else:
            record["embedding"] = None
            record["fusion"] = "none"
        preview = directory / "still.png"
        if best_still is None:
            preview.unlink(missing_ok=True)
        else:
            preview.write_bytes(best_still[1].read_bytes())
        voice = directory / "voice.wav"
        fused = _fuse_speech(speeches, self.voice_seconds)
        if fused is None:
            voice.unlink(missing_ok=True)
        else:
            _write_wav(voice, fused, SPEECH_RATE)
        record["updated"] = _now()
        _write_json(directory / "profile.json", record)

    def _find_dir(self, key: str) -> Path | None:
        text = str(key or "").strip()
        if not text or not self.root.is_dir():
            return None
        direct = self.root / text
        if (direct / "profile.json").is_file():
            return direct
        folded = text.casefold()
        for path in self.root.iterdir():
            record = path / "profile.json"
            if not record.is_file():
                continue
            payload = _read_json(record)
            if str(payload.get("id", "")).casefold() == folded or str(payload.get("name", "")).casefold() == folded:
                return path
        return None

    def _require_dir(self, key: str) -> Path:
        found = self._find_dir(key)
        if found is None:
            raise BarrettFaceswapError(t("library.missing", name=key))
        return found


def library_from_config(config: AppConfig) -> FaceLibrary:
    return FaceLibrary(config.library.dir, voice_seconds=float(config.library.voice_seconds))


def parts_from_file(config: AppConfig, source: Path) -> SampleParts:
    """Read a video, photo, or audio file into one sample.

    A video contributes a fused identity and trimmed speech. A photo contributes
    one face. Audio contributes speech only.
    """

    path = Path(source)
    if not path.is_file():
        raise BarrettFaceswapError(t("vision.missing", path=path))
    if is_video(path):
        return _parts_from_video(config, path)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return _parts_from_photo(config, path)
    if path.suffix.lower() in AUDIO_SUFFIXES:
        return _parts_from_audio(path)
    allowed = ", ".join(sorted(IMAGE_SUFFIXES | AUDIO_SUFFIXES | {".mp4", ".mov", ".mkv", ".webm"}))
    raise UnsupportedMediaError(t("vision.suffix", suffix=path.suffix, allowed=allowed))


def _parts_from_video(config: AppConfig, source: Path) -> SampleParts:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="barrett-library-") as directory:
        work = Path(directory)
        request = SwapRequest(
            source_path=source,
            target_path=source,
            output_path=work / "unused.png",
            voice_enabled=False,
        )
        updated, _notes = prepare_source(config, request, work)
        if not updated.source_embedding:
            raise BarrettFaceswapError(t("library.no_face", name=source.name))
        speech = None
        speech_path = work / "speech.wav"
        if write_speech_reference(source, speech_path, work):
            speech, _rate = _read_wav(speech_path)
        return SampleParts(
            kind="video",
            embedding=np.asarray(updated.source_embedding, dtype=np.float32),
            quality=1.0,
            still=read_image(updated.source_path),
            speech=speech,
            filename=source.name,
        )


def _parts_from_photo(config: AppConfig, source: Path) -> SampleParts:
    vector, quality, still = embed_still(config, read_image(source))
    return SampleParts(
        kind="photo",
        embedding=vector,
        quality=quality,
        still=still,
        speech=None,
        filename=source.name,
    )


def _parts_from_audio(source: Path) -> SampleParts:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="barrett-library-audio-") as directory:
        raw = Path(directory) / "audio.wav"
        extract_audio(source, raw, trim_start=None, trim_end=None, sample_rate=SPEECH_RATE)
        samples, rate = _read_wav(raw)
    trimmed = trim_speech(samples, rate)
    if trimmed is None:
        raise BarrettFaceswapError(t("library.no_speech"))
    return SampleParts(
        kind="audio",
        embedding=None,
        quality=0.0,
        still=None,
        speech=trimmed,
        speech_rate=rate,
        filename=source.name,
    )


def _fuse_speech(pieces: list[np.ndarray], max_seconds: float) -> np.ndarray | None:
    if not pieces:
        return None
    audio = np.concatenate([np.asarray(piece, dtype=np.float32).reshape(-1) for piece in pieces])
    cap = max(1, int(float(max_seconds) * SPEECH_RATE))
    if audio.size > cap:
        audio = audio[-cap:]
    return audio


def _summary(record: dict, directory: Path) -> ProfileSummary:
    return ProfileSummary(
        id=str(record.get("id", "")),
        name=str(record.get("name", "")),
        samples=len(list(record.get("samples") or [])),
        has_voice=(directory / "voice.wav").is_file(),
        has_face=bool(record.get("embedding")),
        fusion=str(record.get("fusion") or "none"),
    )


def _clean_name(name: str) -> str:
    clean = " ".join(str(name or "").split())
    if not clean:
        raise BarrettFaceswapError(t("library.need_name"))
    return clean[:80]


def _fresh_id(root: Path, name: str) -> str:
    base = re.sub(r"[^\w\-]+", "-", name.casefold(), flags=re.UNICODE).strip("-") or "profile"
    base = base[:40]
    candidate = base
    index = 2
    while (root / candidate).exists():
        candidate = f"{base}-{index}"
        index += 1
    return candidate


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BarrettFaceswapError(t("library.corrupt", path=path)) from exc
    if not isinstance(payload, dict):
        raise BarrettFaceswapError(t("library.corrupt", path=path))
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)

