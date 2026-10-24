"""Turn a source video into one identity and an optional speech reference.

A source video is sampled across its length. Each frame keeps its best face.
Those faces are embedded, then a held-out check decides whether the average
of the top faces matches the remaining faces better than the single best
face. The winner is the identity the swapper uses.

Speech for the voice step is the source video's own track with silence and
noise-like stretches removed. A target that has no audio never reaches this
step. An uploaded reference clip replaces the extracted one.
"""

from __future__ import annotations

import shutil
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from barrett_faceswap.config import AppConfig
from barrett_faceswap.detect import face_area
from barrett_faceswap.errors import NoFaceError
from barrett_faceswap.i18n import t
from barrett_faceswap.media.ffmpeg import (
    extract_audio,
    probe,
    sample_video_frames,
    save_middle_frame,
)
from barrett_faceswap.swap_catalog import swapper_by_name
from barrett_faceswap.types import Face, SwapRequest
from barrett_faceswap.vision import is_video, read_image
from barrett_faceswap.voice import target_has_audio, voice_is_enabled

SAMPLE_COUNT = 12
TOP_COUNT = 8
MIN_COSINE = 0.45
SPEECH_RATE = 16000

DetectFaces = Callable[[np.ndarray], list[Face]]
EmbedFace = Callable[[np.ndarray, Face], np.ndarray]


@dataclass(frozen=True)
class IdentityChoice:
    """The vector the swapper should use, and how it was chosen."""

    vector: np.ndarray
    mode: str
    count: int
    mean_score: float
    best_score: float


def source_mode_visibility(mode: str | None) -> dict[str, bool]:
    """Which source uploads belong to the selected mode.

    ``voice_override`` is the optional clip on the voice panel. It is shown
    for a source video, where the extracted speech is the default reference.
    """

    selected = mode if mode in {"video", "photo", "photo_audio"} else "video"
    return {
        "video": selected == "video",
        "photo": selected in {"photo", "photo_audio"},
        "audio": selected == "photo_audio",
        "voice_override": selected == "video",
    }


def embedding_array(request: SwapRequest) -> np.ndarray | None:
    """The prepared identity, or None when the swapper should embed the still."""

    if not request.source_embedding:
        return None
    return np.asarray(request.source_embedding, dtype=np.float32)


def select_identity(
    vectors: list[np.ndarray],
    qualities: list[float],
    *,
    min_cosine: float = MIN_COSINE,
) -> IdentityChoice:
    """Pick the average when it beats the best face on held-out faces.

    Faces whose cosine to the best face is below ``min_cosine`` are dropped
    first, so a second person does not pull the average. The lowest-quality
    third of what remains is the holdout. The average of the other faces is
    used only when that holdout prefers it. Otherwise the best face wins.
    One face, or a template pool of one, is the best face.
    """

    if not vectors or len(vectors) != len(qualities):
        raise NoFaceError(t("identity.no_face"))
    units = [_unit(vector) for vector in vectors]
    order = sorted(range(len(units)), key=lambda index: float(qualities[index]), reverse=True)
    best = units[order[0]]
    kept = [index for index in order if float(np.dot(units[index], best)) >= float(min_cosine)]
    if len(kept) < 3:
        return IdentityChoice(best.astype(np.float32), "single" if len(kept) == 1 else "best", len(kept), 0.0, 0.0)
    probe_count = max(1, len(kept) // 3)
    probes = kept[-probe_count:]
    pool = kept[:-probe_count]
    if len(pool) < 2:
        return IdentityChoice(best.astype(np.float32), "best", len(kept), 0.0, 0.0)
    best_pool = units[pool[0]]
    mean_pool = _unit(np.mean([units[index] for index in pool], axis=0))
    probe_vectors = [units[index] for index in probes]
    score_best = float(np.mean([np.dot(best_pool, probe) for probe in probe_vectors]))
    score_mean = float(np.mean([np.dot(mean_pool, probe) for probe in probe_vectors]))
    if score_mean > score_best + 1e-4:
        chosen = _unit(np.mean([units[index] for index in kept], axis=0))
        return IdentityChoice(chosen.astype(np.float32), "mean", len(kept), score_mean, score_best)
    return IdentityChoice(best.astype(np.float32), "best", len(kept), score_mean, score_best)


def face_quality(frame: np.ndarray, face: Face) -> float:
    """Higher means a larger, sharper, more frontal face."""

    height, width = int(frame.shape[0]), int(frame.shape[1])
    area_ratio = face_area(face) / max(1.0, float(height * width))
    size = min(1.0, area_ratio / 0.08)
    x1, y1, x2, y2 = [int(round(float(value))) for value in face.bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        sharp = 0.0
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp_norm = min(1.0, sharp / 150.0)
    frontal = _frontal(face, max(1.0, float(x2 - x1)))
    return float(face.score) * (0.45 + 0.55 * size) * (0.45 + 0.55 * sharp_norm) * frontal


def trim_speech(samples: np.ndarray, sample_rate: int, *, min_seconds: float = 0.25) -> np.ndarray | None:
    """Drop silence and noise-like frames. Return speech, or None when too little remains.

    A frame is speech when it is loud enough, harmonically peaked (low spectral
    flatness), and concentrated in the speech band. This is not a trained
    voice-activity model. Music with a strong fundamental can pass.
    """

    audio = np.asarray(samples, dtype=np.float64).reshape(-1)
    rate = int(sample_rate)
    if rate < 1000 or audio.size < rate // 5:
        return None
    frame = max(16, int(0.02 * rate))
    count = audio.size // frame
    if count < 3:
        return None
    rms, flatness, band = _speech_features(audio, rate, frame, count)
    peak = float(np.percentile(rms, 95))
    if peak < 1e-4:
        return None
    keep = (rms >= peak * 0.15) & (flatness <= 0.40) & (band >= 0.40)
    keep = _bridge_gaps(keep)
    keep = _pad_mask(keep, 2)
    keep = _drop_short(keep, minimum=max(3, int(0.10 * rate / frame)))
    if not bool(np.any(keep)):
        return None
    pieces = [audio[index * frame : (index + 1) * frame] for index in range(count) if keep[index]]
    trimmed = np.concatenate(pieces)
    if trimmed.size < int(min_seconds * rate):
        return None
    return trimmed.astype(np.float32)


def prepare_source(
    config: AppConfig,
    request: SwapRequest,
    work: Path,
    *,
    detector: DetectFaces | None = None,
    embedder: EmbedFace | None = None,
) -> tuple[SwapRequest, list[str]]:
    """Replace a source video with a still and, when useful, a speech clip.

    A still source is returned unchanged. Dry-run writes one middle frame and
    does not load a detector. A real run embeds the sampled faces. Speech is
    extracted only when voice is on, the target video has audio, and no
    reference clip was uploaded.
    """

    if not is_video(request.source_path):
        return request, []
    work.mkdir(parents=True, exist_ok=True)
    notes = [t("identity.from_video", name=request.source_path.name)]
    still = work / "identity.png"
    if request.dry_run:
        save_middle_frame(request.source_path, still)
        notes.append(t("identity.dry_frame"))
        updated = request.model_copy(update={"source_path": still})
        return _attach_speech(config, updated, request.source_path, work, notes)
    detect, embed = _resolvers(config, request, detector, embedder)
    frames = sample_video_frames(request.source_path, work / "samples", SAMPLE_COUNT)
    ranked: list[tuple[float, Path, np.ndarray]] = []
    for frame_path in frames:
        image = read_image(frame_path)
        faces = list(detect(image))
        if not faces:
            continue
        face = max(faces, key=lambda item: face_quality(image, item))
        ranked.append((face_quality(image, face), frame_path, np.asarray(embed(image, face), dtype=np.float32)))
    if not ranked:
        raise NoFaceError(t("identity.no_face"))
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen_rows = ranked[:TOP_COUNT]
    choice = select_identity([row[2] for row in chosen_rows], [row[0] for row in chosen_rows])
    shutil.copyfile(chosen_rows[0][1], still)
    spec = swapper_by_name(config.swap.model)
    notes.append(_identity_note(choice, crop_model=spec.embedding == "source_image"))
    update: dict[str, object] = {"source_path": still}
    if spec.embedding != "source_image":
        update["source_embedding"] = tuple(float(value) for value in choice.vector.tolist())
    updated = request.model_copy(update=update)
    return _attach_speech(config, updated, request.source_path, work, notes)


def _attach_speech(
    config: AppConfig,
    request: SwapRequest,
    video: Path,
    work: Path,
    notes: list[str],
) -> tuple[SwapRequest, list[str]]:
    if request.voice_reference is not None or not voice_is_enabled(config, request):
        return request, notes
    if not is_video(request.target_path) or not target_has_audio(request.target_path):
        return request, notes
    dest = work / "source-speech.wav"
    if not write_speech_reference(video, dest, work):
        notes.append(t("identity.no_speech"))
        return request, notes
    notes.append(t("identity.speech"))
    return request.model_copy(update={"voice_reference": dest}), notes


def reference_for_mode(mode: str, mode_audio: Path | None, override: Path | None) -> Path | None:
    """Uploaded clip for this source mode.

    Video mode returns the override, or None so the caller can trim speech
    from the source video. Photo never has a reference. Photo + audio uses
    the clip next to the photo.
    """

    if mode == "photo":
        return None
    if mode == "photo_audio":
        return mode_audio
    return override


def write_speech_reference(video: Path, dest: Path, work: Path) -> bool:
    """Trim silence and noise from ``video`` into ``dest``. False when nothing usable remains."""

    if not is_video(video) or not probe(video).has_audio:
        return False
    raw = work / "source-audio.wav"
    extract_audio(video, raw, trim_start=None, trim_end=None, sample_rate=SPEECH_RATE)
    samples, rate = _read_wav(raw)
    trimmed = trim_speech(samples, rate)
    if trimmed is None:
        return False
    _write_wav(dest, trimmed, rate)
    return True


def resolve_live_reference(
    mode: str,
    source: Path | None,
    override: Path | None,
    mode_audio: Path | None,
    work: Path,
) -> tuple[Path | None, str]:
    """Reference for a live session: uploaded clip, else speech from the source video.

    ``upload`` is an override or the photo + audio clip. ``source`` is speech
    trimmed from a source video. ``none`` means the live run stays face-only.
    """

    work.mkdir(parents=True, exist_ok=True)
    uploaded = reference_for_mode(mode, mode_audio, override)
    if uploaded is not None:
        return (uploaded, "upload") if uploaded.is_file() else (None, "none")
    if mode == "video" and source is not None and is_video(source):
        dest = work / "live-speech.wav"
        if write_speech_reference(source, dest, work):
            return dest, "source"
    return None, "none"


def _identity_note(choice: IdentityChoice, *, crop_model: bool) -> str:
    if crop_model:
        return t("identity.crop")
    if choice.mode == "mean":
        return t(
            "identity.mean",
            count=choice.count,
            mean_score=f"{choice.mean_score:.3f}",
            best_score=f"{choice.best_score:.3f}",
        )
    if choice.mode == "single":
        return t("identity.single")
    return t(
        "identity.best",
        count=choice.count,
        mean_score=f"{choice.mean_score:.3f}",
        best_score=f"{choice.best_score:.3f}",
    )


def _resolvers(
    config: AppConfig,
    request: SwapRequest,
    detector: DetectFaces | None,
    embedder: EmbedFace | None,
) -> tuple[DetectFaces, EmbedFace]:
    if detector is not None and embedder is not None:
        return detector, embedder
    from barrett_faceswap.detect import FaceDetector
    from barrett_faceswap.models import ModelStore
    from barrett_faceswap.swap import embed_arcface

    providers = list(request.execution_providers) if request.execution_providers else list(config.execution.providers)
    store = ModelStore(config, providers)
    store.ensure_present(("detector", "recognizer"))
    face_detector = FaceDetector(
        store.session("detector"),
        det_size=config.detect.det_size,
        score_threshold=config.detect.score_threshold,
        nms_threshold=config.detect.nms_threshold,
    )

    def detect(image: np.ndarray) -> list[Face]:
        return face_detector.try_detect(image)

    def embed(image: np.ndarray, face: Face) -> np.ndarray:
        return embed_arcface(store, image, face.landmarks)

    return detect, embed


def _unit(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise NoFaceError(t("identity.no_face"))
    return array / norm


def _frontal(face: Face, box_width: float) -> float:
    points = np.asarray(face.landmarks, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 3 or box_width <= 0:
        return 1.0
    left, right, nose = points[0], points[1], points[2]
    eye = float(np.linalg.norm(right - left))
    span = min(1.0, eye / box_width / 0.35)
    middle = (float(left[0]) + float(right[0])) / 2.0
    center = 1.0 - min(1.0, abs(float(nose[0]) - middle) / box_width / 0.25)
    return max(0.15, 0.5 * span + 0.5 * center)


def _speech_features(
    audio: np.ndarray,
    sample_rate: int,
    frame: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rms = np.empty(count, dtype=np.float64)
    flatness = np.empty(count, dtype=np.float64)
    band_ratio = np.empty(count, dtype=np.float64)
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, d=1.0 / float(sample_rate))
    speech_band = (freqs >= 80.0) & (freqs <= 4000.0)
    for index in range(count):
        chunk = audio[index * frame : (index + 1) * frame]
        rms[index] = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
        spectrum = np.abs(np.fft.rfft(chunk * window)) + 1e-12
        flatness[index] = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
        band_ratio[index] = float(np.sum(spectrum[speech_band]) / np.sum(spectrum))
    return rms, flatness, band_ratio


def _bridge_gaps(mask: np.ndarray) -> np.ndarray:
    bridged = mask.copy()
    for index in range(1, len(bridged) - 1):
        if not bridged[index] and bridged[index - 1] and bridged[index + 1]:
            bridged[index] = True
    return bridged


def _pad_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    padded = mask.copy()
    hits = np.flatnonzero(mask)
    for index in hits:
        start = max(0, int(index) - radius)
        stop = min(len(padded), int(index) + radius + 1)
        padded[start:stop] = True
    return padded


def _drop_short(mask: np.ndarray, minimum: int) -> np.ndarray:
    result = mask.copy()
    start: int | None = None
    flags = list(result) + [False]
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start < minimum:
                result[start:index] = False
            start = None
    return result


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise NoFaceError(t("identity.no_speech"))
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, int(rate)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
