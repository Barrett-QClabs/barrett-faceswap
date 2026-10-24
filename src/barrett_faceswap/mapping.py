"""Assign a different source to each face in one export.

The image and video page scans one frame, shows each face, and writes a
temporary face-map JSON. A row names the source still, the fused identity
when it came from a library profile, and the target person's match vector.
A face left as keep is omitted, so the swapper leaves it unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import BarrettFaceswapError, NoFaceError
from barrett_faceswap.i18n import t
from barrett_faceswap.library import FaceLibrary
from barrett_faceswap.types import Face
from barrett_faceswap.vision import write_image

KEEP = "keep"
UPLOAD = "upload"
MAP_SLOTS = 4


@dataclass
class DetectedFace:
    """One person found in the scanned frame."""

    match: tuple[float, ...]
    reference: str
    thumb: np.ndarray
    bbox: tuple[float, float, float, float]


@dataclass
class ScanResult:
    """Faces shown in the mapper, plus how many the frame actually held."""

    faces: list[DetectedFace]
    total: int


@dataclass
class Assignment:
    """One face-map row ready to write."""

    source: str
    match: tuple[float, ...]
    embedding: tuple[float, ...] | None = None
    reference: str | None = None
    voice: str | None = None


def detect_target_faces(
    config: AppConfig,
    image: np.ndarray,
    directory: Path,
    *,
    limit: int = MAP_SLOTS,
    detector=None,
    embedder=None,
) -> ScanResult:
    """Return the largest faces in ``image``, each with a thumb and a match vector."""

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    if detector is None or embedder is None:
        detector, embedder = _resolvers(config)
    found = list(detector(image))
    ranked = sorted(found, key=_area, reverse=True)
    chosen = ranked[: max(0, int(limit))]
    faces: list[DetectedFace] = []
    for index, face in enumerate(chosen):
        vector = _unit(embedder(image, face))
        thumb = _thumb(image, face.bbox)
        path = folder / f"face-{index + 1}.png"
        write_image(path, thumb)
        faces.append(
            DetectedFace(
                match=tuple(float(value) for value in vector.reshape(-1)),
                reference=str(path),
                thumb=thumb,
                bbox=tuple(float(value) for value in np.asarray(face.bbox).reshape(-1)[:4]),
            )
        )
    return ScanResult(faces=faces, total=len(found))


def mapping_is_complete(picks: object, files: object) -> bool:
    """True when at least one face has a source and every upload has a file."""

    chosen = list(picks or [])
    uploads = list(files or [])
    usable = False
    for index, pick in enumerate(chosen):
        choice = str(pick or "").strip()
        if choice in {"", KEEP}:
            continue
        if choice == UPLOAD:
            if _file_path(uploads[index] if index < len(uploads) else None) is None:
                return False
            usable = True
            continue
        usable = True
    return usable


def compile_assignments(
    state: object,
    picks: object,
    uploads: object,
    library: FaceLibrary,
) -> list[Assignment]:
    """Turn the scanned faces and the pickers into face-map rows.

    Keep is skipped. Upload uses the chosen photo and no saved embedding.
    Any other value is a library profile id or name.
    """

    items = list(state or [])
    chosen = list(picks or [])
    files = list(uploads or [])
    rows: list[Assignment] = []
    for index, item in enumerate(items):
        choice = str(chosen[index] if index < len(chosen) else KEEP).strip() or KEEP
        if choice == KEEP:
            continue
        match = _match_of(item)
        reference = str(item.get("reference") or "") or None if isinstance(item, dict) else None
        if choice == UPLOAD:
            path = _file_path(files[index] if index < len(files) else None)
            if path is None:
                raise BarrettFaceswapError(t("ui.map_need_file"))
            rows.append(Assignment(source=path, match=match, reference=reference))
            continue
        resolved = library.resolve(choice)
        vector = tuple(float(value) for value in np.asarray(resolved.embedding, dtype=np.float32).reshape(-1))
        rows.append(
            Assignment(
                source=str(resolved.still),
                match=match,
                embedding=vector,
                reference=reference,
                voice=None if resolved.voice is None else str(resolved.voice),
            )
        )
    return rows


def write_face_map(path: Path, rows: list[Assignment]) -> None:
    """Write rows that :func:`barrett_faceswap.facesel.load_face_map` can read."""

    payload: list[dict[str, object]] = []
    for row in rows:
        item: dict[str, object] = {
            "source": row.source,
            "match": [float(value) for value in row.match],
        }
        if row.embedding is not None:
            item["embedding"] = [float(value) for value in row.embedding]
        if row.reference:
            item["reference"] = row.reference
        payload.append(item)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload), encoding="utf-8")


def _resolvers(config: AppConfig):
    from barrett_faceswap.detect import FaceDetector
    from barrett_faceswap.models import ModelStore
    from barrett_faceswap.swap import embed_arcface

    store = ModelStore(config, list(config.execution.providers))
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


def _area(face: Face) -> float:
    x1, y1, x2, y2 = [float(value) for value in np.asarray(face.bbox).reshape(-1)[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _thumb(image: np.ndarray, bbox: np.ndarray, pad: float = 0.35) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in np.asarray(bbox).reshape(-1)[:4]]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    left = max(0, int(round(x1 - pad * box_w)))
    top = max(0, int(round(y1 - pad * box_h)))
    right = min(width, int(round(x2 + pad * box_w)))
    bottom = min(height, int(round(y2 + pad * box_h)))
    if right <= left or bottom <= top:
        return image[:1, :1].copy()
    return np.ascontiguousarray(image[top:bottom, left:right])


def _unit(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise NoFaceError(t("identity.no_face"))
    return array / norm


def _match_of(item: object) -> tuple[float, ...]:
    if not isinstance(item, dict):
        raise BarrettFaceswapError(t("ui.map_need_target"))
    raw = item.get("match")
    if not isinstance(raw, list) or not raw:
        raise BarrettFaceswapError(t("ui.map_need_target"))
    try:
        return tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise BarrettFaceswapError(t("ui.map_need_target")) from exc


def _file_path(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        text = str(value).strip()
        return text or None
    if isinstance(value, (list, tuple)):
        return _file_path(value[0]) if value else None
    if isinstance(value, dict):
        return _file_path(value.get("path") or value.get("name"))
    text = str(value).strip()
    return text or None
