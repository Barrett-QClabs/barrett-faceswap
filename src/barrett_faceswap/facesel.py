"""Choose which source identity replaces which target face.

Index mode is handled by the caller. This module covers many-faces,
reference similarity, explicit source-to-target maps, and a small IoU
tracker so a video can keep that choice from frame to frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from barrett_faceswap.errors import ConfigError
from barrett_faceswap.i18n import t


@dataclass
class FaceMapEntry:
    """One row of a face-map file."""

    source: str
    target_index: int | None = None
    reference: str | None = None


@dataclass
class Track:
    """A face observed across frames."""

    track_id: int
    bbox: np.ndarray
    slot: int | None = None
    missed: int = 0


@dataclass
class FaceTracker:
    """Greedy IoU tracker. The slot sticks once a source has been chosen."""

    iou_threshold: float = 0.3
    max_missed: int = 8
    tracks: list[Track] = field(default_factory=list)
    _next_id: int = 1

    def update(self, bboxes: list[np.ndarray]) -> list[Track]:
        """Match ``bboxes`` to existing tracks and return one track per box."""

        unmatched_faces = set(range(len(bboxes)))
        unmatched_tracks = set(range(len(self.tracks)))
        scored: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            for face_index, bbox in enumerate(bboxes):
                score = bbox_iou(track.bbox, bbox)
                if score >= self.iou_threshold:
                    scored.append((score, track_index, face_index))
        scored.sort(reverse=True)
        assigned: dict[int, Track] = {}
        for _score, track_index, face_index in scored:
            if track_index not in unmatched_tracks or face_index not in unmatched_faces:
                continue
            unmatched_tracks.discard(track_index)
            unmatched_faces.discard(face_index)
            track = self.tracks[track_index]
            track.bbox = np.asarray(bboxes[face_index], dtype=np.float32).reshape(4)
            track.missed = 0
            assigned[face_index] = track
        for track_index in sorted(unmatched_tracks, reverse=True):
            self.tracks[track_index].missed += 1
        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]
        ordered: list[Track] = []
        for face_index, bbox in enumerate(bboxes):
            track = assigned.get(face_index)
            if track is None:
                track = Track(self._next_id, np.asarray(bbox, dtype=np.float32).reshape(4))
                self._next_id += 1
                self.tracks.append(track)
            ordered.append(track)
        return ordered


def bbox_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Intersection over union for ``x1, y1, x2, y2`` boxes."""

    ax1, ay1, ax2, ay2 = [float(value) for value in np.asarray(left).reshape(4)]
    bx1, by1, bx2, by2 = [float(value) for value in np.asarray(right).reshape(4)]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity of two vectors."""

    vector_a = np.asarray(left, dtype=np.float32).reshape(-1)
    vector_b = np.asarray(right, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b)) + 1e-8
    return float(np.dot(vector_a, vector_b) / denom)


def assign_slots(
    face_vectors: list[np.ndarray],
    slot_vectors: list[np.ndarray],
    threshold: float,
    *,
    one_to_one: bool,
) -> list[int | None]:
    """Map each face to a slot index, or ``None`` when nothing clears ``threshold``."""

    scores: list[tuple[float, int, int]] = []
    for face_index, face_vector in enumerate(face_vectors):
        for slot_index, slot_vector in enumerate(slot_vectors):
            scores.append((cosine(face_vector, slot_vector), face_index, slot_index))
    scores.sort(reverse=True)
    chosen: list[int | None] = [None] * len(face_vectors)
    used_faces: set[int] = set()
    used_slots: set[int] = set()
    for score, face_index, slot_index in scores:
        if score < threshold:
            break
        if face_index in used_faces:
            continue
        if one_to_one and slot_index in used_slots:
            continue
        chosen[face_index] = slot_index
        used_faces.add(face_index)
        used_slots.add(slot_index)
        if not one_to_one:
            # Independent mode still keeps the best slot per face; later scores
            # for that face are ignored because the face is used.
            continue
    return chosen


def load_face_map(path: str) -> list[FaceMapEntry]:
    """Read a JSON list of ``{source, target_index?, reference?}`` rows."""

    import json
    from pathlib import Path

    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(t("swap.map_read", path=file_path, error=exc)) from exc
    if not isinstance(payload, list) or not payload:
        raise ConfigError(t("swap.map_empty", path=file_path))
    entries: list[FaceMapEntry] = []
    for row in payload:
        if not isinstance(row, dict) or not row.get("source"):
            raise ConfigError(t("swap.map_row", path=file_path))
        index = row.get("target_index")
        entries.append(
            FaceMapEntry(
                source=str(row["source"]),
                target_index=None if index is None else int(index),
                reference=None if row.get("reference") is None else str(row["reference"]),
            )
        )
    return entries
