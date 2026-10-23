"""Detect on some frames and carry the faces forward on the rest.

The tracker from the video path matches a new detection to the previous one.
Between detections the box and landmarks move by the last observed step, so a
live preview does not run the detector on every frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from barrett_faceswap.facesel import bbox_iou
from barrett_faceswap.types import Face


@dataclass
class _Tracked:
    bbox: np.ndarray
    landmarks: np.ndarray
    score: float
    velocity: np.ndarray


class DetectionCache:
    """Hold the last faces and step them forward when detection is skipped."""

    def __init__(self, every: int, iou_threshold: float = 0.3) -> None:
        self.every = max(1, int(every))
        self.iou_threshold = float(iou_threshold)
        self.index = 0
        self._tracks: list[_Tracked] = []

    def wants_detection(self) -> bool:
        """True on the first frame and every ``every`` frames after that."""

        return self.index % self.every == 0 or not self._tracks

    def observe(self, detected: list[Face] | None) -> list[Face]:
        """Record ``detected``, or advance the cache when it is ``None``.

        An empty list clears the tracks. The next skipped frame then has
        nothing to step, so a lost face does not keep drifting.
        """

        self.index += 1
        if detected is not None:
            self._tracks = [self._match(face) for face in detected]
            return self.faces()
        for track in self._tracks:
            track.bbox = track.bbox + track.velocity
            track.landmarks = track.landmarks + track.velocity[:2]
        return self.faces()

    def faces(self) -> list[Face]:
        return [
            Face(bbox=track.bbox.copy(), score=track.score, landmarks=track.landmarks.copy())
            for track in self._tracks
        ]

    def _match(self, face: Face) -> _Tracked:
        bbox = np.asarray(face.bbox, dtype=np.float32).reshape(4)
        landmarks = np.asarray(face.landmarks, dtype=np.float32).reshape(5, 2)
        velocity = np.zeros(4, dtype=np.float32)
        best_iou = 0.0
        best: _Tracked | None = None
        for track in self._tracks:
            score = bbox_iou(track.bbox, bbox)
            if score > best_iou:
                best_iou = score
                best = track
        if best is not None and best_iou >= self.iou_threshold:
            velocity = bbox - best.bbox
        return _Tracked(bbox=bbox, landmarks=landmarks, score=float(face.score), velocity=velocity)
