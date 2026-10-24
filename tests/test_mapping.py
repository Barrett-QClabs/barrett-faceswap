"""One export can send each target face to a different source."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from barrett_faceswap.config import load_config
from barrett_faceswap.errors import ConfigError
from barrett_faceswap.facesel import load_face_map
from barrett_faceswap.library import FaceLibrary, SampleParts
from barrett_faceswap.mapping import (
    KEEP,
    UPLOAD,
    Assignment,
    compile_assignments,
    detect_target_faces,
    mapping_is_complete,
    write_face_map,
)
from barrett_faceswap.models import ModelStore
from barrett_faceswap.processors.face_swapper import FaceSwapper
from barrett_faceswap.types import Face, ProcessContext
from barrett_faceswap.ui.app import source_files_ready
from tests.conftest import write_config


def _config(tmp_path: Path):
    return load_config(write_config(tmp_path, enabled=False))


def _face(x: float, size: float = 40.0, score: float = 0.9) -> Face:
    landmarks = np.array(
        [[x + 12, 40], [x + 28, 40], [x + 20, 55], [x + 14, 70], [x + 26, 70]],
        dtype=np.float32,
    )
    return Face(
        bbox=np.array([x, 10, x + size, 10 + size], dtype=np.float32),
        score=score,
        landmarks=landmarks,
    )


def _vector(index: int) -> np.ndarray:
    vector = np.zeros(8, dtype=np.float32)
    vector[index] = 1.0
    return vector


class _Masks:
    def compose(self, crop, side, blur, landmarks=None, **_ignored):
        del crop, blur, landmarks, _ignored
        return np.ones((side, side), dtype=np.float32)


class _Paint:
    def __init__(self) -> None:
        self.forwards = 0
        self.seen: list[int] = []

    def forward(self, tile, latent):
        self.forwards += 1
        slot = int(np.argmax(np.asarray(latent).reshape(-1)[:4]))
        self.seen.append(slot)
        color = 40 if slot == 0 else 200
        return np.full(tile.shape, color, dtype=np.uint8)


def test_face_map_reads_embedding_and_match(tmp_path: Path) -> None:
    payload = [
        {"source": "left.png", "target_index": 1},
        {"source": "right.png", "embedding": [1.0, 0.0], "match": [0.0, 1.0], "reference": "thumb.png"},
    ]
    path = tmp_path / "map.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = load_face_map(str(path))
    assert rows[0].embedding is None
    assert rows[0].match is None
    assert rows[0].target_index == 1
    assert rows[1].embedding == (1.0, 0.0)
    assert rows[1].match == (0.0, 1.0)
    assert rows[1].reference == "thumb.png"
    broken = tmp_path / "bad.json"
    broken.write_text(json.dumps([{"source": "a.png", "embedding": "nope"}]), encoding="utf-8")
    with pytest.raises(ConfigError, match="embedding"):
        load_face_map(str(broken))


def test_compile_skips_keep_and_resolves_profiles(tmp_path: Path) -> None:
    library = FaceLibrary(tmp_path / "library")
    left = library.create("Left")
    right = library.create("Right")
    still = np.zeros((16, 16, 3), dtype=np.uint8)
    speech = np.full(1600, 0.2, dtype=np.float32)
    library.add_sample(left.id, SampleParts("photo", _vector(0), 1.0, still, speech, filename="left.png"))
    library.add_sample(right.id, SampleParts("photo", _vector(1), 1.0, still, None, filename="right.png"))
    state = [
        {"match": [1.0, 0.0, 0.0], "reference": "thumb-a.png"},
        {"match": [0.0, 1.0, 0.0], "reference": "thumb-b.png"},
        {"match": [0.0, 0.0, 1.0], "reference": "thumb-c.png"},
    ]
    photo = tmp_path / "upload.png"
    photo.write_bytes(b"png")
    rows = compile_assignments(state, [KEEP, left.id, UPLOAD], [None, None, photo], library)
    assert len(rows) == 2
    assert rows[0].embedding is not None
    assert rows[0].embedding[0] == pytest.approx(1.0)
    assert rows[0].voice is not None
    assert Path(rows[0].voice).is_file()
    assert rows[1].source == str(photo)
    assert rows[1].embedding is None
    assert rows[1].match == (0.0, 0.0, 1.0)
    written = tmp_path / "out.json"
    write_face_map(written, rows)
    loaded = load_face_map(str(written))
    assert loaded[0].match == rows[0].match
    assert loaded[0].embedding == rows[0].embedding
    assert loaded[1].embedding is None
    assert mapping_is_complete([KEEP, KEEP], [None, None]) is False
    assert mapping_is_complete([UPLOAD, left.id], [None, None]) is False
    assert mapping_is_complete([UPLOAD, left.id], [photo, None]) is True
    assert source_files_ready("video", None, None, None, "target.png", mapped=True) is True
    assert source_files_ready("video", None, None, None, None, mapped=True) is False


def test_detect_keeps_the_largest_faces(tmp_path: Path) -> None:
    image = np.zeros((200, 320, 3), dtype=np.uint8)
    faces = [_face(20 + index * 40, size=20 + index * 8) for index in range(5)]

    def detect(_image):
        return list(faces)

    def embed(_image, face: Face):
        return _vector(0 if float(face.bbox[0]) < 100 else 1)

    result = detect_target_faces(_config(tmp_path), image, tmp_path / "thumbs", detector=detect, embedder=embed)
    assert result.total == 5
    assert len(result.faces) == 4
    assert result.faces[0].bbox[2] - result.faces[0].bbox[0] == pytest.approx(20 + 4 * 8)
    assert Path(result.faces[0].reference).is_file()
    assert result.faces[-1].thumb.size > 0


def test_two_faces_take_two_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    face_map = tmp_path / "map.json"
    write_face_map(
        face_map,
        [
            Assignment(str(left), tuple(float(value) for value in _vector(0)), tuple(float(value) for value in _vector(0))),
            Assignment(str(right), tuple(float(value) for value in _vector(1)), tuple(float(value) for value in _vector(1))),
        ],
    )
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "swap": config.swap.model_copy(
                update={"selector": "map", "face_map": face_map, "resolution": 32, "similarity": 0.45}
            ),
            "blend": config.blend.model_copy(update={"color": "none", "eye_mode": "swap"}),
            "expression": config.expression.model_copy(update={"enabled": False}),
            "sunglasses": config.sunglasses.model_copy(update={"enabled": False}),
            "landmarks": config.landmarks.model_copy(update={"refine": "off"}),
        }
    )

    def fake_embed(store, image, landmarks):
        del store, image
        x = float(np.asarray(landmarks)[0, 0])
        if x < 80:
            return _vector(0)
        if x < 200:
            return _vector(1)
        return _vector(2)

    monkeypatch.setattr("barrett_faceswap.processors.face_swapper.embed_arcface", fake_embed)
    swapper = FaceSwapper(config, ["cpu"])
    painter = _Paint()
    swapper.store = ModelStore(config, ["cpu"])
    swapper.masks = _Masks()
    swapper._backend = lambda: painter  # type: ignore[method-assign]
    faces = [_face(10, size=70), _face(140, size=70), _face(230, size=50)]
    swapper._detect = lambda image: list(faces)  # type: ignore[method-assign]
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    ctx = ProcessContext(source_bgr=frame, dry_run=False, log_every_frame=False)
    painted = swapper.process_frame(frame.copy(), ctx)
    assert painter.forwards == 2
    assert painter.seen == [0, 1]
    assert int(painted.max()) > 0
    assert not np.array_equal(painted, frame)

    empty = FaceSwapper(config, ["cpu"])
    quiet = _Paint()
    empty.store = swapper.store
    empty.masks = _Masks()
    empty._backend = lambda: quiet  # type: ignore[method-assign]
    empty._detect = lambda image: []  # type: ignore[method-assign]
    untouched = frame.copy()
    passed = empty.process_frame(
        untouched,
        ProcessContext(source_bgr=frame, dry_run=False, allow_missing_face=True, log_every_frame=False),
    )
    assert quiet.forwards == 0
    assert np.array_equal(passed, untouched)

    artifact = Path("/opt/cursor/artifacts")
    if artifact.is_dir():
        import cv2

        cv2.imwrite(str(artifact / "map_two_faces.png"), painted)
