"""CPU checks for landmark refine, expression paste, and the quality defaults."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from barrett_faceswap.config import ExpressionConfig, LandmarkConfig
from barrett_faceswap.expression import closedness, gate_amounts, preserve_expression, pursedness
from barrett_faceswap.fetch import KNOWN_SHA256
from barrett_faceswap.landmarks import (
    FAN_SCORE_FLOOR,
    LandmarkRefiner,
    apply_fan,
    decode_fan,
    fan_matrix,
    points68_to_5,
    profile_ratio,
    should_replace_alignment,
)
from barrett_faceswap.onnx_io import feed_session
from barrett_faceswap.types import Face


def _settings() -> LandmarkConfig:
    return LandmarkConfig()


def _face(score: float, landmarks: np.ndarray) -> Face:
    return Face(
        bbox=np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32),
        score=score,
        landmarks=landmarks.astype(np.float32),
    )


def _frontal() -> np.ndarray:
    return np.array([[40, 40], [60, 40], [50, 55], [42, 70], [58, 70]], dtype=np.float32)


def test_points68_reduce_to_the_published_five() -> None:
    points = np.zeros((68, 2), dtype=np.float32)
    points[36:42] = (10, 20)
    points[42:48] = (30, 20)
    points[30] = (20, 28)
    points[48] = (12, 40)
    points[54] = (28, 40)
    five = points68_to_5(points)
    assert np.allclose(five[0], (10, 20))
    assert np.allclose(five[1], (30, 20))
    assert np.allclose(five[2], (20, 28))
    assert np.allclose(five[3], (12, 40))
    assert np.allclose(five[4], (28, 40))


def test_profile_and_low_score_replace_alignment() -> None:
    settings = _settings()
    frontal = _face(0.9, _frontal())
    assert profile_ratio(_frontal()) < 0.22
    assert should_replace_alignment(frontal, settings) is False
    profile = _frontal().copy()
    profile[2, 0] = 80
    assert profile_ratio(profile) >= 0.22
    assert should_replace_alignment(_face(0.9, profile), settings) is True
    assert should_replace_alignment(_face(0.4, _frontal()), settings) is True
    assert should_replace_alignment(frontal, LandmarkConfig(refine="always")) is True
    assert should_replace_alignment(_face(0.2, profile), LandmarkConfig(refine="off")) is False


def test_decode_scales_a_64_grid_back_to_the_frame() -> None:
    matrix = fan_matrix(np.array([0.0, 0.0, 100.0, 100.0]))
    raw = np.full((1, 68, 2), 32.0, dtype=np.float32)
    heatmap = np.ones((1, 68, 64, 64), dtype=np.float32)
    points, score = decode_fan([raw, heatmap], matrix)
    assert points is not None
    assert score >= FAN_SCORE_FLOOR
    assert np.allclose(points, 50.0, atol=1e-3)
    pixel = np.full((68, 2), 128.0, dtype=np.float32)
    unscaled, _score = decode_fan([pixel], matrix)
    assert unscaled is not None
    assert np.allclose(unscaled, 50.0, atol=1e-3)


class _Input:
    def __init__(self, name: str, shape: list[int]) -> None:
        self.name = name
        self.shape = shape
        self.type = "tensor(float)"


class _FanSession:
    def __init__(self, shape: list[int], heatmap: np.ndarray) -> None:
        self.shape = shape
        self.heatmap = heatmap
        self.feeds: dict[str, np.ndarray] = {}

    def get_inputs(self):
        return [_Input("input", self.shape)]

    def run(self, _none, feeds):
        self.feeds = feeds
        landmarks = np.full((1, 68, 2), 32.0, dtype=np.float32)
        return [landmarks, self.heatmap]


def test_fan_session_decodes_and_keeps_frontal_points() -> None:
    session = _FanSession([1, 3, 256, 256], np.ones((1, 68, 8, 8), dtype=np.float32))
    face = _face(0.9, _frontal())
    frame = np.full((120, 120, 3), 180, dtype=np.uint8)
    refined = apply_fan(frame, face, session, _settings())
    assert refined.landmarks_68 is not None
    assert np.allclose(refined.landmarks, face.landmarks)
    assert session.feeds["input"].shape == (1, 3, 256, 256)
    assert session.feeds["input"].dtype == np.float32
    assert float(session.feeds["input"].max()) <= 1.0


def test_low_heatmap_and_missing_file_leave_the_face() -> None:
    session = _FanSession([1, 256, 256, 3], np.zeros((1, 68, 4, 4), dtype=np.float32))
    face = _face(0.4, _frontal())
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    kept = apply_fan(frame, face, session, _settings())
    assert kept.landmarks_68 is None
    assert session.feeds["input"].shape == (1, 256, 256, 3)

    class _Store:
        def paths(self):
            return {"landmark": Path("/tmp/barrett-missing-2dfan4.onnx")}

    class _Config:
        landmarks = _settings()

    again = LandmarkRefiner(_Store(), _Config()).refine(frame, face)
    assert again.landmarks_68 is None
    assert np.allclose(again.landmarks, face.landmarks)


def test_expression_amounts_follow_ear_and_mar() -> None:
    settings = ExpressionConfig()
    closed = _ring(spread=0.8)
    opened = _ring(spread=4.0)
    shut_eye, shut_mouth, shut = gate_amounts(_mouth(closed, gap=0.0), settings)
    open_eye, open_mouth, wide = gate_amounts(_mouth(opened, gap=8.0), settings)
    assert shut == pytest.approx(1.0)
    assert shut_eye == pytest.approx(0.90)
    assert shut_mouth == pytest.approx(0.55)
    assert wide == pytest.approx(0.0)
    assert open_eye == pytest.approx(0.05)
    assert open_mouth == pytest.approx(0.0)
    assert closedness(0.0, 0.24, 0.15) == pytest.approx(1.0)
    assert pursedness(0.1, 0.045, 0.012) == pytest.approx(0.0)


def test_expression_paste_keeps_a_closed_eye_and_a_pursed_mouth() -> None:
    settings = ExpressionConfig()
    closed = preserve_expression(
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.full((64, 64, 3), 255, dtype=np.uint8),
        _mouth(_ring(spread=0.8), gap=0.0),
        np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        settings,
    )
    opened = preserve_expression(
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.full((64, 64, 3), 255, dtype=np.uint8),
        _mouth(_ring(spread=4.0), gap=8.0),
        np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        settings,
    )
    assert closed is not None and opened is not None
    closed_crop, closed_openness = closed
    open_crop, open_openness = opened
    assert int(closed_crop[20, 16, 0]) > 180
    assert int(open_crop[20, 16, 0]) < 40
    assert int(closed_crop[40, 40, 0]) > 80
    assert int(open_crop[40, 40, 0]) < 20
    assert closed_openness == pytest.approx(0.0)
    assert open_openness == pytest.approx(1.0)
    assert preserve_expression(
        np.zeros((8, 8, 3), np.uint8),
        np.zeros((8, 8, 3), np.uint8),
        np.zeros((5, 2), np.float32),
        np.eye(2, 3, dtype=np.float32),
        settings,
    ) is None


def test_codeformer_weight_is_float64_when_the_input_is_double() -> None:
    seen: dict[str, np.ndarray] = {}

    class _Item:
        def __init__(self, name: str, shape: list[int], type_name: str) -> None:
            self.name = name
            self.shape = shape
            self.type = type_name

    class _Session:
        def get_inputs(self):
            return [
                _Item("input", [1, 3, 4, 4], "tensor(float)"),
                _Item("weight", [1], "tensor(double)"),
            ]

        def run(self, _none, feeds):
            seen.update(feeds)
            return [feeds["input"]]

    image = np.zeros((1, 3, 4, 4), dtype=np.float32)
    feed_session(_Session(), image, weight=0.7)
    assert seen["input"].dtype == np.float32
    assert seen["weight"].dtype == np.float64
    assert float(seen["weight"].reshape(-1)[0]) == pytest.approx(0.7)


def test_2dfan4_pin_is_the_lfs_oid() -> None:
    assert KNOWN_SHA256["2dfan4.onnx"] == "678c6fa539d52335a31c980feefdf4a6e02d781d83dce00af8a894f114557285"


def _ring(spread: float) -> np.ndarray:
    points = np.zeros((68, 2), dtype=np.float32)
    for index, x in enumerate(range(17, 27)):
        points[x] = (8 + index * 2, 8)
    eye = [
        (10, 20),
        (13, 20 - spread),
        (19, 20 - spread),
        (22, 20),
        (19, 20 + spread),
        (13, 20 + spread),
    ]
    points[36:42] = eye
    points[42:48] = [(x + 24, y) for x, y in eye]
    return points


def _mouth(points: np.ndarray, gap: float) -> np.ndarray:
    updated = points.copy()
    for index, angle in enumerate(np.linspace(0, 2 * np.pi, 20, endpoint=False)):
        updated[48 + index] = (40 + 6 * np.cos(angle), 40 + 6 * np.sin(angle))
    updated[48] = (30, 40)
    updated[54] = (50, 40)
    updated[62] = (40, 40 - gap / 2)
    updated[66] = (40, 40 + gap / 2)
    return updated
