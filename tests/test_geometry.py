"""Alignment template, SCRFD decode, and mask merge."""

from __future__ import annotations

import numpy as np
import pytest

from barrett_faceswap.align import ffhq_template, template_for_size
from barrett_faceswap.detect import FaceDetector, decode_scrfd, letterbox_bgr
from barrett_faceswap.masks import box_feather_mask, merge_min, occlusion_mask, region_mask


def test_ffhq_template_scales_from_512() -> None:
    full = ffhq_template(512)
    half = ffhq_template(256)
    assert full.shape == (5, 2)
    assert np.allclose(half * 2.0, full)


def test_inswapper_template_shifts_the_112_points() -> None:
    template_112 = template_for_size(112)
    template_128 = template_for_size(128)
    assert template_112.shape == (5, 2)
    assert np.allclose(template_128[:, 0], template_112[:, 0] + 8.0)


def test_scrfd_decoder_keeps_a_high_score_anchor() -> None:
    scores = np.array([[0.95], [0.10]], dtype=np.float32)
    boxes = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
    keypoints = np.zeros((2, 10), dtype=np.float32)
    faces = decode_scrfd([scores, boxes, keypoints], det_size=32, score_threshold=0.5, nms_threshold=0.4)
    assert len(faces) == 1
    assert faces[0].score == pytest.approx(0.95)
    assert faces[0].landmarks.shape == (5, 2)


def test_letterbox_keeps_aspect_and_maps_the_box_back() -> None:
    image = np.zeros((200, 100, 3), dtype=np.uint8)
    image[:, :50] = 255
    image[:, 50:] = 10
    canvas, scale, pad_x, pad_y = letterbox_bgr(image, 64, 64)
    assert canvas.shape == (64, 64, 3)
    assert pad_x == 0.0 and pad_y == 0.0
    assert scale == pytest.approx(64 / 200)
    assert int(canvas[0, 8, 0]) == 255
    assert int(canvas[0, 24, 0]) == 10
    assert int(canvas[0, 48, 0]) == 0

    scores = np.array([[0.95], [0.10]], dtype=np.float32)
    boxes = np.array([[0.0, 0.0, 0.5, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    keypoints = np.zeros((2, 10), dtype=np.float32)
    keypoints[0, 0] = 16.0 / 64.0
    keypoints[0, 1] = 32.0 / 64.0
    faces = decode_scrfd(
        [scores, boxes, keypoints],
        det_size=64,
        score_threshold=0.5,
        nms_threshold=0.4,
        origin_size=(100, 200),
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    assert len(faces) == 1
    assert float(faces[0].bbox[2]) == pytest.approx(99.0)
    assert float(faces[0].bbox[3]) == pytest.approx(199.0)
    assert float(faces[0].landmarks[0, 0]) == pytest.approx(50.0)
    assert float(faces[0].landmarks[0, 1]) == pytest.approx(100.0)


def test_letterbox_pad_is_applied_when_mapping() -> None:
    scores = np.array([[0.9]], dtype=np.float32)
    boxes = np.array([[0.0, 0.0, 0.25, 0.25]], dtype=np.float32)
    keypoints = np.zeros((1, 10), dtype=np.float32)
    keypoints[0, 0] = 16.0 / 64.0
    keypoints[0, 1] = 16.0 / 64.0
    faces = decode_scrfd(
        [scores, boxes, keypoints],
        det_size=64,
        score_threshold=0.5,
        nms_threshold=0.4,
        origin_size=(80, 80),
        scale=1.0,
        pad_x=8.0,
        pad_y=4.0,
    )
    assert float(faces[0].landmarks[0, 0]) == pytest.approx(8.0)
    assert float(faces[0].landmarks[0, 1]) == pytest.approx(12.0)


def test_try_detect_letterboxes_before_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Input:
        name = "input"
        shape = [1, 3, 64, 64]

    class _Session:
        def get_inputs(self):
            return [_Input()]

        def get_providers(self):
            return ["CPUExecutionProvider"]

    def fake_run(session, feeds, cuda_graph=False):
        del session, cuda_graph
        blob = next(iter(feeds.values()))
        assert blob.shape == (1, 3, 64, 64)
        black = (0.0 - 127.5) / 128.0
        assert np.allclose(blob[0, :, :, 40:], black, atol=1e-4)
        assert not np.allclose(blob[0, :, :, :16], black, atol=1e-3)
        scores = np.array([[0.95], [0.10]], dtype=np.float32)
        boxes = np.array([[0.0, 0.0, 0.5, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        keypoints = np.zeros((2, 10), dtype=np.float32)
        keypoints[0, 0] = 16.0 / 64.0
        keypoints[0, 1] = 32.0 / 64.0
        return [scores, boxes, keypoints]

    monkeypatch.setattr("barrett_faceswap.detect.run_feeds", fake_run)
    image = np.zeros((200, 100, 3), dtype=np.uint8)
    image[:] = (30, 60, 90)
    faces = FaceDetector(_Session(), 64, 0.5, 0.4).try_detect(image)
    assert len(faces) == 1
    assert float(faces[0].bbox[2]) == pytest.approx(99.0)
    assert float(faces[0].landmarks[0, 0]) == pytest.approx(50.0)
    assert float(faces[0].landmarks[0, 1]) == pytest.approx(100.0)


def test_mask_min_merge_and_occlusion_stub() -> None:
    left = np.array([[1.0, 0.2], [0.4, 0.0]], dtype=np.float32)
    right = np.array([[0.5, 0.8], [0.4, 0.3]], dtype=np.float32)
    assert np.allclose(merge_min([left, right]), np.array([[0.5, 0.2], [0.4, 0.0]]))
    occlusion = occlusion_mask(16)
    assert occlusion.shape == (16, 16)
    assert float(occlusion.min()) == 1.0
    box = box_feather_mask(32, blur=0)
    region = region_mask(32, blur=0)
    assert box[0, 0] == 0
    assert region[16, 16] > 0
