"""Alignment template, SCRFD decode, and mask merge."""

from __future__ import annotations

import numpy as np
import pytest

from barrett_faceswap.align import ffhq_template, template_for_size
from barrett_faceswap.detect import decode_scrfd
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
