"""Sunglasses keep a fuller identity swap and do not drive the eyes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from barrett_faceswap.align import swap_template
from barrett_faceswap.config import ExpressionConfig, load_config
from barrett_faceswap.expression import preserve_expression
from barrett_faceswap.i18n import set_language, t
from barrett_faceswap.processors.face_swapper import FaceSwapper
from barrett_faceswap.sunglasses import apply_sunglass_mask, sunglass_plan, wearing_sunglasses
from barrett_faceswap.types import Face, ProcessContext
from tests.conftest import write_config


def _template(side: int = 128) -> np.ndarray:
    return swap_template("inswapper", side)


def _bar(crop: np.ndarray, template: np.ndarray) -> np.ndarray:
    painted = crop.copy()
    left, right = template[0], template[1]
    span = abs(float(right[0] - left[0]))
    y = int(round(float(left[1] + right[1]) * 0.5))
    x0 = int(min(left[0], right[0]) - 0.1 * span)
    x1 = int(max(left[0], right[0]) + 0.1 * span)
    painted[y - 8 : y + 6, x0:x1] = 0
    return painted


def _ring(spread: float) -> np.ndarray:
    points = np.zeros((68, 2), dtype=np.float32)
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
    for index, x in enumerate(range(17, 27)):
        points[x] = (8 + index * 2, 8)
    return points


def _mouth(points: np.ndarray) -> np.ndarray:
    updated = points.copy()
    for index, angle in enumerate(np.linspace(0, 2 * np.pi, 20, endpoint=False)):
        updated[48 + index] = (40 + 6 * np.cos(angle), 40 + 6 * np.sin(angle))
    updated[48] = (30, 40)
    updated[54] = (50, 40)
    updated[62] = (40, 40)
    updated[66] = (40, 40)
    return updated


def test_dark_bar_and_parser_glasses_are_sunglasses() -> None:
    template = _template()
    plain = np.full((128, 128, 3), 180, dtype=np.uint8)
    assert wearing_sunglasses(plain, template) is False
    assert wearing_sunglasses(_bar(plain, template), template) is True
    pupils = plain.copy()
    for point in template[:2]:
        x, y = int(point[0]), int(point[1])
        pupils[y - 2 : y + 2, x - 2 : x + 2] = 0
    assert wearing_sunglasses(pupils, template) is False
    assert wearing_sunglasses(np.zeros_like(plain), template) is False

    labels = np.ones((64, 64), dtype=np.int32)
    assert wearing_sunglasses(plain[:64, :64], template, labels) is False
    labels[18:30, 6:58] = 6
    assert wearing_sunglasses(plain[:64, :64], template, labels) is True
    speck = np.ones((64, 64), dtype=np.int32)
    speck[24:26, 30:33] = 6
    assert wearing_sunglasses(plain[:64, :64], template, speck) is False


def test_plan_stays_off_when_disabled_or_the_face_is_bare() -> None:
    from barrett_faceswap.config import SunglassesConfig

    template = _template()
    plain = np.full((128, 128, 3), 180, dtype=np.uint8)
    settings = SunglassesConfig()
    assert sunglass_plan(settings, plain, template).active is False
    active = sunglass_plan(settings, _bar(plain, template), template)
    assert active.active is True
    assert active.expression is True
    assert active.mouth_scale == pytest.approx(0.45)
    assert active.box_inset == pytest.approx(0.04)
    disabled = SunglassesConfig(enabled=False)
    assert sunglass_plan(disabled, _bar(plain, template), template).active is False


def test_fuller_mask_fills_the_midface_hole() -> None:
    template = _template()
    mask = np.ones((128, 128), dtype=np.float32)
    mask[:4, :4] = 0.0
    nose = template[2]
    y, x = int(nose[1]), int(nose[0])
    mask[y - 6 : y + 6, x - 10 : x + 10] = 0.0
    filled = apply_sunglass_mask(mask, 128, 1, template, 0.04)
    assert filled[y, x] == pytest.approx(1.0)
    assert filled[0, 0] == pytest.approx(0.0)
    assert int(np.count_nonzero(filled[y - 6 : y + 6, x - 10 : x + 10] < 0.99)) == 0


def test_sunglass_expression_skips_the_eyes_and_keeps_a_mild_mouth() -> None:
    settings = ExpressionConfig()
    points = _mouth(_ring(0.8))
    matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    swapped = np.zeros((64, 64, 3), dtype=np.uint8)
    target = np.full((64, 64, 3), 255, dtype=np.uint8)
    full = preserve_expression(swapped, target, points, matrix, settings)
    mild = preserve_expression(swapped, target, points, matrix, settings, include_eyes=False, mouth_scale=0.45)
    assert full is not None and mild is not None
    full_crop, _openness = full
    mild_crop, mild_openness = mild
    assert int(full_crop[20, 16, 0]) > 180
    assert int(mild_crop[20, 16, 0]) == 0
    assert int(mild_crop[40, 40, 0]) > 20
    assert int(mild_crop[40, 40, 0]) < int(full_crop[40, 40, 0])
    assert mild_openness == pytest.approx(1.0)


def test_swapper_uses_the_sunglass_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    set_language("en")
    config = load_config(write_config(tmp_path, enabled=False))
    config = config.model_copy(
        update={
            "blend": config.blend.model_copy(update={"color": "none", "eye_mode": "target", "eye_sharpen": 0, "mask_blur": 1}),
            "landmarks": config.landmarks.model_copy(update={"refine": "off"}),
        }
    )
    template = _template()
    frame = np.full((160, 160, 3), 180, dtype=np.uint8)
    frame = _bar(frame, template)
    nose = template[2]
    y, x = int(nose[1]), int(nose[0])
    hole = np.ones((128, 128), dtype=np.float32)
    hole[y - 6 : y + 6, x - 10 : x + 10] = 0.0
    seen: dict[str, np.ndarray] = {}

    def paste(base, colored, matrix, mask):
        del base, matrix
        seen["colored"] = colored.copy()
        seen["mask"] = mask.copy()
        return frame

    monkeypatch.setattr("barrett_faceswap.processors.face_swapper.paste_back", paste)
    face = Face(bbox=np.array([8, 8, 140, 140], dtype=np.float32), score=0.9, landmarks=template.copy())
    context = ProcessContext(
        source_bgr=frame,
        dry_run=False,
        known_faces=[face],
        source_embedding=(1.0, 0.0, 0.0, 0.0),
        log_every_frame=True,
    )

    class _Backend:
        def forward(self, tile, latent):
            del latent
            output = np.zeros_like(tile)
            output[..., 2] = 220
            return output

    def _run(updated) -> ProcessContext:
        seen.clear()
        swapper = FaceSwapper(updated, ["cpu"])
        swapper.store = object()
        swapper.masks = type("Masks", (), {"compose": lambda *args, **kwargs: hole.copy(), "eye_mask": None, "labels": None})()
        swapper._backend = lambda: _Backend()  # type: ignore[method-assign]
        ctx = ProcessContext(
            source_bgr=context.source_bgr,
            dry_run=False,
            known_faces=[face],
            source_embedding=(1.0, 0.0, 0.0, 0.0),
            log_every_frame=True,
        )
        swapper.process_frame(frame, ctx)
        return ctx

    covered = _run(config)
    eye = template[0]
    ey, ex = int(eye[1]), int(eye[0])
    assert seen["mask"][y, x] == pytest.approx(1.0)
    assert int(seen["colored"][ey, ex, 2]) == 220
    assert covered.eye_openness == pytest.approx(1.0)
    assert t("sunglasses.path") in covered.notes

    plain = _run(config.model_copy(update={"sunglasses": config.sunglasses.model_copy(update={"enabled": False})}))
    assert seen["mask"][y, x] == pytest.approx(0.0)
    assert int(seen["colored"][ey, ex, 2]) < 30
    assert t("sunglasses.path") not in plain.notes
