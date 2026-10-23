"""BiSeNet-style face-parsing labels and mask rasterization.

The 19 indexes are the CelebAMask-HQ order used by public face-parsing
exports: 0 is background, then skin through hat. This module does not ship a
checkpoint and does not vendor the training code. The ONNX file, when the
operator places one on disk, is expected to return either per-class scores or
an integer label map.
"""

from __future__ import annotations

PARSING_REGIONS: dict[str, int] = {
    "background": 0,
    "skin": 1,
    "left_eyebrow": 2,
    "right_eyebrow": 3,
    "left_eye": 4,
    "right_eye": 5,
    "eye_glasses": 6,
    "left_ear": 7,
    "right_ear": 8,
    "earring": 9,
    "nose": 10,
    "mouth": 11,
    "upper_lip": 12,
    "lower_lip": 13,
    "neck": 14,
    "necklace": 15,
    "cloth": 16,
    "hair": 17,
    "hat": 18,
}

# Facial surface used when the operator does not override masks.regions.
# Hair, cloth, hat, neck, and ears stay out so the paste does not cover them.
DEFAULT_PARSE_REGIONS: tuple[str, ...] = (
    "skin",
    "left_eyebrow",
    "right_eyebrow",
    "left_eye",
    "right_eye",
    "eye_glasses",
    "nose",
    "mouth",
    "upper_lip",
    "lower_lip",
)
