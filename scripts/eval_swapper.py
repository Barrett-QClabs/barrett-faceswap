#!/usr/bin/env python3
"""Score SSIM on a synthetic pair. This script does not load a network."""

from __future__ import annotations

import argparse

import numpy as np

from barrett_faceswap.i18n import t
from barrett_faceswap.train import ssim


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SSIM self-test. Does not download weights.")
    parser.add_argument("--check", action="store_true", help="Exit 0 without scoring pixels.")
    parser.add_argument("--self-test", action="store_true", help="Score an image against itself.")
    args = parser.parse_args(argv)
    if args.check and not args.self_test:
        print("eval: ssim, identity_cosine")
        return 0
    image = np.full((32, 32, 3), 40, dtype=np.uint8)
    score = ssim(image, image)
    print(t("train.eval_self", score=f"{score:.4f}"))
    return 0 if score > 0.99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
