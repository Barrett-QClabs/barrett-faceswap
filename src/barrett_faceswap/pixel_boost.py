"""Tile a large aligned crop into native-size swapper inferences.

The crop is already aligned at the boosted resolution. Each tile is a
full-face subsample, ``crop[row::count, col::count]``, not a quadrant.
Tiles are inferred independently with the same identity vector and written
back on the same lattice.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from barrett_faceswap.errors import SwapError
from barrett_faceswap.i18n import t


def run_tiled(crop_bgr: np.ndarray, native: int, infer: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Run ``infer`` on native tiles of ``crop_bgr`` and reassemble.

    A crop whose side equals ``native`` is one inference. A side that is not
    a multiple of ``native`` is also one inference, so a legacy resolution
    knob can still reach a mocked graph. Real weights should use a listed
    pixel-boost size.
    """

    if crop_bgr.ndim != 3 or crop_bgr.shape[0] != crop_bgr.shape[1]:
        raise SwapError(t("swap.tile_shape", shape=tuple(int(dim) for dim in crop_bgr.shape)))
    side = int(crop_bgr.shape[0])
    native = int(native)
    if native <= 0:
        raise SwapError(t("swap.tile_shape", shape=(native,)))
    if side == native or side % native != 0:
        return infer(crop_bgr)
    count = side // native
    canvas = np.empty_like(crop_bgr)
    for row in range(count):
        for col in range(count):
            tile = np.ascontiguousarray(crop_bgr[row::count, col::count])
            restored = infer(tile)
            if restored.shape[0] != tile.shape[0] or restored.shape[1] != tile.shape[1]:
                raise SwapError(
                    t("swap.tile_output", expected=native, shape=tuple(int(dim) for dim in restored.shape))
                )
            canvas[row::count, col::count] = restored
    return canvas
