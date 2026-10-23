"""Grid of swapper and enhancer pairs.

A dry run draws a labeled contact sheet and a timing table without opening
weights. A real run processes one still, or the first frames of a short clip,
and skips a cell whose weight file is missing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from barrett_faceswap.config import ENHANCER_MODELS, AppConfig
from barrett_faceswap.errors import BarrettFaceswapError
from barrett_faceswap.i18n import t
from barrett_faceswap.swap_catalog import SWAPPER_NAMES
from barrett_faceswap.types import SwapRequest
from barrett_faceswap.vision import is_video, read_image, write_image


@dataclass
class CompareCell:
    swapper: str
    enhancer: str
    elapsed_s: float
    status: str


def run_compare(
    config: AppConfig,
    source: Path,
    target: Path,
    output: Path,
    *,
    swappers: list[str] | None = None,
    enhancers: list[str] | None = None,
    frames: int = 1,
    dry_run: bool = False,
    providers: list[str] | None = None,
) -> list[CompareCell]:
    """Write ``output`` and a sibling ``.timing.txt``. Return one row per cell."""

    chosen_swappers = list(swappers or SWAPPER_NAMES)
    chosen_enhancers = list(enhancers or ENHANCER_MODELS)
    limit = max(1, int(frames))
    samples = [] if dry_run else _samples(target, limit)
    cells: list[CompareCell] = []
    tiles: list[list[np.ndarray]] = []
    for swapper in chosen_swappers:
        row: list[np.ndarray] = []
        for enhancer in chosen_enhancers:
            started = time.perf_counter()
            if dry_run:
                image = _blank(target if target.is_file() and not is_video(target) else None)
                status = "dry-run"
            else:
                image, status = _run_cell(
                    config,
                    source,
                    target,
                    samples,
                    swapper,
                    enhancer,
                    providers,
                )
            elapsed = time.perf_counter() - started
            cells.append(CompareCell(swapper, enhancer, elapsed, status))
            row.append(_label(image, swapper, enhancer, status))
        tiles.append(row)
    sheet = _sheet(tiles)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_image(output, sheet)
    table = output.with_suffix(".timing.txt")
    lines = [t("compare.line", swapper="swapper", enhancer="enhancer", seconds="seconds", status="status")]
    for cell in cells:
        lines.append(
            t(
                "compare.line",
                swapper=cell.swapper,
                enhancer=cell.enhancer,
                seconds=f"{cell.elapsed_s:.4f}",
                status=cell.status,
            )
        )
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cells


def _samples(target: Path, frames: int) -> list[np.ndarray]:
    if not is_video(target):
        return [read_image(target)]
    capture = cv2.VideoCapture(str(target))
    samples: list[np.ndarray] = []
    try:
        while len(samples) < frames:
            ok, frame = capture.read()
            if not ok:
                break
            samples.append(frame)
    finally:
        capture.release()
    if not samples:
        raise BarrettFaceswapError(t("compare.missing", path=target))
    return samples


def _run_cell(config, source, target, samples, swapper, enhancer, providers) -> tuple[np.ndarray, str]:
    from barrett_faceswap.pipeline import FaceSwapPipeline

    tuned = config.model_copy(
        update={
            "swap": config.swap.model_copy(update={"model": swapper}),
            "enhancer": config.enhancer.model_copy(update={"model": enhancer}),
            "processors": ["face_swapper", "face_enhancer"],
        }
    )
    last = _blank(None)
    status = "ok"
    work = Path(str(target) + f".compare-{swapper}-{enhancer}.png")
    try:
        for frame in samples:
            still = work
            write_image(still, frame)
            request = SwapRequest(
                source_path=source,
                target_path=still,
                output_path=still.with_name(still.stem + "-out.png"),
                dry_run=False,
                execution_providers=providers,
                with_enhancer=True,
                swapper_model=swapper,
                enhancer_model=enhancer,
            )
            result = FaceSwapPipeline(tuned).run(request)
            last = read_image(result.output_path)
    except BarrettFaceswapError:
        status = "missing"
        last = _blank(None)
    finally:
        work.unlink(missing_ok=True)
        work.with_name(work.stem + "-out.png").unlink(missing_ok=True)
    return last, status


def _blank(path: Path | None) -> np.ndarray:
    if path is not None and path.is_file() and not is_video(path):
        image = read_image(path)
        return cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
    return np.full((160, 160, 3), 32, np.uint8)


def _label(image: np.ndarray, swapper: str, enhancer: str, status: str) -> np.ndarray:
    tile = cv2.resize(image, (220, 180), interpolation=cv2.INTER_AREA)
    canvas = np.full((220, 220, 3), 16, np.uint8)
    canvas[32:212, 0:220] = tile
    cv2.putText(canvas, swapper[:22], (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{enhancer[:18]} {status}",
        (6, 214),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (180, 220, 180),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _sheet(rows: list[list[np.ndarray]]) -> np.ndarray:
    if not rows or not rows[0]:
        return np.zeros((32, 32, 3), np.uint8)
    height, width = rows[0][0].shape[:2]
    sheet = np.zeros((height * len(rows), width * len(rows[0]), 3), np.uint8)
    for y, row in enumerate(rows):
        for x, tile in enumerate(row):
            sheet[y * height : (y + 1) * height, x * width : (x + 1) * width] = tile
    return sheet
