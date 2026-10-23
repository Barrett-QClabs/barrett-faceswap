"""Time the processor chain and report frames per second.

VRAM is read from ``nvidia-smi`` when that tool is on PATH. A machine without
it, including CI, reports VRAM as unavailable. ``--dry-run`` times the chain
without opening ONNX weights.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import ConfigError
from barrett_faceswap.i18n import t
from barrett_faceswap.processors import create_processor
from barrett_faceswap.types import ProcessContext


@dataclass(frozen=True)
class BenchmarkReport:
    """One timing result. ``vram_mb`` is ``None`` when it cannot be read."""

    frames: int
    width: int
    height: int
    elapsed_s: float
    fps: float
    vram_mb: float | None
    dry_run: bool
    providers: str
    stages: tuple[tuple[str, float], ...] = ()
    realtime_fps: float = 0.0

    def line(self) -> str:
        vram = t("benchmark.vram_unavailable") if self.vram_mb is None else f"{self.vram_mb:.1f}"
        stage_text = ",".join(f"{name}:{elapsed * 1000.0:.2f}ms" for name, elapsed in self.stages) or "-"
        return t(
            "benchmark.report",
            frames=self.frames,
            width=self.width,
            height=self.height,
            elapsed=f"{self.elapsed_s:.4f}",
            fps=f"{self.fps:.2f}",
            realtime_fps=f"{self.realtime_fps:.2f}",
            stages=stage_text,
            vram=vram,
            dry_run=t("benchmark.yes") if self.dry_run else t("benchmark.no"),
            providers=self.providers,
        )


def query_vram_mb(device_id: int = 0) -> float | None:
    """GPU memory in use, in MiB, or ``None`` when nvidia-smi cannot answer."""

    nvidia = shutil.which("nvidia-smi")
    if nvidia is None:
        return None
    try:
        completed = subprocess.run(
            [nvidia, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if device_id < 0 or device_id >= len(lines):
        return None
    try:
        return float(lines[device_id])
    except ValueError:
        return None


def run_benchmark(
    config: AppConfig,
    *,
    frames: int,
    width: int,
    height: int,
    dry_run: bool,
    providers: list[str] | None = None,
) -> BenchmarkReport:
    """Process ``frames`` synthetic images through the configured processors."""

    if frames < 1:
        raise ConfigError(t("benchmark.frames"))
    if width < 16 or height < 16:
        raise ConfigError(t("benchmark.size"))
    selected = list(providers or config.execution.providers)
    source = np.zeros((height, width, 3), dtype=np.uint8)
    source[:, :] = (40, 90, 180)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (16, 16, 16)
    names = list(config.processors)
    processors = [create_processor(name, config, selected) for name in names]
    context = ProcessContext(source_bgr=source, dry_run=dry_run, notes=[], log_every_frame=False)
    prepare_started = time.perf_counter()
    for processor in processors:
        processor.pre_check(context)
    prepare_s = time.perf_counter() - prepare_started
    current = frame
    started = time.perf_counter()
    for _ in range(frames):
        current = frame.copy()
        if dry_run:
            cv2.putText(
                current,
                t("processor.dry_run_banner"),
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        for processor in processors:
            current = processor.process_frame(current, context)
    elapsed = max(time.perf_counter() - started, 1e-9)
    stages = (("prepare", prepare_s), ("swap", elapsed))
    for processor in processors:
        processor.post_process()
    del current
    return BenchmarkReport(
        frames=frames,
        width=width,
        height=height,
        elapsed_s=elapsed,
        fps=frames / elapsed,
        vram_mb=query_vram_mb(config.execution.device_id),
        dry_run=dry_run,
        providers=",".join(selected),
        stages=stages,
        realtime_fps=frames / elapsed,
    )


def run_live_benchmark(
    config: AppConfig,
    *,
    frames: int,
    width: int,
    height: int,
    dry_run: bool,
    providers: list[str] | None = None,
) -> BenchmarkReport:
    """Time a synthetic live session and report capture, detect, swap, and output."""

    from barrett_faceswap.live import LivePipeline
    from barrett_faceswap.types import SwapRequest

    if frames < 1:
        raise ConfigError(t("benchmark.frames"))
    selected = list(providers or config.execution.providers)
    source = _solid(width, height, (40, 90, 180))
    sequence = [_solid(width, height, (16, 16, 16)) for _ in range(frames)]
    import tempfile

    directory = Path(tempfile.mkdtemp(prefix="barrett-bench-"))
    source_path = directory / "source.png"
    cv2.imwrite(str(source_path), source)
    request = SwapRequest(
        source_path=source_path,
        target_path=Path("camera-bench"),
        output_path=directory / "session.txt",
        dry_run=dry_run,
        live_session=True,
        live_frames=frames,
        execution_providers=selected,
    )
    paced = config.model_copy(
        update={"live": config.live.model_copy(update={"queue_size": max(frames, 1), "preview": False})}
    )
    started = time.perf_counter()
    _result, stats = LivePipeline(paced).run(request, sequence, preview=False)
    elapsed = max(time.perf_counter() - started, 1e-9)
    order = ("capture", "detect", "swap", "output")
    stages = tuple((name, float(stats.stage_s.get(name, 0.0))) for name in order)
    produced = max(stats.frames_out, 1)
    return BenchmarkReport(
        frames=stats.frames_out,
        width=width,
        height=height,
        elapsed_s=elapsed,
        fps=produced / elapsed,
        vram_mb=query_vram_mb(config.execution.device_id),
        dry_run=dry_run,
        providers=",".join(selected),
        stages=stages,
        realtime_fps=produced / elapsed,
    )


def _solid(width: int, height: int, color: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = color
    return image
