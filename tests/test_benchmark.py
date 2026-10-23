"""Benchmark command without weights or a GPU."""

from __future__ import annotations

from pathlib import Path

from barrett_faceswap.benchmark import run_benchmark
from barrett_faceswap.cli import main
from barrett_faceswap.config import load_config
from tests.conftest import write_config


def test_dry_run_benchmark_reports_fps(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, enabled=False))
    report = run_benchmark(config, frames=4, width=32, height=24, dry_run=True, providers=["cpu"])
    assert report.frames == 4
    assert report.fps > 0
    assert report.dry_run is True
    assert report.vram_mb is None or report.vram_mb >= 0
    text = report.line()
    assert "fps=" in text
    assert "vram_mb=" in text


def test_benchmark_cli(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "benchmark",
            "--config",
            str(write_config(tmp_path, enabled=False)),
            "--dry-run",
            "--frames",
            "3",
            "--width",
            "32",
            "--height",
            "32",
            "--execution-provider",
            "cpu",
        ]
    )
    assert code == 0
    assert "fps=" in capsys.readouterr().out
