"""Import, dry-run, and missing-model behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from barrett_faceswap import __version__
from barrett_faceswap.cli import main
from barrett_faceswap.config import load_config
from barrett_faceswap.errors import ModelNotFoundError
from barrett_faceswap.execution import prefer_cuda
from barrett_faceswap.i18n import t
from barrett_faceswap.pipeline import FaceSwapPipeline
from barrett_faceswap.processors import available_processors
from barrett_faceswap.types import SwapRequest
from tests.conftest import REPO, write_config, write_png


def test_package_imports() -> None:
    assert __version__ == "0.4.0"
    assert "face_swapper" in available_processors()
    assert "face_enhancer" in available_processors()


def test_shipped_compliance_default_is_off() -> None:
    config = load_config(REPO / "configs" / "default.yaml")
    assert config.compliance.enabled is False
    assert config.ui.language == "en"
    assert config.models.swapper.sha256 is None
    assert config.models.detector.path.name == "det_10g.onnx"
    assert config.processors == ["face_swapper"]
    assert config.video.encoder == "libx264"
    assert config.enhancer.model == "gfpgan_1.4"
    assert config.swap.model == "inswapper_128"
    assert config.video.mode == "disk"
    assert config.blend.mode == "alpha"
    assert config.blend.color == "lab"
    assert config.blend.eye_mode == "blend"
    assert config.blend.eye_sharpen == 0.6
    assert config.enhancer.blend == 0.6
    assert config.enhancer.eye_blend == 0.35
    assert config.enhancer.temporal == 0.4
    assert config.enhancer.temporal_motion == 0.04
    assert config.video.max_face_gap == 10
    assert config.video.smooth == "one_euro"
    assert config.compliance.enabled is False
    assert config.models.gfpgan.path.name == "gfpgan_1.4.onnx"
    assert config.execution.commercial is False
    assert config.execution.precision == "fp32"
    assert config.execution.cuda_graph is False
    assert config.live.queue_size == 1
    assert config.live.detect_every == 3
    assert config.compliance.watermark.font_path is None


def test_dry_run_without_models_or_consent(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    output = tmp_path / "out.png"
    write_png(source, (20, 40, 60))
    write_png(target, (0, 0, 0))
    config = load_config(write_config(tmp_path, enabled=False))
    caplog.set_level("WARNING")
    result = FaceSwapPipeline(config, admin_path=tmp_path / "missing-admin.toml").run(
        SwapRequest(source_path=source, target_path=target, output_path=output, dry_run=True)
    )
    assert result.compliance_enabled is False
    assert output.is_file()
    assert not (tmp_path / "audit.jsonl").exists()
    assert t("compliance.disabled_note") in result.notes
    assert t("compliance.disabled_note") in caplog.text


def test_real_run_without_models_explains_where_to_look(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    write_png(source)
    write_png(target)
    config = load_config(write_config(tmp_path, enabled=False))
    with pytest.raises(ModelNotFoundError) as caught:
        FaceSwapPipeline(config, admin_path=tmp_path / "missing-admin.toml").run(
            SwapRequest(
                source_path=source,
                target_path=target,
                output_path=tmp_path / "out.png",
                dry_run=False,
            )
        )
    message = str(caught.value)
    assert "det_10g.onnx" in message
    assert "inswapper_128.onnx" in message
    assert "scripts/download_models.md" in message


def test_cuda_is_preferred_and_cpu_stays_last() -> None:
    assert prefer_cuda(["cpu", "tensorrt", "cuda"]) == ["cuda", "tensorrt", "cpu"]


def test_cli_help_switches_language() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--lang", "zh-CN", "--help"])
    assert caught.value.code == 0


def test_cli_dry_run_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    output = tmp_path / "out.png"
    write_png(source)
    write_png(target)
    config = write_config(tmp_path, enabled=False)
    code = main(
        [
            "swap",
            "--source",
            str(source),
            "--target",
            str(target),
            "--output",
            str(output),
            "--config",
            str(config),
            "--dry-run",
        ]
    )
    assert code == 0
    assert output.is_file()
    captured = capsys.readouterr()
    assert "Wrote" in captured.out or str(output) in captured.out


def test_cli_missing_models_returns_4(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    write_png(source)
    write_png(target)
    code = main(
        [
            "swap",
            "--source",
            str(source),
            "--target",
            str(target),
            "--output",
            str(tmp_path / "out.png"),
            "--config",
            str(write_config(tmp_path, enabled=False)),
        ]
    )
    assert code == 4


def test_cli_chinese_help_contains_translated_description(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--lang", "zh-CN", "--help"])
    assert "合规" in capsys.readouterr().out
