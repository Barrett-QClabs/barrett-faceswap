"""Video extract, process, encode, and compliance sampling."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from barrett_faceswap.cli import main
from barrett_faceswap.compliance.content_filter import FilterDecision
from barrett_faceswap.compliance.policy import CompliancePolicy
from barrett_faceswap.config import load_config
from barrett_faceswap.errors import MediaToolError
from barrett_faceswap.media.ffmpeg import (
    build_encode_command,
    build_extract_command,
    probe,
    read_first_frame,
    sample_indexes,
)
from barrett_faceswap.pipeline import FaceSwapPipeline
from barrett_faceswap.types import SwapRequest
from barrett_faceswap.vision import read_image
from tests.conftest import write_config, write_consent, write_png


def _ffmpeg_or_skip() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg is not installed")


def _make_clip(path: Path, *, seconds: float = 0.4, size: str = "160x120", audio: bool = False) -> None:
    _ffmpeg_or_skip()
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=red:s={size}:r=10:d={seconds}",
    ]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-shortest", "-c:a", "aac"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True, capture_output=True, text=True)


def test_sample_indexes_include_the_last_frame() -> None:
    assert sample_indexes(6, 3) == [0, 3, 5]
    assert sample_indexes(4, 1) == [0, 1, 2, 3]
    assert sample_indexes(0, 30) == []


def test_nvenc_command_selects_that_encoder(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    frames = tmp_path / "frames"
    command = build_encode_command(
        frames,
        tmp_path / "out.mp4",
        fps=25,
        encoder="h264_nvenc",
        crf=23,
        audio_source=tmp_path / "in.mp4",
        audio_mode="aac",
        trim_start=1.0,
        trim_end=2.0,
    )
    assert "h264_nvenc" in command
    assert "libx264" not in command
    assert "-c:a" in command and "aac" in command
    assert "-ss" in command and "-to" in command
    extract = build_extract_command(tmp_path / "in.mp4", frames, trim_start=0.5, trim_end=None)
    assert extract[0].endswith("ffmpeg") or "ffmpeg" in extract[0]
    assert "-ss" in extract


def test_missing_ffprobe_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not-a-video")
    monkeypatch.setattr("barrett_faceswap.media.ffmpeg.shutil.which", lambda _name: None)
    with pytest.raises(MediaToolError):
        probe(video)


def test_dry_run_video_keeps_audio_and_skips_audit(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    source = tmp_path / "source.png"
    target = tmp_path / "clip.mp4"
    output = tmp_path / "out.mp4"
    write_png(source, (0, 0, 255), size=(160, 120))
    _make_clip(target, audio=True)
    config = load_config(write_config(tmp_path, enabled=False))
    config = config.model_copy(update={"video": config.video.model_copy(update={"threads": 2})})
    result = FaceSwapPipeline(config, admin_path=tmp_path / "missing.toml").run(
        SwapRequest(source_path=source, target_path=target, output_path=output, dry_run=True)
    )
    assert result.compliance_enabled is False
    assert output.is_file()
    assert not (tmp_path / "audit.jsonl").exists()
    info = probe(output)
    assert info.has_audio is True
    assert info.width > 0


def test_trim_shortens_the_clip(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    source = tmp_path / "source.png"
    target = tmp_path / "clip.mp4"
    output = tmp_path / "trimmed.mp4"
    write_png(source, size=(160, 120))
    _make_clip(target, seconds=1.0)
    original = probe(target)
    config = load_config(write_config(tmp_path, enabled=False))
    FaceSwapPipeline(config, admin_path=tmp_path / "missing.toml").run(
        SwapRequest(
            source_path=source,
            target_path=target,
            output_path=output,
            dry_run=True,
            trim_start=0.3,
            trim_end=0.6,
        )
    )
    trimmed = probe(output)
    assert trimmed.duration < original.duration


def test_compliance_samples_frames_and_watermarks_every_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ffmpeg_or_skip()
    source = tmp_path / "source.png"
    target = tmp_path / "clip.mp4"
    output = tmp_path / "marked.mp4"
    write_png(source, (0, 0, 255), size=(160, 120))
    _make_clip(target, seconds=0.4)
    write_consent(tmp_path / "consent.json")
    config = load_config(write_config(tmp_path, enabled=True))
    config = config.model_copy(
        update={"video": config.video.model_copy(update={"threads": 2, "filter_every_n_frames": 1})}
    )
    calls: list[str] = []

    class _Counter:
        name = "counter"
        supports_minor_detection = False

        def assess(self, image, *, role: str) -> FilterDecision:
            del image
            calls.append(role)
            return FilterDecision(
                allowed=True,
                provider="stub",
                reason="ok",
                minor_detection="unavailable",
            )

    monkeypatch.setattr(
        "barrett_faceswap.compliance.policy.content_filter.create_filter",
        lambda _settings: _Counter(),
    )
    befores = {"count": 0}
    original = CompliancePolicy.before_process

    def _counting(self, request, **kwargs):
        befores["count"] += 1
        return original(self, request, **kwargs)

    monkeypatch.setattr(CompliancePolicy, "before_process", _counting)
    result = FaceSwapPipeline(config, admin_path=tmp_path / "missing.toml").run(
        SwapRequest(
            source_path=source,
            target_path=target,
            output_path=output,
            dry_run=True,
            enable_compliance=True,
            consent_file=tmp_path / "consent.json",
        )
    )
    assert befores["count"] == 1
    assert result.compliance_enabled is True
    assert output.is_file()
    assert output.with_suffix(output.suffix + ".meta.json").is_file()
    records = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["event"] == "completed"
    assert records[0]["watermark_applied"] is True
    targets = [role for role in calls if role.startswith("target[")]
    outputs = [role for role in calls if role.startswith("output[")]
    assert calls[0] == "source"
    assert targets
    assert len(targets) == len(outputs)
    frame = read_first_frame(output)
    assert int(frame[-4, frame.shape[1] // 2].mean()) < 80


def test_blocked_output_sample_is_audited_and_not_encoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ffmpeg_or_skip()
    source = tmp_path / "source.png"
    target = tmp_path / "clip.mp4"
    output = tmp_path / "blocked.mp4"
    write_png(source, size=(160, 120))
    _make_clip(target, seconds=0.3)
    write_consent(tmp_path / "consent.json")
    config = load_config(write_config(tmp_path, enabled=True))

    class _BlockOutput:
        name = "block"
        supports_minor_detection = False

        def assess(self, image, *, role: str) -> FilterDecision:
            del image
            allowed = not role.startswith("output")
            return FilterDecision(
                allowed=allowed,
                provider="stub",
                reason="blocked" if not allowed else "ok",
                minor_detection="unavailable",
            )

    monkeypatch.setattr(
        "barrett_faceswap.compliance.policy.content_filter.create_filter",
        lambda _settings: _BlockOutput(),
    )
    with pytest.raises(Exception):
        FaceSwapPipeline(config, admin_path=tmp_path / "missing.toml").run(
            SwapRequest(
                source_path=source,
                target_path=target,
                output_path=output,
                dry_run=True,
                consent_file=tmp_path / "consent.json",
            )
        )
    assert not output.exists()
    records = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records[-1]["event"] == "refused"


def test_image_trim_flag_is_rejected(tmp_path: Path) -> None:
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
            "--dry-run",
            "--trim-start",
            "0.2",
        ]
    )
    assert code == 2


def test_cli_video_dry_run(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    source = tmp_path / "source.png"
    target = tmp_path / "clip.mp4"
    output = tmp_path / "out.mp4"
    write_png(source, size=(160, 120))
    _make_clip(target)
    code = main(
        [
            "--lang",
            "zh-CN",
            "swap",
            "--source",
            str(source),
            "--target",
            str(target),
            "--output",
            str(output),
            "--config",
            str(write_config(tmp_path, enabled=False)),
            "--dry-run",
            "--video-encoder",
            "libx264",
        ]
    )
    assert code == 0
    assert output.is_file()
    assert read_image(source).shape[0] == 120
