"""Gradio label switching. The server is not started."""

from __future__ import annotations

import pytest

from barrett_faceswap.i18n import get_language, set_language
from barrett_faceswap.ui.app import APP_CSS, CONTROLS, PRESETS, language_updates, resolve_preset
from tests.conftest import REPO


def test_language_dropdown_updates_every_control() -> None:
    set_language("en")
    english = language_updates("en")
    chinese = language_updates("zh-CN")
    assert get_language() == "zh-CN"
    assert len(english) == len(CONTROLS) == len(chinese)
    source_index = next(index for index, spec in enumerate(CONTROLS) if spec.key == "ui.source")
    assert english[source_index]["label"] == "Source face"
    assert chinese[source_index]["label"] == "源脸"
    steps = next(index for index, spec in enumerate(CONTROLS) if spec.key == "ui.steps")
    assert english[steps]["value"] != chinese[steps]["value"]
    run = next(index for index, spec in enumerate(CONTROLS) if spec.key == "ui.run")
    assert chinese[run]["value"] == "开始换脸"
    occluder = next(index for index, spec in enumerate(CONTROLS) if spec.key == "ui.occluder_model")
    assert english[occluder]["label"] == "Occluder"
    assert chinese[occluder]["label"] == "遮挡模型"
    media = next(index for index, spec in enumerate(CONTROLS) if spec.key == "ui.media_tab")
    assert chinese[media]["label"] == "图片·视频"
    live = next(index for index, spec in enumerate(CONTROLS) if spec.key == "ui.live_tab")
    assert english[live]["label"] == "Camera"
    assert chinese[live]["label"] == "摄像头"


def test_ui_language_prefers_the_cli_flag() -> None:
    from barrett_faceswap.ui.app import ui_language

    assert ui_language("zh-CN", "en") == "zh-CN"
    assert ui_language("en", "zh-CN") == "en"
    assert ui_language(None, "zh-CN") == "zh-CN"
    assert ui_language(None, "en") == "en"


def test_ui_command_forwards_lang(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_launch(*_args, **kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("barrett_faceswap.ui.app.launch_ui", fake_launch)
    from barrett_faceswap.cli import main

    assert main(["ui", "--lang", "zh-CN", "--port", "1"]) == 0
    assert captured["language"] == "zh-CN"


def test_still_image_run_does_not_send_the_encoder_dropdown() -> None:
    """The encoder control always has a value. A still Run must ignore it."""

    from barrett_faceswap.ui.app import video_encoder_for_run

    still = REPO / "tests"  # path suffix is what is_video checks; use a png name
    still = still / "fixture.png"
    video = REPO / "clip.mp4"
    assert video_encoder_for_run(still, "libx264", preview_only=False) is None
    assert video_encoder_for_run(still, "libx264", preview_only=True) is None
    assert video_encoder_for_run(video, "h264_nvenc", preview_only=True) is None
    assert video_encoder_for_run(video, "h264_nvenc", preview_only=False) == "h264_nvenc"


def test_quality_presets_do_not_collapse_fast_into_balanced() -> None:
    balanced = PRESETS["balanced"]
    fast = PRESETS["fast"]
    hq = PRESETS["hq"]
    assert (balanced["swapper"], balanced["pixel_boost"], balanced["enhancer"], balanced["occluder"]) == (
        fast["swapper"],
        fast["pixel_boost"],
        fast["enhancer"],
        fast["occluder"],
    )
    assert balanced["pixel_boost"] == 0
    assert balanced["enhancer"] is False
    assert balanced["occluder"] == "xseg_1"
    assert "hyperswap_1a" in str(balanced["swapper"])
    assert resolve_preset("balanced", "hyperswap_1a_256", 0, False, "xseg_1") == "balanced"
    assert resolve_preset("fast", "hyperswap_1a_256", 0, False, "xseg_1") == "fast"
    assert resolve_preset("hq", "hyperswap_1a_256", 512, True, "xseg_1") == "hq"
    assert resolve_preset("custom", "hyperswap_1a_256", 512, True, "xseg_1") == "hq"
    assert resolve_preset("balanced", "hyperswap_1a_256", 512, False, "xseg_1") == "custom"
    assert resolve_preset("hq", "hyperswap_1a_256", 0, False, "xseg_1") == "custom"
    assert hq["pixel_boost"] == 512
    assert hq["enhancer"] is True
    assert hq["blend"] == 0.35
    assert "#0B0D10" in APP_CSS
    assert "#2563EB" in APP_CSS
    assert "#14181F" in APP_CSS
    assert "#E8ECF1" in APP_CSS
    assert "#FFFFFF" in APP_CSS
    assert "button.secondary" in APP_CSS
    assert "label.float" in APP_CSS
    assert 'input[type="radio"]' in APP_CSS
    assert "sticky" in APP_CSS


def test_video_progress_hook_ignores_an_unknown_total(tmp_path) -> None:
    from barrett_faceswap.config import load_config
    from barrett_faceswap.video import VideoPipeline
    from tests.conftest import write_config

    config = load_config(write_config(tmp_path, enabled=False))
    pipeline = VideoPipeline(config)
    seen: list[tuple[int, int]] = []
    pipeline._on_progress = lambda done, total: seen.append((done, total))
    pipeline._note_progress(3, 10)
    pipeline._note_progress(1, 0)
    assert seen == [(3, 10)]
    assert pipeline._on_progress is not None
    pipeline._on_progress = None
    pipeline._note_progress(4, 10)
    assert seen == [(3, 10)]


def test_gradio_blocks_build() -> None:
    pytest.importorskip("gradio")
    from barrett_faceswap.i18n import get_language
    from barrett_faceswap.ui.app import build_demo

    demo = build_demo(REPO / "configs" / "default.yaml")
    assert demo is not None
    assert get_language() == "zh-CN"
    values = [getattr(block, "value", None) for block in demo.blocks.values()]
    assert "开始换脸" in values
    assert "均衡" in values or any(
        isinstance(getattr(block, "value", None), str) and getattr(block, "value", None) == "balanced"
        for block in demo.blocks.values()
    )
    english = build_demo(REPO / "configs" / "default.yaml", language="en")
    assert english is not None
    assert get_language() == "en"
