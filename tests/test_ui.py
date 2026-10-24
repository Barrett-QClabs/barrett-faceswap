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
    assert english[source_index]["label"] == "Photo"
    assert chinese[source_index]["label"] == "照片"
    mode_index = next(index for index, spec in enumerate(CONTROLS) if spec.key == "ui.source_mode")
    assert english[mode_index]["label"] == "Source"
    assert chinese[mode_index]["label"] == "源"
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
    voice = next(index for index, spec in enumerate(CONTROLS) if spec.key == "ui.voice_title")
    assert english[voice]["value"] == "Voice clone"
    assert chinese[voice]["value"] == "声音克隆"


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
    assert resolve_preset("hq", "hyperswap_1a_256", 0, False, "xseg_1") == "hq"
    assert resolve_preset("custom", "hyperswap_1a_256", 0, False, "xseg_1") == "custom"
    assert resolve_preset("balanced", "hyperswap_1a_256", 512, True, "xseg_1") == "custom"
    assert resolve_preset("hq", "hyperswap_1a_256", 512, True, "xseg_1") == "custom"
    assert hq["pixel_boost"] == 0
    assert hq["enhancer"] is False
    assert "enhancer_model" not in hq
    assert (hq["swapper"], hq["pixel_boost"], hq["enhancer"], hq["occluder"]) == (
        balanced["swapper"],
        balanced["pixel_boost"],
        balanced["enhancer"],
        balanced["occluder"],
    )
    assert "#0B0D10" in APP_CSS
    assert "#1E40AF" in APP_CSS
    assert "#14181F" in APP_CSS
    assert "#FFFFFF" in APP_CSS
    assert "font-size: 18px" in APP_CSS
    assert "font-size: 17px" in APP_CSS
    assert "font-weight: 700" in APP_CSS
    assert "button.secondary" in APP_CSS
    assert "label.float" in APP_CSS
    assert "block-info" in APP_CSS
    assert "min_value" in APP_CSS
    assert "ul.options" in APP_CSS
    assert 'input[type="radio"]' in APP_CSS
    assert "sticky" in APP_CSS
    assert "bf-voice" in APP_CSS
    assert "bf-stage-split" in APP_CSS


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


def _choice_labels(demo) -> list[str]:
    labels: list[str] = []
    for block in demo.blocks.values():
        for choice in getattr(block, "choices", None) or []:
            if isinstance(choice, (list, tuple)) and choice:
                labels.append(str(choice[0]))
            elif isinstance(choice, str):
                labels.append(choice)
    return labels


def _radio_value(demo, label: str):
    for block in demo.blocks.values():
        choices = getattr(block, "choices", None) or []
        texts = [str(choice[0]) if isinstance(choice, (list, tuple)) and choice else str(choice) for choice in choices]
        if label in texts:
            return getattr(block, "value", None)
    return None


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
    labels = [getattr(block, "label", None) for block in demo.blocks.values()]
    values = [getattr(block, "value", None) for block in demo.blocks.values()]
    assert "检测阈值" in labels
    assert "声音克隆" in values
    assert "启用声音克隆" in labels
    assert _choice_labels(demo).count("视频（脸+声音，推荐）") == 2
    assert _choice_labels(demo).count("资料库") == 2
    assert _radio_value(demo, "视频（脸+声音，推荐）") == "video"
    assert "麦克风" in labels
    assert "覆盖参考音色" in labels
    assert "没有麦克风或没有参考音色，只换脸。" in values
    assert any(isinstance(item, str) and "误检" in item for item in values)
    assert any(getattr(block, "value", None) == 0.5 for block in demo.blocks.values())
    english = build_demo(REPO / "configs" / "default.yaml", language="en")
    assert english is not None
    assert get_language() == "en"
    english_labels = [getattr(block, "label", None) for block in english.blocks.values()]
    english_values = [getattr(block, "value", None) for block in english.blocks.values()]
    assert "Detect score threshold" in english_labels
    assert "Voice clone" in english_values
    assert "Enable voice clone" in english_labels
    assert _choice_labels(english).count("Video (face + voice, recommended)") == 2
    assert _choice_labels(english).count("Library") == 2
    assert "Microphone" in english_labels
    assert "Override reference" in english_labels
    hidden = [
        block
        for block in demo.blocks.values()
        if getattr(block, "label", None) in {"照片", "参考音频"}
    ]
    assert hidden
    assert all(getattr(block, "visible", True) is False for block in hidden)
    video_blocks = [block for block in demo.blocks.values() if getattr(block, "label", None) == "源视频"]
    assert len(video_blocks) == 2
    assert all(getattr(block, "visible", True) is not False for block in video_blocks)
    overrides = [block for block in demo.blocks.values() if getattr(block, "label", None) == "覆盖参考音色"]
    assert overrides
    assert all(getattr(block, "visible", True) is not False for block in overrides)
    library_blocks = []
    for block in demo.blocks.values():
        label = getattr(block, "label", None)
        value = getattr(block, "value", None)
        if label in {"资料", "新名字", "样本", "样本列表", "资料库"}:
            library_blocks.append(block)
        elif isinstance(value, str) and value in {"新建", "加入样本", "移除样本", "删除资料"}:
            library_blocks.append(block)
    assert library_blocks
    assert all(getattr(block, "visible", True) is False for block in library_blocks)
    assert "识别脸" in values
    face_one = [block for block in demo.blocks.values() if getattr(block, "label", None) == "脸 1"]
    assert face_one
    assert all(getattr(block, "visible", True) is False for block in face_one)
    assert "Find faces" in english_values
    english_face = [block for block in english.blocks.values() if getattr(block, "label", None) == "Face 1"]
    assert english_face
    assert all(getattr(block, "visible", True) is False for block in english_face)
    assert any(isinstance(item, str) and "false positives" in item for item in english_values)
