"""Gradio label switching. The server is not started."""

from __future__ import annotations

import pytest

from barrett_faceswap.i18n import get_language, set_language
from barrett_faceswap.ui.app import CONTROLS, language_updates
from tests.conftest import REPO


def test_language_dropdown_updates_every_control() -> None:
    set_language("en")
    english = language_updates("en")
    chinese = language_updates("zh-CN")
    assert get_language() == "zh-CN"
    assert len(english) == len(CONTROLS) == len(chinese)
    source_index = next(index for index, spec in enumerate(CONTROLS) if spec.key == "ui.source")
    assert english[source_index]["label"] == "Source"
    assert chinese[source_index]["label"] == "源图像"
    intro = next(index for index, spec in enumerate(CONTROLS) if spec.key == "ui.intro")
    assert english[intro]["value"] != chinese[intro]["value"]
    run = next(index for index, spec in enumerate(CONTROLS) if spec.key == "ui.run")
    assert chinese[run]["value"] == "运行"


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


def test_gradio_blocks_build() -> None:
    pytest.importorskip("gradio")
    from barrett_faceswap.ui.app import build_demo

    demo = build_demo(REPO / "configs" / "default.yaml")
    assert demo is not None
