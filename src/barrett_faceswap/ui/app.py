"""Gradio shell. Labels come from the catalogs and switch with the header toggle.

Importing this module does not import Gradio. ``build_demo`` and ``launch_ui``
import it, and they fail with an install hint when the extra is missing.
"""

from __future__ import annotations

import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from barrett_faceswap.config import (
    ENHANCER_MODELS,
    EXECUTION_PROVIDERS,
    VIDEO_ENCODERS,
    AppConfig,
    apply_request_overrides,
    load_config,
)
from barrett_faceswap.errors import BarrettFaceswapError
from barrett_faceswap.i18n import LanguageError, get_language, normalize_language, set_language, t
from barrett_faceswap.media.ffmpeg import VIDEO_OUTPUT_SUFFIXES, read_first_frame
from barrett_faceswap.pipeline import FaceSwapPipeline
from barrett_faceswap.sourceprep import reference_for_mode, resolve_live_reference, source_mode_visibility
from barrett_faceswap.voice import target_has_audio, voice_toggle_for_target
from barrett_faceswap.swap_catalog import (
    BLEND_MODES,
    COLOR_TRANSFERS,
    EYE_MODES,
    OCCLUDER_MODELS,
    SWAPPER_NAMES,
    VIDEO_MODES,
)
from barrett_faceswap.types import SwapRequest
from barrett_faceswap.vision import is_video, read_image, write_image

# Page colors. Both theme modes use the same dark values so the app stays dark.
_BG = "#0B0D10"
_CARD = "#14181F"
_LINE = "#1E2430"
_TEXT = "#FFFFFF"
_INK = "#0B0D10"
_PAPER = "#FFFFFF"
# Darker than #2563EB so white labels stay solid (about 8.7:1) instead of looking washed out.
_CTA = "#1E40AF"
_CTA_HOVER = "#1D4ED8"

APP_CSS = f"""
.gradio-container {{
  background: {_BG} !important;
  color: {_TEXT};
  font-size: 17px !important;
  font-family: "PingFang SC", "Noto Sans SC", system-ui, sans-serif !important;
}}
.gradio-container .block {{
  background: {_CARD};
  border-color: {_LINE};
}}
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .md {{
  font-size: 17px !important;
  line-height: 1.5 !important;
  color: {_TEXT} !important;
}}
.gradio-container [data-testid="block-info"],
.gradio-container [data-testid="block-label"] {{
  color: {_TEXT} !important;
  font-size: 18px !important;
  font-weight: 700 !important;
  background: transparent !important;
}}
.gradio-container .min_value,
.gradio-container .max_value {{
  color: {_TEXT} !important;
  font-size: 16px !important;
  font-weight: 700 !important;
}}
.gradio-container .secondary-wrap input,
.gradio-container .secondary-wrap input:disabled {{
  color: {_TEXT} !important;
  -webkit-text-fill-color: {_TEXT} !important;
  opacity: 1 !important;
  font-size: 17px !important;
  font-weight: 650 !important;
  background: transparent !important;
}}
.gradio-container .tab-like-container {{
  height: auto !important;
  min-height: 40px !important;
}}
.gradio-container .tab-like-container input[type="number"] {{
  min-height: 40px !important;
  font-size: 17px !important;
  font-weight: 700 !important;
}}
.gradio-container ul.options,
.gradio-container ul.options li,
ul.options,
ul.options li {{
  background: {_BG} !important;
  color: {_TEXT} !important;
  font-size: 17px !important;
  font-weight: 600 !important;
}}
.gradio-container ul.options li:hover,
.gradio-container ul.options li.active,
.gradio-container ul.options li.selected,
ul.options li:hover,
ul.options li.active,
ul.options li.selected {{
  background: {_CTA} !important;
  color: {_PAPER} !important;
}}
.gradio-container .tab-container button {{
  font-size: 17px !important;
  font-weight: 700 !important;
  color: {_TEXT} !important;
}}
.gradio-container .tab-container button.selected {{
  color: {_PAPER} !important;
}}
.gradio-container button.label-wrap {{
  font-size: 17px !important;
  font-weight: 700 !important;
  color: {_PAPER} !important;
}}
footer {{ display: none !important; }}
.bf-layout {{ align-items: stretch; gap: 16px; }}
.bf-stage-split {{ align-items: stretch; gap: 16px; }}
.bf-preview {{ min-height: 420px; }}
.bf-voice {{
  border: 1px solid {_LINE};
  border-radius: 14px;
  padding: 4px 8px 8px;
  min-width: 260px;
}}
.bf-source-mode fieldset,
.bf-source-mode .wrap {{
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 8px !important;
}}
.bf-source-mode label:has(input[type="radio"]) {{
  flex: 1 1 140px !important;
  justify-content: center !important;
  text-align: center !important;
  min-height: 52px !important;
}}
.gradio-container button.primary,
.gradio-container .bf-cta {{
  min-height: 48px;
  background: {_CTA} !important;
  color: {_PAPER} !important;
  border-color: {_CTA} !important;
  font-size: 18px !important;
  font-weight: 700 !important;
}}
.gradio-container button.primary:hover,
.gradio-container .bf-cta:hover {{
  background: {_CTA_HOVER} !important;
  color: {_PAPER} !important;
}}
.gradio-container button.primary *,
.gradio-container .bf-cta * {{
  color: {_PAPER} !important;
  font-size: 18px !important;
  font-weight: 700 !important;
}}
.gradio-container button.secondary {{
  min-height: 48px;
  background: {_PAPER} !important;
  color: {_INK} !important;
  border: 1px solid {_INK} !important;
  font-size: 18px !important;
  font-weight: 700 !important;
}}
.gradio-container button.secondary:hover {{
  background: {_TEXT} !important;
  color: {_INK} !important;
}}
.gradio-container button.secondary * {{
  color: {_INK} !important;
  font-size: 18px !important;
  font-weight: 700 !important;
}}
.gradio-container label.float,
.gradio-container label.float * {{
  background: transparent !important;
  color: {_TEXT} !important;
  -webkit-text-fill-color: {_TEXT} !important;
  border-color: transparent !important;
  box-shadow: none !important;
  font-size: 18px !important;
  font-weight: 700 !important;
  opacity: 1 !important;
}}
.gradio-container button.boundedheight,
.gradio-container button.boundedheight * {{
  color: {_PAPER} !important;
  font-size: 17px !important;
  font-weight: 650 !important;
}}
.gradio-container label:has(input[type="radio"]) {{
  background: {_PAPER} !important;
  color: {_INK} !important;
  -webkit-text-fill-color: {_INK} !important;
  border: 1px solid {_INK} !important;
  font-size: 16px !important;
  font-weight: 700 !important;
  padding: 8px 14px !important;
}}
.gradio-container label:has(input[type="radio"]) * {{
  color: {_INK} !important;
  -webkit-text-fill-color: {_INK} !important;
  font-size: 16px !important;
  font-weight: 700 !important;
}}
.gradio-container label:has(input[type="radio"]):has(input:checked) {{
  background: {_CTA} !important;
  color: {_PAPER} !important;
  -webkit-text-fill-color: {_PAPER} !important;
  border-color: {_CTA} !important;
}}
.gradio-container label:has(input[type="radio"]):has(input:checked) * {{
  color: {_PAPER} !important;
  -webkit-text-fill-color: {_PAPER} !important;
}}
.gradio-container label:has(input[type="checkbox"]) {{
  background: transparent !important;
  color: {_PAPER} !important;
  -webkit-text-fill-color: {_PAPER} !important;
  border-color: transparent !important;
  font-size: 16px !important;
  font-weight: 650 !important;
}}
.gradio-container label:has(input[type="checkbox"]) * {{
  color: {_PAPER} !important;
  -webkit-text-fill-color: {_PAPER} !important;
}}
.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container select {{
  background: {_BG} !important;
  color: {_PAPER} !important;
  -webkit-text-fill-color: {_PAPER} !important;
  font-size: 17px !important;
  font-weight: 650 !important;
  opacity: 1 !important;
}}
.bf-actions button:disabled {{ opacity: 0.4; }}
@media (max-width: 959px) {{
  .bf-layout {{ flex-direction: column !important; }}
  .bf-stage-split {{ flex-direction: column !important; }}
  .bf-stage {{ order: -1; }}
  .bf-actions {{
    position: sticky;
    bottom: 0;
    z-index: 20;
    background: {_BG};
    padding: 8px 0 12px;
  }}
}}
"""

# Fast, Balanced, and High share the open-app stack: HyperSwap, enhancer off,
# xseg_1, pixel boost 0. resolve_preset keeps the current name when it is one
# of the matches, so the radios do not collapse into each other. High must not
# enable GFPGAN or CodeFormer. Expression and landmark refine stay on the
# config, which the page loads before these widgets.
PRESETS: dict[str, dict[str, object]] = {
    "fast": {
        "swapper": "hyperswap_1a_256",
        "pixel_boost": 0,
        "enhancer": False,
        "occluder": "xseg_1",
    },
    "balanced": {
        "swapper": "hyperswap_1a_256",
        "pixel_boost": 0,
        "enhancer": False,
        "occluder": "xseg_1",
    },
    "hq": {
        "swapper": "hyperswap_1a_256",
        "pixel_boost": 0,
        "enhancer": False,
        "occluder": "xseg_1",
    },
}


def _preset_signature(spec: dict[str, object]) -> tuple[str, int, bool, str]:
    return (
        str(spec["swapper"]),
        int(spec["pixel_boost"]),
        bool(spec["enhancer"]),
        str(spec["occluder"]),
    )


def resolve_preset(
    current: str,
    swapper: str,
    pixel_boost: int,
    enhancer_on: bool,
    occluder: str,
) -> str:
    """Pick a preset name from the widgets.

    The widgets stay the source of truth. Fast, Balanced, and High are the
    same numbers, so a current match is kept. Several matches that do not
    include the current name become Custom instead of snapping to one of them.
    One match selects that preset.
    """

    signature = (str(swapper), int(pixel_boost), bool(enhancer_on), str(occluder))
    hits = [name for name, spec in PRESETS.items() if _preset_signature(spec) == signature]
    if current in hits:
        return current
    if len(hits) == 1:
        return hits[0]
    return "custom"


@dataclass(frozen=True)
class ControlSpec:
    """One Gradio control whose caption is a catalog key."""

    key: str
    prop: str


# Order matches the components created in :func:`build_demo`.
CONTROLS: tuple[ControlSpec, ...] = (
    ControlSpec("ui.brand", "value"),
    ControlSpec("ui.language", "label"),
    ControlSpec("ui.media_tab", "label"),
    ControlSpec("ui.steps", "value"),
    ControlSpec("ui.source_mode", "label"),
    ControlSpec("ui.source_video", "label"),
    ControlSpec("ui.source", "label"),
    ControlSpec("ui.source_audio", "label"),
    ControlSpec("ui.target", "label"),
    ControlSpec("ui.swapper_model", "label"),
    ControlSpec("ui.swapper_hint", "value"),
    ControlSpec("ui.quality", "label"),
    ControlSpec("ui.advanced", "label"),
    ControlSpec("ui.detect_score", "label"),
    ControlSpec("ui.detect_score_hint", "value"),
    ControlSpec("ui.face_select", "label"),
    ControlSpec("ui.many_faces", "label"),
    ControlSpec("ui.reference", "label"),
    ControlSpec("ui.face_map", "label"),
    ControlSpec("ui.target_index", "label"),
    ControlSpec("ui.enhancer_group", "label"),
    ControlSpec("ui.enhancer_enable", "label"),
    ControlSpec("ui.enhancer", "label"),
    ControlSpec("ui.enhancer_blend", "label"),
    ControlSpec("ui.enhancer_weight", "label"),
    ControlSpec("ui.mask_group", "label"),
    ControlSpec("ui.occluder_model", "label"),
    ControlSpec("ui.pixel_boost", "label"),
    ControlSpec("ui.blend_mode", "label"),
    ControlSpec("ui.color_transfer", "label"),
    ControlSpec("ui.eye_mode", "label"),
    ControlSpec("ui.video_group", "label"),
    ControlSpec("ui.video_mode", "label"),
    ControlSpec("ui.video_encoder", "label"),
    ControlSpec("ui.trim_start", "label"),
    ControlSpec("ui.trim_end", "label"),
    ControlSpec("ui.keep_audio", "label"),
    ControlSpec("ui.execution_group", "label"),
    ControlSpec("ui.execution", "label"),
    ControlSpec("ui.dry_run", "label"),
    ControlSpec("ui.compliance_group", "label"),
    ControlSpec("ui.compliance", "label"),
    ControlSpec("ui.commercial", "label"),
    ControlSpec("ui.consent", "label"),
    ControlSpec("ui.consent_subject", "label"),
    ControlSpec("ui.output_group", "label"),
    ControlSpec("ui.output", "label"),
    ControlSpec("ui.preview", "value"),
    ControlSpec("ui.run", "value"),
    ControlSpec("ui.preview_image", "label"),
    ControlSpec("ui.voice_title", "value"),
    ControlSpec("ui.voice_enable", "label"),
    ControlSpec("ui.voice_reference", "label"),
    ControlSpec("ui.voice_hint", "value"),
    ControlSpec("ui.voice_advanced", "label"),
    ControlSpec("ui.voice_strength", "label"),
    ControlSpec("ui.voice_tau", "label"),
    ControlSpec("ui.voice_status", "label"),
    ControlSpec("ui.progress_label", "label"),
    ControlSpec("ui.status", "label"),
    ControlSpec("ui.live_tab", "label"),
    ControlSpec("ui.live_intro", "value"),
    ControlSpec("ui.source_mode", "label"),
    ControlSpec("ui.source_video", "label"),
    ControlSpec("ui.source", "label"),
    ControlSpec("ui.source_audio", "label"),
    ControlSpec("ui.live_voice_override", "label"),
    ControlSpec("ui.live_start", "value"),
    ControlSpec("ui.live_stop", "value"),
    ControlSpec("ui.live_status", "label"),
    ControlSpec("ui.live_voice", "label"),
    ControlSpec("ui.advanced", "label"),
    ControlSpec("ui.dry_run", "label"),
    ControlSpec("ui.compliance", "label"),
    ControlSpec("ui.live_camera", "label"),
    ControlSpec("ui.live_mic", "label"),
    ControlSpec("ui.live_speaker", "label"),
    ControlSpec("ui.live_preview", "label"),
)


def outcome_text(result: object, fallback_key: str) -> str:
    """Status line after a run. Voice outcomes replace the plain wrote-path line."""

    status = str(getattr(result, "voice_status", "off") or "off")
    key = {
        "converted": "ui.done_voice",
        "no_audio": "ui.done_no_audio",
        "no_reference": "ui.done_no_reference",
        "dry_run": "ui.done_dry_voice",
        "strength_zero": "ui.done_strength_zero",
    }.get(status, fallback_key)
    return t(key, path=getattr(result, "output_path", ""))


def video_encoder_for_run(target: Path, encoder: str | None, *, preview_only: bool) -> str | None:
    """Return an encoder only for a full video run.

    The dropdown always has a selection. Passing it for a still image raises
    the same error as ``--video-encoder`` on the CLI.
    """

    if preview_only or not is_video(target):
        return None
    return encoder


def language_updates(language: str) -> list[dict[str, str]]:
    """Switch the active catalog and return caption updates in ``CONTROLS`` order."""

    normalized = normalize_language(language)
    if normalized is None:
        raise LanguageError(t("config.bad_language", value=language))
    set_language(normalized)
    return [{spec.prop: t(spec.key)} for spec in CONTROLS]


def ui_language(cli: str | None, config_language: str) -> str:
    """Prefer ``--lang`` over the config language for the first set of labels."""

    if cli:
        normalized = normalize_language(cli)
        if normalized:
            return normalized
    return normalize_language(config_language) or "zh-CN"


def launch_ui(
    config_path: str | Path,
    *,
    host: str,
    port: int,
    share: bool = False,
    language: str | None = None,
) -> None:
    """Build the Blocks app and serve it."""

    demo = build_demo(config_path, language=language)
    demo.launch(server_name=host, server_port=int(port), share=share)


def _source_mode_choices() -> list[tuple[str, str]]:
    return [
        (t("ui.source_mode_video"), "video"),
        (t("ui.source_mode_photo"), "photo"),
        (t("ui.source_mode_photo_audio"), "photo_audio"),
    ]


def _voice_hint_key(mode: str | None) -> str:
    return {
        "photo": "ui.voice_hint_photo",
        "photo_audio": "ui.voice_hint_audio",
    }.get(str(mode or "video"), "ui.voice_hint_video")


def source_files_ready(mode: str | None, video: object, photo: object, audio: object, target: object) -> bool:
    """True when the uploads for the selected source mode, and a target, are set."""

    if _as_path(target) is None:
        return False
    flags = source_mode_visibility(mode)
    if flags["video"]:
        return _as_path(video) is not None
    if _as_path(photo) is None:
        return False
    if flags["audio"]:
        return _as_path(audio) is not None
    return True


def _quality_choices() -> list[tuple[str, str]]:
    return [
        (t("ui.quality_fast"), "fast"),
        (t("ui.quality_balanced"), "balanced"),
        (t("ui.quality_hq"), "hq"),
        (t("ui.quality_custom"), "custom"),
    ]


def _language_choices() -> list[tuple[str, str]]:
    return [(t("ui.lang_en"), "en"), (t("ui.lang_zh"), "zh-CN")]


def _dark_theme(gr):
    return (
        gr.themes.Soft(primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate)
        .set(
            body_background_fill=_BG,
            body_background_fill_dark=_BG,
            background_fill_primary=_CARD,
            background_fill_primary_dark=_CARD,
            background_fill_secondary=_BG,
            background_fill_secondary_dark=_BG,
            block_background_fill=_CARD,
            block_background_fill_dark=_CARD,
            block_border_color=_LINE,
            block_border_color_dark=_LINE,
            block_label_text_color=_TEXT,
            block_label_text_color_dark=_TEXT,
            block_title_text_color=_TEXT,
            block_title_text_color_dark=_TEXT,
            body_text_color=_TEXT,
            body_text_color_dark=_TEXT,
            body_text_color_subdued=_TEXT,
            body_text_color_subdued_dark=_TEXT,
            input_placeholder_color=_TEXT,
            input_placeholder_color_dark=_TEXT,
            block_title_background_fill="transparent",
            block_title_background_fill_dark="transparent",
            body_text_size="17px",
            body_text_weight="600",
            block_label_text_size="16px",
            block_label_text_weight="700",
            block_title_text_size="18px",
            block_title_text_weight="700",
            section_header_text_size="17px",
            section_header_text_weight="700",
            prose_text_size="17px",
            input_text_size="17px",
            button_large_text_size="18px",
            button_large_text_weight="700",
            button_large_padding="12px 18px",
            button_medium_text_size="17px",
            button_medium_text_weight="700",
            button_small_text_size="16px",
            button_small_text_weight="700",
            checkbox_label_background_fill=_PAPER,
            checkbox_label_background_fill_dark=_PAPER,
            checkbox_label_background_fill_hover=_TEXT,
            checkbox_label_background_fill_hover_dark=_TEXT,
            checkbox_label_background_fill_selected=_CTA,
            checkbox_label_background_fill_selected_dark=_CTA,
            checkbox_label_border_color=_INK,
            checkbox_label_border_color_dark=_INK,
            checkbox_label_border_color_selected=_CTA,
            checkbox_label_border_color_selected_dark=_CTA,
            checkbox_label_text_color=_INK,
            checkbox_label_text_color_dark=_INK,
            checkbox_label_text_color_selected=_PAPER,
            checkbox_label_text_color_selected_dark=_PAPER,
            checkbox_label_text_size="16px",
            checkbox_label_text_weight="700",
            checkbox_label_padding="8px 14px",
            button_primary_background_fill=_CTA,
            button_primary_background_fill_dark=_CTA,
            button_primary_background_fill_hover=_CTA_HOVER,
            button_primary_background_fill_hover_dark=_CTA_HOVER,
            button_primary_text_color=_PAPER,
            button_primary_text_color_dark=_PAPER,
            button_primary_text_color_hover=_PAPER,
            button_primary_text_color_hover_dark=_PAPER,
            button_secondary_background_fill=_PAPER,
            button_secondary_background_fill_dark=_PAPER,
            button_secondary_background_fill_hover=_TEXT,
            button_secondary_background_fill_hover_dark=_TEXT,
            button_secondary_border_color=_INK,
            button_secondary_border_color_dark=_INK,
            button_secondary_border_color_hover=_INK,
            button_secondary_border_color_hover_dark=_INK,
            button_secondary_text_color=_INK,
            button_secondary_text_color_dark=_INK,
            button_secondary_text_color_hover=_INK,
            button_secondary_text_color_hover_dark=_INK,
            block_label_background_fill=_CARD,
            block_label_background_fill_dark=_CARD,
            color_accent=_CTA,
            color_accent_soft=_LINE,
            color_accent_soft_dark=_LINE,
            input_background_fill=_BG,
            input_background_fill_dark=_BG,
            input_border_color=_LINE,
            input_border_color_dark=_LINE,
            input_border_color_focus=_CTA_HOVER,
            input_border_color_focus_dark=_CTA_HOVER,
            border_color_accent=_CTA,
            border_color_accent_dark=_CTA,
            slider_color=_CTA,
            slider_color_dark=_CTA,
            panel_background_fill=_CARD,
            panel_background_fill_dark=_CARD,
            block_radius="14px",
            button_large_radius="8px",
        )
    )


def build_demo(config_path: str | Path, language: str | None = None):
    """Construct the Gradio Blocks app. Gradio is imported here."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise BarrettFaceswapError(t("ui.gradio_missing")) from exc

    config_file = Path(config_path)
    initial = load_config(config_file)
    active_language = ui_language(language, initial.ui.language)
    set_language(active_language)
    enhancer_default = "face_enhancer" in initial.processors
    initial_quality = resolve_preset(
        "balanced",
        initial.swap.model,
        int(initial.swap.pixel_boost),
        enhancer_default,
        initial.masks.occluder_model,
    )

    def on_language(selected: str, source_mode: str):
        updates = language_updates(selected)
        wrapped = []
        for spec, payload in zip(CONTROLS, updates, strict=True):
            data = dict(payload)
            if spec.key == "ui.language":
                data["choices"] = _language_choices()
            elif spec.key == "ui.quality":
                data["choices"] = _quality_choices()
            elif spec.key == "ui.source_mode":
                data["choices"] = _source_mode_choices()
            elif spec.key == "ui.voice_hint":
                data["value"] = t(_voice_hint_key(source_mode))
            wrapped.append(gr.update(**data))
        return wrapped

    def apply_preset(name, swapper, boost, enhancer_on, model, blend, weight, occluder):
        spec = PRESETS.get(str(name))
        if spec is None:
            return (gr.skip(),) * 7

        def keep(new, old, numeric: bool = False):
            if numeric:
                same = round(float(new), 4) == round(float(old), 4)
            else:
                same = new == old
            return gr.skip() if same else gr.update(value=new)

        model_update = gr.skip()
        blend_update = gr.skip()
        weight_update = gr.skip()
        if "enhancer_model" in spec:
            model_update = keep(spec["enhancer_model"], model)
            blend_update = keep(spec["blend"], blend, numeric=True)
            weight_update = keep(spec["weight"], weight, numeric=True)
        return (
            keep(spec["swapper"], swapper),
            keep(int(spec["pixel_boost"]), int(boost)),
            keep(bool(spec["enhancer"]), bool(enhancer_on)),
            model_update,
            blend_update,
            weight_update,
            keep(spec["occluder"], occluder),
        )

    def sync_preset(current, swapper, boost, enhancer_on, occluder):
        nxt = resolve_preset(str(current), str(swapper), int(boost), bool(enhancer_on), str(occluder))
        if nxt == current:
            return gr.skip()
        return gr.update(value=nxt)

    def _voice_for_mode(mode: str | None, target: Path | None) -> tuple[bool, str]:
        if mode == "photo":
            return False, t("ui.voice_hint_photo")
        enabled, message = voice_toggle_for_target(target, auto_enable=bool(initial.voice.enabled))
        if mode == "photo_audio" and enabled:
            message = t("ui.voice_ready_audio")
        return enabled, message

    def _files_ready(mode, video, photo, audio, target):
        ready = source_files_ready(mode, video, photo, audio, target)
        return gr.update(interactive=ready), gr.update(interactive=ready)

    def _target_changed(mode, video, photo, audio, target):
        preview_button_update, run_button_update = _files_ready(mode, video, photo, audio, target)
        enabled, message = _voice_for_mode(mode, _as_path(target))
        return preview_button_update, run_button_update, gr.update(value=enabled), message

    def _mode_view(mode, video, photo, audio, target):
        flags = source_mode_visibility(mode)
        preview_button_update, run_button_update = _files_ready(mode, video, photo, audio, target)
        enabled, message = _voice_for_mode(mode, _as_path(target))
        return (
            gr.update(visible=flags["video"]),
            gr.update(visible=flags["photo"]),
            gr.update(visible=flags["audio"]),
            gr.update(visible=flags["voice_override"]),
            t(_voice_hint_key(mode)),
            gr.update(value=enabled),
            message,
            preview_button_update,
            run_button_update,
        )

    def _preview_array(path: Path):
        image = read_first_frame(path) if is_video(path) else read_image(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()

    def _stream_video(request: SwapRequest, status_key: str, progress, *, voice_planned: bool):
        box: dict[str, object] = {"done": 0, "total": 0, "result": None, "error": None, "stage": "face"}

        def hook(done: int, total: int) -> None:
            box["done"] = int(done)
            box["total"] = int(total)
            if box.get("stage") in {None, "", "face"}:
                box["stage"] = "face"

        def stage(name: str) -> None:
            box["stage"] = name

        def worker() -> None:
            try:
                config = apply_request_overrides(load_config(config_file), request)
                box["result"] = FaceSwapPipeline(config).run(request, on_progress=hook, on_stage=stage)
            except Exception as exc:  # noqa: BLE001 - re-raised on the handler thread
                box["error"] = exc

        def lines(done: int, total: int, stage_name: str) -> tuple[str, float, str]:
            if stage_name == "voice":
                return t("ui.progress_voice"), 0.9, t("ui.progress_voice")
            if stage_name == "mux":
                return t("ui.progress_mux"), 0.97, t("ui.progress_mux")
            if total and voice_planned:
                message = t("ui.progress_face", done=done, total=total)
                return message, 0.85 * (done / total), message
            if total:
                message = t("ui.progress", done=done, total=total)
                return message, done / total, message
            waiting = t("ui.waiting")
            return waiting, 0, waiting if voice_planned else ""

        thread = threading.Thread(target=worker, name="barrett-ui-video", daemon=True)
        thread.start()
        while thread.is_alive():
            done = int(box["done"])
            total = int(box["total"])
            stage_name = str(box.get("stage") or "face")
            message, fraction, voice_line = lines(done, total, stage_name)
            if stage_name == "voice":
                progress(0.9, desc=message)
            elif stage_name == "mux":
                progress(0.97, desc=message)
            elif total:
                progress((done, total), desc=message)
            else:
                progress(0, desc=message)
            voice_update = voice_line if voice_planned or stage_name in {"voice", "mux"} else gr.skip()
            yield gr.skip(), message, fraction, voice_update
            thread.join(0.4)
        error = box["error"]
        if error is not None:
            raise error
        result = box["result"]
        done = int(box["done"]) or 1
        total = int(box["total"]) or 1
        finished = outcome_text(result, status_key)
        voice_final = finished if getattr(result, "voice_status", "off") != "off" else gr.skip()
        progress((total, total), desc=finished)
        yield _preview_array(result.output_path), finished, 1, voice_final

    def on_execute(
        selected_language: str,
        source_mode: str,
        source_video: object,
        source_photo: object,
        source_audio: object,
        target: object,
        output_text: str,
        swapper_model: str,
        pixel_boost: int,
        enhancer_on: bool,
        enhancer_model: str,
        blend: float,
        weight: float,
        occluder_model: str,
        many_faces: bool,
        reference: object,
        face_map: object,
        target_index: float,
        detect_score: float,
        blend_mode: str,
        color_transfer: str,
        eye_mode: str,
        video_mode: str,
        encoder: str,
        trim_start_text: str,
        trim_end_text: str,
        keep_audio: bool,
        providers: list[str] | None,
        dry_run: bool,
        compliance: bool,
        commercial: bool,
        consent: object,
        subject: str,
        voice_on: bool,
        voice_clip: object,
        voice_strength_value: float,
        voice_tau_value: float,
        preview_only: bool,
        progress=gr.Progress(track_tqdm=False),
    ):
        try:
            language_updates(selected_language or get_language())
            mode = source_mode if source_mode in {"video", "photo", "photo_audio"} else "video"
            flags = source_mode_visibility(mode)
            source_path = _as_path(source_video) if flags["video"] else _as_path(source_photo)
            target_path = _as_path(target)
            if source_path is None or target_path is None:
                yield None, t("ui.need_files"), 0, t("ui.voice_idle")
                return
            if flags["video"] and not is_video(source_path):
                yield None, t("ui.need_source_video"), 0, t("ui.voice_idle")
                return
            if not flags["video"] and is_video(source_path):
                yield None, t("ui.need_source_photo"), 0, t("ui.voice_idle")
                return
            mode_audio = _as_path(source_audio) if flags["audio"] else None
            if flags["audio"] and mode_audio is None:
                yield None, t("ui.need_source_audio"), 0, t("ui.voice_idle")
                return
            processors = ["face_swapper"]
            if enhancer_on:
                processors.append("face_enhancer")
            if not processors:
                yield None, t("ui.need_processor"), 0, t("ui.voice_idle")
                return
            with tempfile.TemporaryDirectory(prefix="barrett-faceswap-ui-") as directory:
                temp = Path(directory)
                status_key = "ui.done"
                request_target = target_path
                video_encoder = video_encoder_for_run(target_path, encoder, preview_only=preview_only)
                selected_video_mode = video_mode if is_video(target_path) and not preview_only else None
                trim_start = _seconds(trim_start_text)
                trim_end = _seconds(trim_end_text)
                if preview_only:
                    trim_start = None
                    trim_end = None
                    if is_video(target_path):
                        write_image(temp / "frame.png", read_first_frame(target_path))
                        request_target = temp / "frame.png"
                        status_key = "ui.preview_video"
                    else:
                        status_key = "ui.preview_still"
                    output = temp / "preview.png"
                else:
                    output = _as_path(output_text) or Path("outputs/ui-output.png")
                    if is_video(target_path) and output.suffix.lower() not in VIDEO_OUTPUT_SUFFIXES:
                        output = output.with_suffix(".mp4")
                consent_path = _as_path(consent)
                subject_text = (subject or "").strip()
                request = SwapRequest(
                    source_path=source_path,
                    target_path=request_target,
                    output_path=output,
                    dry_run=bool(dry_run),
                    enable_compliance=bool(compliance),
                    consent_file=consent_path,
                    acknowledge_authorized_use=bool(subject_text) and consent_path is None,
                    consent_subject=subject_text or None,
                    processors=list(processors),
                    execution_providers=list(providers) if providers else None,
                    trim_start=trim_start,
                    trim_end=trim_end,
                    video_encoder=video_encoder,
                    keep_audio=bool(keep_audio),
                    enhancer_model=enhancer_model,
                    enhancer_blend=float(blend),
                    enhancer_weight=float(weight),
                    swapper_model=swapper_model,
                    occluder_model=occluder_model,
                    pixel_boost=int(pixel_boost),
                    many_faces=bool(many_faces),
                    reference_path=_as_path(reference),
                    face_map=_as_path(face_map),
                    blend_mode=blend_mode,
                    color_transfer=color_transfer,
                    eye_mode=eye_mode,
                    video_mode=selected_video_mode,
                    commercial=bool(commercial),
                    target_face_index=int(target_index or 0),
                    with_enhancer=bool(enhancer_on),
                    detect_score=float(detect_score),
                    voice_enabled=False if mode == "photo" or preview_only else bool(voice_on),
                    voice_reference=_voice_reference_for_mode(
                        mode,
                        mode_audio,
                        None if preview_only else _as_path(voice_clip),
                    ),
                    voice_strength=float(voice_strength_value),
                    voice_tau=float(voice_tau_value),
                )
                full_video = is_video(target_path) and not preview_only
                will_reference = request.voice_reference is not None or (
                    mode == "video" and bool(request.voice_enabled)
                )
                voice_planned = (
                    full_video
                    and bool(request.voice_enabled)
                    and will_reference
                    and not dry_run
                    and target_has_audio(target_path)
                )
                if full_video:
                    yield from _stream_video(request, status_key, progress, voice_planned=voice_planned)
                    return
                progress(0, desc=t("ui.progress_running"))
                config = apply_request_overrides(load_config(config_file), request)
                result = FaceSwapPipeline(config).run(request)
                progress(1, desc=t("ui.progress_running"))
                text = t(status_key, path=result.output_path) if preview_only else outcome_text(result, status_key)
                voice_final = gr.skip() if preview_only or result.voice_status == "off" else text
                yield _preview_array(result.output_path), text, 1, voice_final
        except (BarrettFaceswapError, ValidationError, LanguageError) as exc:
            yield None, str(exc), 0, str(exc)

    def on_preview(*args, progress=gr.Progress(track_tqdm=False)):
        yield from on_execute(*args, True, progress)

    def on_run(*args, progress=gr.Progress(track_tqdm=False)):
        yield from on_execute(*args, False, progress)

    with gr.Blocks(title=t("ui.title"), theme=_dark_theme(gr), css=APP_CSS, fill_height=True) as demo:
        with gr.Row(elem_classes=["bf-header"]):
            brand = gr.Markdown(t("ui.brand"))
            language = gr.Radio(
                choices=_language_choices(),
                value=active_language,
                label=t("ui.language"),
            )
        with gr.Tabs():
            with gr.Tab(t("ui.media_tab")) as media_tab:
                with gr.Row(elem_classes=["bf-layout"]):
                    with gr.Column(scale=38, elem_classes=["bf-controls"]):
                        steps = gr.Markdown(t("ui.steps"))
                        source_mode = gr.Radio(
                            choices=_source_mode_choices(),
                            value="video",
                            label=t("ui.source_mode"),
                            elem_classes=["bf-source-mode"],
                        )
                        source_video = gr.File(
                            type="filepath",
                            file_count="single",
                            file_types=[".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"],
                            label=t("ui.source_video"),
                        )
                        source_photo = gr.Image(
                            type="filepath",
                            sources=["upload", "clipboard"],
                            label=t("ui.source"),
                            visible=False,
                        )
                        source_audio = gr.Audio(
                            type="filepath",
                            sources=["upload"],
                            label=t("ui.source_audio"),
                            visible=False,
                        )
                        target = gr.File(type="filepath", label=t("ui.target"), file_count="single")
                        swapper_model = gr.Dropdown(
                            choices=list(SWAPPER_NAMES),
                            value=initial.swap.model,
                            label=t("ui.swapper_model"),
                        )
                        swapper_hint = gr.Markdown(t("ui.swapper_hint"))
                        quality = gr.Radio(
                            choices=_quality_choices(),
                            value=initial_quality,
                            label=t("ui.quality"),
                        )
                        with gr.Accordion(t("ui.advanced"), open=False) as advanced:
                            detect_score = gr.Slider(
                                0,
                                1,
                                value=float(initial.detect.score_threshold),
                                step=0.01,
                                label=t("ui.detect_score"),
                            )
                            detect_score_hint = gr.Markdown(t("ui.detect_score_hint"))
                            with gr.Accordion(t("ui.face_select"), open=False) as face_select:
                                many_faces = gr.Checkbox(
                                    value=initial.swap.selector == "many",
                                    label=t("ui.many_faces"),
                                )
                                reference = gr.File(type="filepath", label=t("ui.reference"), file_count="single")
                                face_map = gr.File(type="filepath", label=t("ui.face_map"), file_count="single")
                                target_index = gr.Number(
                                    value=0,
                                    precision=0,
                                    minimum=0,
                                    label=t("ui.target_index"),
                                )
                            with gr.Accordion(t("ui.enhancer_group"), open=False) as enhancer_group:
                                enhancer_on = gr.Checkbox(value=enhancer_default, label=t("ui.enhancer_enable"))
                                enhancer = gr.Dropdown(
                                    choices=list(ENHANCER_MODELS),
                                    value=initial.enhancer.model,
                                    label=t("ui.enhancer"),
                                )
                                blend = gr.Slider(
                                    0,
                                    1,
                                    value=initial.enhancer.blend,
                                    step=0.05,
                                    label=t("ui.enhancer_blend"),
                                )
                                weight = gr.Slider(
                                    0,
                                    1,
                                    value=initial.enhancer.weight,
                                    step=0.05,
                                    label=t("ui.enhancer_weight"),
                                )
                            with gr.Accordion(t("ui.mask_group"), open=False) as mask_group:
                                occluder_model = gr.Dropdown(
                                    choices=list(OCCLUDER_MODELS),
                                    value=initial.masks.occluder_model,
                                    label=t("ui.occluder_model"),
                                )
                                pixel_boost = gr.Dropdown(
                                    choices=[0, 128, 256, 512, 768, 1024],
                                    value=int(initial.swap.pixel_boost),
                                    label=t("ui.pixel_boost"),
                                )
                                blend_mode = gr.Dropdown(
                                    choices=list(BLEND_MODES),
                                    value=initial.blend.mode,
                                    label=t("ui.blend_mode"),
                                )
                                color_transfer = gr.Dropdown(
                                    choices=list(COLOR_TRANSFERS),
                                    value=initial.blend.color,
                                    label=t("ui.color_transfer"),
                                )
                                eye_mode = gr.Dropdown(
                                    choices=list(EYE_MODES),
                                    value=initial.blend.eye_mode,
                                    label=t("ui.eye_mode"),
                                )
                            with gr.Accordion(t("ui.video_group"), open=False) as video_group:
                                video_mode = gr.Dropdown(
                                    choices=list(VIDEO_MODES),
                                    value=initial.video.mode,
                                    label=t("ui.video_mode"),
                                )
                                encoder = gr.Dropdown(
                                    choices=list(VIDEO_ENCODERS),
                                    value=initial.video.encoder,
                                    label=t("ui.video_encoder"),
                                )
                                trim_start = gr.Textbox(value="", label=t("ui.trim_start"))
                                trim_end = gr.Textbox(value="", label=t("ui.trim_end"))
                                keep_audio = gr.Checkbox(value=initial.video.keep_audio, label=t("ui.keep_audio"))
                            with gr.Accordion(t("ui.execution_group"), open=False) as execution_group:
                                execution = gr.Dropdown(
                                    choices=list(EXECUTION_PROVIDERS),
                                    value=list(initial.execution.providers),
                                    multiselect=True,
                                    label=t("ui.execution"),
                                )
                                dry_run = gr.Checkbox(value=False, label=t("ui.dry_run"))
                            with gr.Accordion(t("ui.compliance_group"), open=False) as compliance_group:
                                compliance = gr.Checkbox(
                                    value=bool(initial.compliance.enabled),
                                    label=t("ui.compliance"),
                                )
                                commercial = gr.Checkbox(value=False, label=t("ui.commercial"))
                                consent = gr.File(type="filepath", label=t("ui.consent"), file_count="single")
                                subject = gr.Textbox(value="", label=t("ui.consent_subject"))
                            with gr.Accordion(t("ui.output_group"), open=False) as output_group:
                                output = gr.Textbox(value="outputs/ui-output.png", label=t("ui.output"))
                        with gr.Row(elem_classes=["bf-actions"]):
                            preview_button = gr.Button(t("ui.preview"), variant="secondary", interactive=False)
                            run_button = gr.Button(
                                t("ui.run"),
                                variant="primary",
                                elem_classes=["bf-cta"],
                                interactive=False,
                            )
                    with gr.Column(scale=62, elem_classes=["bf-stage"]):
                        with gr.Row(elem_classes=["bf-stage-split"]):
                            with gr.Column(scale=3):
                                preview = gr.Image(
                                    type="numpy",
                                    label=t("ui.preview_image"),
                                    height=520,
                                    elem_classes=["bf-preview"],
                                )
                            with gr.Column(scale=2, elem_classes=["bf-voice"]):
                                voice_title = gr.Markdown(t("ui.voice_title"))
                                voice_enable = gr.Checkbox(value=False, label=t("ui.voice_enable"))
                                voice_reference = gr.Audio(
                                    type="filepath",
                                    sources=["upload"],
                                    label=t("ui.voice_reference"),
                                )
                                voice_hint = gr.Markdown(t("ui.voice_hint"))
                                with gr.Accordion(t("ui.voice_advanced"), open=False) as voice_advanced:
                                    voice_strength = gr.Slider(
                                        0,
                                        1,
                                        value=float(initial.voice.strength),
                                        step=0.05,
                                        label=t("ui.voice_strength"),
                                    )
                                    voice_tau = gr.Slider(
                                        0,
                                        1,
                                        value=float(initial.voice.tau),
                                        step=0.05,
                                        label=t("ui.voice_tau"),
                                    )
                                voice_status = gr.Textbox(
                                    value=t("ui.voice_idle"),
                                    label=t("ui.voice_status"),
                                    interactive=False,
                                )
                        progress_bar = gr.Slider(
                            0,
                            1,
                            value=0,
                            step=0.01,
                            interactive=False,
                            label=t("ui.progress_label"),
                        )
                        status = gr.Textbox(
                            value=t("ui.empty_preview"),
                            label=t("ui.status"),
                            interactive=False,
                        )
            (
                live_tab,
                live_intro,
                live_source_mode,
                live_source_video,
                live_source,
                live_source_audio,
                live_voice_override,
                live_start,
                live_stop,
                live_status,
                live_voice,
                live_advanced,
                live_dry,
                live_compliance,
                live_camera,
                live_mic,
                live_speaker,
                live_preview,
            ) = _live_tab(
                gr,
                initial,
                config_file,
                swapper_model,
                pixel_boost,
                occluder_model,
                enhancer_on,
                enhancer,
                blend,
                weight,
                execution,
                detect_score,
            )
        labeled = [
            brand,
            language,
            media_tab,
            steps,
            source_mode,
            source_video,
            source_photo,
            source_audio,
            target,
            swapper_model,
            swapper_hint,
            quality,
            advanced,
            detect_score,
            detect_score_hint,
            face_select,
            many_faces,
            reference,
            face_map,
            target_index,
            enhancer_group,
            enhancer_on,
            enhancer,
            blend,
            weight,
            mask_group,
            occluder_model,
            pixel_boost,
            blend_mode,
            color_transfer,
            eye_mode,
            video_group,
            video_mode,
            encoder,
            trim_start,
            trim_end,
            keep_audio,
            execution_group,
            execution,
            dry_run,
            compliance_group,
            compliance,
            commercial,
            consent,
            subject,
            output_group,
            output,
            preview_button,
            run_button,
            preview,
            voice_title,
            voice_enable,
            voice_reference,
            voice_hint,
            voice_advanced,
            voice_strength,
            voice_tau,
            voice_status,
            progress_bar,
            status,
            live_tab,
            live_intro,
            live_source_mode,
            live_source_video,
            live_source,
            live_source_audio,
            live_voice_override,
            live_start,
            live_stop,
            live_status,
            live_voice,
            live_advanced,
            live_dry,
            live_compliance,
            live_camera,
            live_mic,
            live_speaker,
            live_preview,
        ]
        form = [
            language,
            source_mode,
            source_video,
            source_photo,
            source_audio,
            target,
            output,
            swapper_model,
            pixel_boost,
            enhancer_on,
            enhancer,
            blend,
            weight,
            occluder_model,
            many_faces,
            reference,
            face_map,
            target_index,
            detect_score,
            blend_mode,
            color_transfer,
            eye_mode,
            video_mode,
            encoder,
            trim_start,
            trim_end,
            keep_audio,
            execution,
            dry_run,
            compliance,
            commercial,
            consent,
            subject,
            voice_enable,
            voice_reference,
            voice_strength,
            voice_tau,
        ]
        if len(labeled) != len(CONTROLS):
            raise BarrettFaceswapError(t("ui.control_mismatch", found=len(labeled), expected=len(CONTROLS)))
        tab_js = _tab_button_script()
        language.change(on_language, inputs=[language, source_mode], outputs=labeled, js=tab_js)
        preset_inputs = [quality, swapper_model, pixel_boost, enhancer_on, enhancer, blend, weight, occluder_model]
        preset_outputs = [swapper_model, pixel_boost, enhancer_on, enhancer, blend, weight, occluder_model]
        quality.change(apply_preset, inputs=preset_inputs, outputs=preset_outputs)
        for widget in (swapper_model, pixel_boost, enhancer_on, occluder_model):
            widget.change(
                sync_preset,
                inputs=[quality, swapper_model, pixel_boost, enhancer_on, occluder_model],
                outputs=quality,
            )
        mode_inputs = [source_mode, source_video, source_photo, source_audio, target]
        source_mode.change(
            _mode_view,
            inputs=mode_inputs,
            outputs=[
                source_video,
                source_photo,
                source_audio,
                voice_reference,
                voice_hint,
                voice_enable,
                voice_status,
                preview_button,
                run_button,
            ],
        )
        for widget in (source_video, source_photo, source_audio):
            widget.change(_files_ready, inputs=mode_inputs, outputs=[preview_button, run_button])
        target.change(
            _target_changed,
            inputs=mode_inputs,
            outputs=[preview_button, run_button, voice_enable, voice_status],
        )
        preview_button.click(on_preview, inputs=form, outputs=[preview, status, progress_bar, voice_status])
        run_button.click(on_run, inputs=form, outputs=[preview, status, progress_bar, voice_status])
    return demo


def _tab_button_script() -> str:
    """Gradio does not repaint tab buttons from a label update. The browser does."""

    current = get_language()
    captured: dict[str, str] = {}
    for code in ("en", "zh-CN"):
        set_language(code)
        captured[f"media-{code}"] = t("ui.media_tab")
        captured[f"live-{code}"] = t("ui.live_tab")
    set_language(current)
    media_en = captured["media-en"]
    media_zh = captured["media-zh-CN"]
    live_en = captured["live-en"]
    live_zh = captured["live-zh-CN"]
    return (
        "(lang) => {"
        f" const media = lang === 'zh-CN' ? {media_zh!r} : {media_en!r};"
        f" const live = lang === 'zh-CN' ? {live_zh!r} : {live_en!r};"
        " const known = new Map(["
        f"[{media_en!r}, media], [{media_zh!r}, media],"
        f"[{live_en!r}, live], [{live_zh!r}, live]"
        " ]);"
        " const paint = () => {"
        "   document.querySelectorAll('button').forEach((button) => {"
        "     const text = (button.innerText || '').trim();"
        "     const next = known.get(text);"
        "     if (next) button.innerText = next;"
        "   });"
        " };"
        " paint();"
        " setTimeout(paint, 0);"
        " setTimeout(paint, 250);"
        " return lang;"
        "}"
    )


def _live_tab(
    gr,
    initial,
    config_file: Path,
    swapper_model,
    pixel_boost,
    occluder_model,
    enhancer_on,
    enhancer,
    blend,
    weight,
    execution,
    detect_score,
):
    """Webcam tab. The source choices match the media tab. Voice needs the mic."""

    import shutil
    import tempfile

    from barrett_faceswap.live import LiveSession
    from barrett_faceswap.live_voice import apply_live_chunk
    from barrett_faceswap.sourceprep import prepare_source

    holder: dict[str, object] = {
        "session": None,
        "armed": False,
        "sig": None,
        "camera_on": False,
        "work": None,
        "request": None,
        "config": None,
        "reference": None,
        "converter": None,
    }
    with gr.Tab(t("ui.live_tab")) as live_tab:
        live_intro = gr.Markdown(t("ui.live_intro"))
        with gr.Row(elem_classes=["bf-layout"]):
            with gr.Column(scale=38, elem_classes=["bf-controls"]):
                live_source_mode = gr.Radio(
                    choices=_source_mode_choices(),
                    value="video",
                    label=t("ui.source_mode"),
                    elem_classes=["bf-source-mode"],
                )
                live_source_video = gr.File(
                    type="filepath",
                    file_count="single",
                    file_types=[".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"],
                    label=t("ui.source_video"),
                )
                live_source = gr.Image(
                    type="filepath",
                    sources=["upload", "clipboard"],
                    label=t("ui.source"),
                    visible=False,
                )
                live_source_audio = gr.Audio(
                    type="filepath",
                    sources=["upload"],
                    label=t("ui.source_audio"),
                    visible=False,
                )
                live_voice_override = gr.Audio(
                    type="filepath",
                    sources=["upload"],
                    label=t("ui.live_voice_override"),
                )
                with gr.Row(elem_classes=["bf-actions"]):
                    live_start = gr.Button(
                        t("ui.live_start"),
                        variant="primary",
                        elem_classes=["bf-cta"],
                        interactive=False,
                    )
                    live_stop = gr.Button(t("ui.live_stop"), variant="secondary")
                live_status = gr.Textbox(label=t("ui.live_status"), interactive=False)
                live_voice = gr.Textbox(
                    value=t("ui.live_face_only"),
                    label=t("ui.live_voice"),
                    interactive=False,
                )
                with gr.Accordion(t("ui.advanced"), open=False) as live_advanced:
                    live_dry = gr.Checkbox(value=True, label=t("ui.dry_run"))
                    live_compliance = gr.Checkbox(
                        value=bool(initial.compliance.enabled),
                        label=t("ui.compliance"),
                    )
            with gr.Column(scale=62, elem_classes=["bf-stage"]):
                live_camera = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label=t("ui.live_camera"),
                    height=220,
                )
                live_mic = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    type="numpy",
                    label=t("ui.live_mic"),
                )
                live_speaker = gr.Audio(
                    label=t("ui.live_speaker"),
                    interactive=False,
                    autoplay=True,
                )
                live_preview = gr.Image(
                    type="numpy",
                    label=t("ui.live_preview"),
                    height=480,
                    elem_classes=["bf-preview"],
                )

    source_inputs = [
        live_source_mode,
        live_source_video,
        live_source,
        live_source_audio,
        live_voice_override,
    ]

    def _source_choice(mode, video, photo, audio_file):
        selected = mode if mode in {"video", "photo", "photo_audio"} else "video"
        flags = source_mode_visibility(selected)
        source_path = _as_path(video) if flags["video"] else _as_path(photo)
        if source_path is None:
            raise BarrettFaceswapError(t("ui.need_files"))
        if flags["video"] and not is_video(source_path):
            raise BarrettFaceswapError(t("ui.need_source_video"))
        if not flags["video"] and is_video(source_path):
            raise BarrettFaceswapError(t("ui.need_source_photo"))
        mode_audio = _as_path(audio_file) if flags["audio"] else None
        if flags["audio"] and mode_audio is None:
            raise BarrettFaceswapError(t("ui.need_source_audio"))
        return selected, source_path, mode_audio

    def _request(
        mode,
        video,
        photo,
        audio_file,
        dry_run: bool,
        compliance: bool,
        swapper: str,
        boost: int,
        occluder: str,
        enhancer_enabled: bool,
        enhancer_model: str,
        enhancer_blend: float,
        enhancer_weight: float,
        providers: list[str] | None,
        detect_score: float,
    ) -> tuple[str, Path, Path | None, SwapRequest]:
        selected, source_path, mode_audio = _source_choice(mode, video, photo, audio_file)
        processors = ["face_swapper"]
        if enhancer_enabled:
            processors.append("face_enhancer")
        request = SwapRequest(
            source_path=source_path,
            target_path=Path("camera-ui"),
            output_path=Path("outputs/live-ui-session.txt"),
            dry_run=bool(dry_run),
            enable_compliance=bool(compliance),
            live_session=True,
            acknowledge_authorized_use=False,
            processors=processors,
            swapper_model=swapper,
            pixel_boost=int(boost),
            occluder_model=occluder,
            enhancer_model=enhancer_model,
            enhancer_blend=float(enhancer_blend),
            enhancer_weight=float(enhancer_weight),
            execution_providers=list(providers) if providers else None,
            with_enhancer=bool(enhancer_enabled),
            detect_score=float(detect_score),
            voice_enabled=False,
        )
        return selected, source_path, mode_audio, request

    def _signature(mode, video, photo, audio_file, override, dry_run, compliance, *model_args):
        swapper, boost, occluder, enhancer_enabled, model, blend_value, weight_value, providers, detect_score = model_args
        return (
            str(mode),
            str(_as_path(video)),
            str(_as_path(photo)),
            str(_as_path(audio_file)),
            str(_as_path(override)),
            bool(dry_run),
            bool(compliance),
            str(swapper),
            int(boost),
            str(occluder),
            bool(enhancer_enabled),
            str(model),
            round(float(blend_value), 4),
            round(float(weight_value), 4),
            tuple(providers or ()),
            round(float(detect_score), 4),
        )

    shared = [
        swapper_model,
        pixel_boost,
        occluder_model,
        enhancer_on,
        enhancer,
        blend,
        weight,
        execution,
        detect_score,
    ]

    def _arm(mode, video, photo, audio_file, override, dry_run, compliance, *model_args):
        _close()
        selected, source_path, mode_audio, request = _request(
            mode, video, photo, audio_file, dry_run, compliance, *model_args
        )
        work = Path(tempfile.mkdtemp(prefix="barrett-live-"))
        config = apply_request_overrides(load_config(config_file), request)
        prepared, _notes = prepare_source(
            config,
            request.model_copy(update={"voice_enabled": False}),
            work,
        )
        reference, _kind = resolve_live_reference(
            selected,
            source_path,
            _as_path(override),
            mode_audio,
            work,
        )
        holder["work"] = work
        holder["config"] = config
        holder["request"] = prepared
        holder["reference"] = reference
        holder["converter"] = None
        holder["armed"] = True
        holder["camera_on"] = False
        holder["sig"] = _signature(mode, video, photo, audio_file, override, dry_run, compliance, *model_args)
        if reference is None:
            return t("ui.live_ready"), t("ui.live_face_only")
        if dry_run:
            return t("ui.live_ready"), t("ui.live_voice_dry")
        return t("ui.live_ready"), t("ui.live_voice_wait")

    def on_start(mode, video, photo, audio_file, override, dry_run: bool, compliance: bool, *model_args):
        try:
            status, voice = _arm(mode, video, photo, audio_file, override, dry_run, compliance, *model_args)
        except (BarrettFaceswapError, ValidationError, LanguageError) as exc:
            _close()
            return str(exc), t("ui.live_face_only")
        return status, voice

    def on_frame(frame, mode, video, photo, audio_file, override, dry_run: bool, compliance: bool, *model_args):
        if frame is None:
            return None
        try:
            if not holder.get("armed"):
                return frame
            signature = _signature(mode, video, photo, audio_file, override, dry_run, compliance, *model_args)
            session = holder.get("session")
            if session is None or holder.get("sig") != signature:
                _arm(mode, video, photo, audio_file, override, dry_run, compliance, *model_args)
                request = holder.get("request")
                config = holder.get("config")
                if not isinstance(request, SwapRequest) or config is None:
                    return frame
                session = LiveSession(config, request)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                session.start(bgr)
                holder["session"] = session
            else:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            holder["camera_on"] = True
            output = session.step(bgr)
            return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        except (BarrettFaceswapError, ValidationError, LanguageError):
            _close()
            return None

    def on_mic(audio, mode, video, photo, audio_file, override, dry_run: bool, compliance: bool, *model_args):
        del mode, video, photo, audio_file, override, compliance, model_args
        samples, sample_rate = _mic_chunk(audio)
        reference = holder.get("reference")
        reference_path = reference if isinstance(reference, Path) else None
        work = holder.get("work")
        if not isinstance(work, Path):
            work = Path(tempfile.gettempdir()) / "barrett-live-idle"
        request = holder.get("request")
        request_dry = bool(getattr(request, "dry_run", False)) if request is not None else False
        config = holder.get("config")
        converted, plan = apply_live_chunk(
            samples,
            sample_rate,
            camera_on=bool(holder.get("camera_on")),
            reference=reference_path,
            dry_run=bool(dry_run) or request_dry,
            tau=float(initial.voice.tau),
            work=work,
            converter=holder.get("converter") if callable(holder.get("converter")) else None,
            config=config if isinstance(config, AppConfig) else None,
            cache=holder,
        )
        played = None if converted is None else (sample_rate, converted)
        return played, t(_live_voice_key(plan))

    def _close() -> None:
        session = holder.get("session")
        work = holder.get("work")
        holder["session"] = None
        holder["sig"] = None
        holder["camera_on"] = False
        holder["reference"] = None
        holder["request"] = None
        holder["config"] = None
        holder["converter"] = None
        holder["work"] = None
        if session is not None:
            try:
                session.finish()
            except Exception:
                holder["session"] = None
        if isinstance(work, Path):
            shutil.rmtree(work, ignore_errors=True)

    def on_stop():
        session = holder.get("session")
        holder["armed"] = False
        holder["camera_on"] = False
        if session is None:
            _close()
            return t("ui.live_idle"), t("ui.live_face_only")
        result = session.finish()
        holder["session"] = None
        _close()
        return t("ui.live_stopped", path=result.output_path), t("ui.live_face_only")

    def _mode_view(mode, video, photo, audio_file, override):
        del override
        flags = source_mode_visibility(mode)
        ready = source_files_ready(mode, video, photo, audio_file, "camera")
        return (
            gr.update(visible=flags["video"]),
            gr.update(visible=flags["photo"]),
            gr.update(visible=flags["audio"]),
            gr.update(visible=flags["voice_override"]),
            gr.update(interactive=ready),
        )

    def _files_ready(mode, video, photo, audio_file):
        ready = source_files_ready(mode, video, photo, audio_file, "camera")
        return gr.update(interactive=ready)

    live_source_mode.change(
        _mode_view,
        inputs=source_inputs,
        outputs=[live_source_video, live_source, live_source_audio, live_voice_override, live_start],
    )
    for widget in (live_source_video, live_source, live_source_audio):
        widget.change(_files_ready, inputs=source_inputs[:4], outputs=live_start)
    live_start.click(
        on_start,
        inputs=[*source_inputs, live_dry, live_compliance, *shared],
        outputs=[live_status, live_voice],
    )
    live_camera.stream(
        on_frame,
        inputs=[live_camera, *source_inputs, live_dry, live_compliance, *shared],
        outputs=live_preview,
    )
    live_mic.stream(
        on_mic,
        inputs=[live_mic, *source_inputs, live_dry, live_compliance, *shared],
        outputs=[live_speaker, live_voice],
    )
    live_stop.click(on_stop, outputs=[live_status, live_voice])
    return (
        live_tab,
        live_intro,
        live_source_mode,
        live_source_video,
        live_source,
        live_source_audio,
        live_voice_override,
        live_start,
        live_stop,
        live_status,
        live_voice,
        live_advanced,
        live_dry,
        live_compliance,
        live_camera,
        live_mic,
        live_speaker,
        live_preview,
    )


def _live_voice_key(plan: str) -> str:
    """Status catalog key for one live microphone chunk."""

    return {
        "run": "ui.live_voice_ready",
        "dry_run": "ui.live_voice_dry",
        "wait": "ui.live_voice_wait",
        "unavailable": "ui.live_voice_unavailable",
    }.get(plan, "ui.live_face_only")


def _mic_chunk(audio: object) -> tuple[np.ndarray | None, int]:
    """Gradio's streaming microphone value is ``(sample_rate, samples)``."""

    if audio is None:
        return None, 16000
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], (int, float)):
        rate = int(audio[0]) or 16000
        data = audio[1]
        if data is None:
            return None, rate
        return np.asarray(data), rate
    return np.asarray(audio), 16000


def _voice_reference_for_mode(mode: str, mode_audio: Path | None, override: Path | None) -> Path | None:
    """Uploaded clip for this source mode. See :func:`reference_for_mode`."""

    return reference_for_mode(mode, mode_audio, override)


def _as_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if isinstance(value, dict):
        raw = value.get("path") or value.get("name")
        return Path(str(raw)) if raw else None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _seconds(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise BarrettFaceswapError(t("video.trim_number", value=text)) from exc
