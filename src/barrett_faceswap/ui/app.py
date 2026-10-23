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
from pydantic import ValidationError

from barrett_faceswap.config import (
    ENHANCER_MODELS,
    EXECUTION_PROVIDERS,
    VIDEO_ENCODERS,
    apply_request_overrides,
    load_config,
)
from barrett_faceswap.errors import BarrettFaceswapError
from barrett_faceswap.i18n import LanguageError, get_language, normalize_language, set_language, t
from barrett_faceswap.media.ffmpeg import VIDEO_OUTPUT_SUFFIXES, read_first_frame
from barrett_faceswap.pipeline import FaceSwapPipeline
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
_TEXT = "#E8ECF1"
_MUTED = "#9AA3B2"
_CTA = "#2563EB"
_CTA_HOVER = "#3B82F6"

APP_CSS = f"""
.gradio-container {{
  background: {_BG} !important;
  color: {_TEXT};
  font-family: "PingFang SC", "Noto Sans SC", system-ui, sans-serif !important;
}}
.gradio-container .block {{
  background: {_CARD};
  border-color: {_LINE};
}}
footer {{ display: none !important; }}
.bf-layout {{ align-items: stretch; gap: 16px; }}
.bf-preview {{ min-height: 420px; }}
.bf-cta {{ min-height: 44px; background: {_CTA} !important; color: #ffffff !important; }}
.bf-cta:hover {{ background: {_CTA_HOVER} !important; }}
.bf-actions button:disabled {{ opacity: 0.4; }}
@media (max-width: 959px) {{
  .bf-layout {{ flex-direction: column !important; }}
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

# Fast and Balanced share one field tuple. resolve_preset keeps the current
# name when it is one of the matches, so the two radios do not collapse.
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
        "pixel_boost": 512,
        "enhancer": True,
        "occluder": "xseg_1",
        "enhancer_model": "gfpgan_1.4",
        "blend": 0.35,
        "weight": 0.7,
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

    The widgets stay the source of truth. Fast and Balanced are the same
    numbers, so a current match is kept. Two matches that do not include the
    current name become Custom instead of snapping to one of them. One match
    selects that preset.
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
    ControlSpec("ui.source", "label"),
    ControlSpec("ui.target", "label"),
    ControlSpec("ui.swapper_model", "label"),
    ControlSpec("ui.swapper_hint", "value"),
    ControlSpec("ui.quality", "label"),
    ControlSpec("ui.advanced", "label"),
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
    ControlSpec("ui.progress_label", "label"),
    ControlSpec("ui.status", "label"),
    ControlSpec("ui.live_tab", "label"),
    ControlSpec("ui.live_intro", "value"),
    ControlSpec("ui.live_source", "label"),
    ControlSpec("ui.live_start", "value"),
    ControlSpec("ui.live_stop", "value"),
    ControlSpec("ui.live_status", "label"),
    ControlSpec("ui.advanced", "label"),
    ControlSpec("ui.dry_run", "label"),
    ControlSpec("ui.compliance", "label"),
    ControlSpec("ui.live_camera", "label"),
    ControlSpec("ui.live_preview", "label"),
)


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
            body_text_color_subdued=_MUTED,
            body_text_color_subdued_dark=_MUTED,
            button_primary_background_fill=_CTA,
            button_primary_background_fill_dark=_CTA,
            button_primary_background_fill_hover=_CTA_HOVER,
            button_primary_background_fill_hover_dark=_CTA_HOVER,
            button_primary_text_color="#FFFFFF",
            button_primary_text_color_dark="#FFFFFF",
            button_secondary_background_fill="transparent",
            button_secondary_background_fill_dark="transparent",
            button_secondary_border_color=_CTA_HOVER,
            button_secondary_border_color_dark=_CTA_HOVER,
            button_secondary_text_color=_TEXT,
            button_secondary_text_color_dark=_TEXT,
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

    def on_language(selected: str):
        updates = language_updates(selected)
        wrapped = []
        for spec, payload in zip(CONTROLS, updates, strict=True):
            data = dict(payload)
            if spec.key == "ui.language":
                data["choices"] = _language_choices()
            elif spec.key == "ui.quality":
                data["choices"] = _quality_choices()
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

    def _files_ready(source, target):
        ready = _as_path(source) is not None and _as_path(target) is not None
        return gr.update(interactive=ready), gr.update(interactive=ready)

    def _preview_array(path: Path):
        image = read_first_frame(path) if is_video(path) else read_image(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()

    def _stream_video(request: SwapRequest, status_key: str, progress):
        box: dict[str, object] = {"done": 0, "total": 0, "result": None, "error": None}

        def hook(done: int, total: int) -> None:
            box["done"] = int(done)
            box["total"] = int(total)

        def worker() -> None:
            try:
                config = apply_request_overrides(load_config(config_file), request)
                box["result"] = FaceSwapPipeline(config).run(request, on_progress=hook)
            except Exception as exc:  # noqa: BLE001 - re-raised on the handler thread
                box["error"] = exc

        thread = threading.Thread(target=worker, name="barrett-ui-video", daemon=True)
        thread.start()
        while thread.is_alive():
            done = int(box["done"])
            total = int(box["total"])
            if total:
                progress((done, total), desc=t("ui.progress", done=done, total=total))
                message = t("ui.progress", done=done, total=total)
                fraction = done / total
            else:
                progress(0, desc=t("ui.waiting"))
                message = t("ui.waiting")
                fraction = 0
            yield gr.skip(), message, fraction
            thread.join(0.4)
        error = box["error"]
        if error is not None:
            raise error
        result = box["result"]
        done = int(box["done"]) or 1
        total = int(box["total"]) or 1
        progress((total, total), desc=t("ui.progress", done=total, total=total))
        yield _preview_array(result.output_path), t(status_key, path=result.output_path), 1

    def on_execute(
        selected_language: str,
        source: object,
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
        preview_only: bool,
        progress=gr.Progress(track_tqdm=False),
    ):
        try:
            language_updates(selected_language or get_language())
            source_path = _as_path(source)
            target_path = _as_path(target)
            if source_path is None or target_path is None:
                yield None, t("ui.need_files"), 0
                return
            processors = ["face_swapper"]
            if enhancer_on:
                processors.append("face_enhancer")
            if not processors:
                yield None, t("ui.need_processor"), 0
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
                )
                full_video = is_video(target_path) and not preview_only
                if full_video:
                    yield from _stream_video(request, status_key, progress)
                    return
                progress(0, desc=t("ui.progress_running"))
                config = apply_request_overrides(load_config(config_file), request)
                result = FaceSwapPipeline(config).run(request)
                progress(1, desc=t("ui.progress_running"))
                yield _preview_array(result.output_path), t(status_key, path=result.output_path), 1
        except (BarrettFaceswapError, ValidationError, LanguageError) as exc:
            yield None, str(exc), 0

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
                        source = gr.Image(
                            type="filepath",
                            sources=["upload", "clipboard"],
                            label=t("ui.source"),
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
                        preview = gr.Image(
                            type="numpy",
                            label=t("ui.preview_image"),
                            height=520,
                            elem_classes=["bf-preview"],
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
                live_source,
                live_start,
                live_stop,
                live_status,
                live_advanced,
                live_dry,
                live_compliance,
                live_camera,
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
            )
        labeled = [
            brand,
            language,
            media_tab,
            steps,
            source,
            target,
            swapper_model,
            swapper_hint,
            quality,
            advanced,
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
            progress_bar,
            status,
            live_tab,
            live_intro,
            live_source,
            live_start,
            live_stop,
            live_status,
            live_advanced,
            live_dry,
            live_compliance,
            live_camera,
            live_preview,
        ]
        form = [
            language,
            source,
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
        ]
        if len(labeled) != len(CONTROLS):
            raise BarrettFaceswapError(t("ui.control_mismatch", found=len(labeled), expected=len(CONTROLS)))
        tab_js = _tab_button_script()
        language.change(on_language, inputs=[language], outputs=labeled, js=tab_js)
        preset_inputs = [quality, swapper_model, pixel_boost, enhancer_on, enhancer, blend, weight, occluder_model]
        preset_outputs = [swapper_model, pixel_boost, enhancer_on, enhancer, blend, weight, occluder_model]
        quality.change(apply_preset, inputs=preset_inputs, outputs=preset_outputs)
        for widget in (swapper_model, pixel_boost, enhancer_on, occluder_model):
            widget.change(
                sync_preset,
                inputs=[quality, swapper_model, pixel_boost, enhancer_on, occluder_model],
                outputs=quality,
            )
        source.change(_files_ready, inputs=[source, target], outputs=[preview_button, run_button])
        target.change(_files_ready, inputs=[source, target], outputs=[preview_button, run_button])
        preview_button.click(on_preview, inputs=form, outputs=[preview, status, progress_bar])
        run_button.click(on_run, inputs=form, outputs=[preview, status, progress_bar])
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
):
    """Webcam tab. Model controls are the ones on the image/video tab."""

    from barrett_faceswap.live import LiveSession

    holder: dict[str, object] = {"session": None, "armed": False, "sig": None}
    with gr.Tab(t("ui.live_tab")) as live_tab:
        live_intro = gr.Markdown(t("ui.live_intro"))
        with gr.Row(elem_classes=["bf-layout"]):
            with gr.Column(scale=38, elem_classes=["bf-controls"]):
                live_source = gr.Image(
                    type="filepath",
                    sources=["upload", "clipboard"],
                    label=t("ui.live_source"),
                )
                with gr.Row(elem_classes=["bf-actions"]):
                    live_start = gr.Button(t("ui.live_start"), variant="primary", elem_classes=["bf-cta"])
                    live_stop = gr.Button(t("ui.live_stop"), variant="secondary")
                live_status = gr.Textbox(label=t("ui.live_status"), interactive=False)
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
                live_preview = gr.Image(
                    type="numpy",
                    label=t("ui.live_preview"),
                    height=480,
                    elem_classes=["bf-preview"],
                )

    def _request(
        source,
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
    ) -> SwapRequest:
        source_path = _as_path(source)
        if source_path is None:
            raise BarrettFaceswapError(t("ui.need_files"))
        processors = ["face_swapper"]
        if enhancer_enabled:
            processors.append("face_enhancer")
        return SwapRequest(
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
        )

    def _signature(source, dry_run, compliance, swapper, boost, occluder, enhancer_enabled, model, blend_value, weight_value, providers):
        return (
            str(_as_path(source)),
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
    ]

    def on_start(source, dry_run: bool, compliance: bool, *model_args):
        _close()
        try:
            _request(source, dry_run, compliance, *model_args)
        except BarrettFaceswapError as exc:
            return str(exc)
        holder["armed"] = True
        return t("ui.live_ready")

    def on_frame(frame, source, dry_run: bool, compliance: bool, *model_args):
        if frame is None:
            return None
        try:
            if not holder.get("armed"):
                return frame
            signature = _signature(source, dry_run, compliance, *model_args)
            session = holder.get("session")
            if session is None or holder.get("sig") != signature:
                _close()
                config = load_config(config_file)
                request = _request(source, dry_run, compliance, *model_args)
                config = apply_request_overrides(config, request)
                session = LiveSession(config, request)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                session.start(bgr)
                holder["session"] = session
                holder["sig"] = signature
                holder["armed"] = True
            else:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            output = session.step(bgr)
            return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        except (BarrettFaceswapError, ValidationError, LanguageError):
            _close()
            return None

    def _close() -> None:
        session = holder.get("session")
        holder["session"] = None
        holder["sig"] = None
        if session is not None:
            try:
                session.finish()
            except Exception:
                holder["session"] = None

    def on_stop():
        session = holder.get("session")
        holder["session"] = None
        holder["sig"] = None
        holder["armed"] = False
        if session is None:
            return t("ui.live_idle")
        result = session.finish()
        return t("ui.live_stopped", path=result.output_path)

    live_start.click(on_start, inputs=[live_source, live_dry, live_compliance, *shared], outputs=live_status)
    live_camera.stream(
        on_frame,
        inputs=[live_camera, live_source, live_dry, live_compliance, *shared],
        outputs=live_preview,
    )
    live_stop.click(on_stop, outputs=live_status)
    return (
        live_tab,
        live_intro,
        live_source,
        live_start,
        live_stop,
        live_status,
        live_advanced,
        live_dry,
        live_compliance,
        live_camera,
        live_preview,
    )


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
