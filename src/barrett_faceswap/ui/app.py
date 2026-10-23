"""Gradio shell. Labels come from the catalogs and switch with the dropdown.

Importing this module does not import Gradio. ``build_demo`` and ``launch_ui``
import it, and they fail with an install hint when the extra is missing.
"""

from __future__ import annotations

import tempfile
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
from barrett_faceswap.processors import available_processors
from barrett_faceswap.types import SwapRequest
from barrett_faceswap.vision import is_video, read_image, write_image


@dataclass(frozen=True)
class ControlSpec:
    """One Gradio control whose caption is a catalog key."""

    key: str
    prop: str


# Order matches the components created in :func:`build_demo`.
CONTROLS: tuple[ControlSpec, ...] = (
    ControlSpec("ui.intro", "value"),
    ControlSpec("ui.language", "label"),
    ControlSpec("ui.source", "label"),
    ControlSpec("ui.target", "label"),
    ControlSpec("ui.output", "label"),
    ControlSpec("ui.processors", "label"),
    ControlSpec("ui.execution", "label"),
    ControlSpec("ui.enhancer", "label"),
    ControlSpec("ui.enhancer_blend", "label"),
    ControlSpec("ui.enhancer_weight", "label"),
    ControlSpec("ui.video_encoder", "label"),
    ControlSpec("ui.trim_start", "label"),
    ControlSpec("ui.trim_end", "label"),
    ControlSpec("ui.keep_audio", "label"),
    ControlSpec("ui.dry_run", "label"),
    ControlSpec("ui.compliance", "label"),
    ControlSpec("ui.consent", "label"),
    ControlSpec("ui.consent_subject", "label"),
    ControlSpec("ui.preview", "value"),
    ControlSpec("ui.run", "value"),
    ControlSpec("ui.preview_image", "label"),
    ControlSpec("ui.status", "label"),
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


def launch_ui(config_path: str | Path, *, host: str, port: int, share: bool = False) -> None:
    """Build the Blocks app and serve it."""

    demo = build_demo(config_path)
    demo.launch(server_name=host, server_port=int(port), share=share)


def build_demo(config_path: str | Path):
    """Construct the Gradio Blocks app. Gradio is imported here."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise BarrettFaceswapError(t("ui.gradio_missing")) from exc

    config_file = Path(config_path)
    initial = load_config(config_file)
    set_language(initial.ui.language)

    def on_language(language: str):
        updates = language_updates(language)
        wrapped = []
        for spec, payload in zip(CONTROLS, updates, strict=True):
            data = dict(payload)
            if spec.key == "ui.language":
                data["choices"] = [(t("ui.lang_en"), "en"), (t("ui.lang_zh"), "zh-CN")]
            wrapped.append(gr.update(**data))
        return wrapped

    def on_execute(
        language: str,
        source: object,
        target: object,
        output_text: str,
        processors: list[str] | None,
        providers: list[str] | None,
        enhancer_model: str,
        blend: float,
        weight: float,
        encoder: str,
        trim_start_text: str,
        trim_end_text: str,
        keep_audio: bool,
        dry_run: bool,
        compliance: bool,
        consent: object,
        subject: str,
        preview_only: bool,
    ):
        try:
            language_updates(language or get_language())
            source_path = _as_path(source)
            target_path = _as_path(target)
            if source_path is None or target_path is None:
                return None, t("ui.need_files")
            if not processors:
                return None, t("ui.need_processor")
            with tempfile.TemporaryDirectory(prefix="barrett-faceswap-ui-") as directory:
                temp = Path(directory)
                status_key = "ui.done"
                request_target = target_path
                video_encoder = video_encoder_for_run(target_path, encoder, preview_only=preview_only)
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
                )
                config = apply_request_overrides(load_config(config_file), request)
                result = FaceSwapPipeline(config).run(request)
                if is_video(result.output_path):
                    image = read_first_frame(result.output_path)
                else:
                    image = read_image(result.output_path)
                # Copy pixels out of the temp directory before it is deleted.
                preview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
                return preview, t(status_key, path=result.output_path)
        except (BarrettFaceswapError, ValidationError, LanguageError) as exc:
            return None, str(exc)

    with gr.Blocks(title=t("ui.title")) as demo:
        intro = gr.Markdown(t("ui.intro"))
        language = gr.Dropdown(
            choices=[(t("ui.lang_en"), "en"), (t("ui.lang_zh"), "zh-CN")],
            value=initial.ui.language,
            label=t("ui.language"),
        )
        source = gr.Image(type="filepath", label=t("ui.source"))
        target = gr.File(type="filepath", label=t("ui.target"), file_count="single")
        output = gr.Textbox(value="outputs/ui-output.png", label=t("ui.output"))
        processors = gr.CheckboxGroup(
            choices=available_processors(),
            value=list(initial.processors),
            label=t("ui.processors"),
        )
        execution = gr.Dropdown(
            choices=list(EXECUTION_PROVIDERS),
            value=list(initial.execution.providers),
            multiselect=True,
            label=t("ui.execution"),
        )
        enhancer = gr.Dropdown(
            choices=list(ENHANCER_MODELS),
            value=initial.enhancer.model,
            label=t("ui.enhancer"),
        )
        blend = gr.Slider(0, 1, value=initial.enhancer.blend, step=0.05, label=t("ui.enhancer_blend"))
        weight = gr.Slider(0, 1, value=initial.enhancer.weight, step=0.05, label=t("ui.enhancer_weight"))
        encoder = gr.Dropdown(
            choices=list(VIDEO_ENCODERS),
            value=initial.video.encoder,
            label=t("ui.video_encoder"),
        )
        trim_start = gr.Textbox(value="", label=t("ui.trim_start"))
        trim_end = gr.Textbox(value="", label=t("ui.trim_end"))
        keep_audio = gr.Checkbox(value=initial.video.keep_audio, label=t("ui.keep_audio"))
        dry_run = gr.Checkbox(value=False, label=t("ui.dry_run"))
        compliance = gr.Checkbox(value=bool(initial.compliance.enabled), label=t("ui.compliance"))
        consent = gr.File(type="filepath", label=t("ui.consent"), file_count="single")
        subject = gr.Textbox(value="", label=t("ui.consent_subject"))
        preview_button = gr.Button(t("ui.preview"))
        run_button = gr.Button(t("ui.run"))
        preview = gr.Image(type="numpy", label=t("ui.preview_image"))
        status = gr.Textbox(label=t("ui.status"), interactive=False)
        labeled = [
            intro,
            language,
            source,
            target,
            output,
            processors,
            execution,
            enhancer,
            blend,
            weight,
            encoder,
            trim_start,
            trim_end,
            keep_audio,
            dry_run,
            compliance,
            consent,
            subject,
            preview_button,
            run_button,
            preview,
            status,
        ]
        if len(labeled) != len(CONTROLS):
            raise BarrettFaceswapError(t("ui.control_mismatch", found=len(labeled), expected=len(CONTROLS)))
        form = [
            language,
            source,
            target,
            output,
            processors,
            execution,
            enhancer,
            blend,
            weight,
            encoder,
            trim_start,
            trim_end,
            keep_audio,
            dry_run,
            compliance,
            consent,
            subject,
        ]
        language.change(on_language, inputs=[language], outputs=labeled)
        preview_button.click(
            lambda *args: on_execute(*args, True),
            inputs=form,
            outputs=[preview, status],
        )
        run_button.click(
            lambda *args: on_execute(*args, False),
            inputs=form,
            outputs=[preview, status],
        )
    return demo


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
