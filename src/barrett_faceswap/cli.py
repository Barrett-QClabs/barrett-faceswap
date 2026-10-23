"""Command line entry points for ``barrett-faceswap`` and ``python -m``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from barrett_faceswap import __version__
from barrett_faceswap.benchmark import run_benchmark, run_live_benchmark
from barrett_faceswap.config import ENHANCER_MODELS, VIDEO_ENCODERS, apply_request_overrides, load_config
from barrett_faceswap.swap_catalog import (
    BLEND_MODES,
    COLOR_TRANSFERS,
    EYE_MODES,
    SWAPPER_NAMES,
    VIDEO_MODES,
)
from barrett_faceswap.errors import (
    BarrettFaceswapError,
    ComplianceError,
    ConfigError,
    ModelHashError,
    ModelNotFoundError,
)
from barrett_faceswap.i18n import LanguageError, activate_language, t
from barrett_faceswap.model_registry import describe_models
from barrett_faceswap.pipeline import FaceSwapPipeline, list_processor_names
from barrett_faceswap.types import SwapRequest

_DEFAULT_CONFIG = "configs/default.yaml"


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""

    args_list = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        _activate_from_argv(args_list)
        parser = build_parser()
        namespace = parser.parse_args(args_list)
        if getattr(namespace, "lang", None):
            activate_language(cli=namespace.lang, env=None, config=None)
        return _dispatch(namespace)
    except LanguageError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ComplianceError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except (ModelNotFoundError, ModelHashError) as exc:
        print(str(exc), file=sys.stderr)
        return 4
    except BarrettFaceswapError as exc:
        print(str(exc), file=sys.stderr)
        return 5
    except ValidationError as exc:
        print(t("config.validation", detail=exc), file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    """Build the parser after the language catalog is active."""

    parser = argparse.ArgumentParser(prog="barrett-faceswap", description=t("cli.prog_description"))
    parser.add_argument("--version", action="version", version=f"barrett-faceswap {__version__}")
    parser.add_argument("--lang", default=None, help=t("cli.lang"))
    sub = parser.add_subparsers(dest="command", required=True)

    swap = sub.add_parser("swap", help=t("cli.swap_help"))
    swap.add_argument("--source", required=True, type=Path, help=t("cli.source"))
    swap.add_argument("--target", required=True, type=Path, help=t("cli.target"))
    swap.add_argument("--output", required=True, type=Path, help=t("cli.output"))
    swap.add_argument("--config", default=_DEFAULT_CONFIG, help=t("cli.config"))
    swap.add_argument("--dry-run", action="store_true", help=t("cli.dry_run"))
    swap.add_argument("--enable-compliance", action="store_true", help=t("cli.enable_compliance"))
    swap.add_argument("--lang", default=None, help=t("cli.lang"))
    swap.add_argument("--consent-file", type=Path, default=None, help=t("cli.consent_file"))
    swap.add_argument("--acknowledge-authorized-use", action="store_true", help=t("cli.acknowledge"))
    swap.add_argument("--consent-subject", default=None, help=t("cli.consent_subject"))
    swap.add_argument("--consent-statement", default=None, help=t("cli.consent_statement"))
    swap.add_argument("--no-watermark", action="store_true", help=t("cli.no_watermark"))
    swap.add_argument("--disable-content-filter", action="store_true", help=t("cli.disable_content_filter"))
    swap.add_argument(
        "--content-filter-provider",
        default=None,
        choices=["none", "stub", "nudenet", "custom"],
        help=t("cli.content_filter_provider"),
    )
    swap.add_argument("--content-filter-threshold", type=float, default=None, help=t("cli.content_filter_threshold"))
    swap.add_argument("--no-block-on-minor", action="store_true", help=t("cli.no_block_on_minor"))
    swap.add_argument("--processors", default=None, help=t("cli.processors"))
    swap.add_argument("--execution-provider", action="append", default=None, help=t("cli.execution_provider"))
    swap.add_argument("--target-face-index", type=int, default=0, help=t("cli.target_face_index"))
    swap.add_argument("--trim-start", type=float, default=None, help=t("cli.trim_start"))
    swap.add_argument("--trim-end", type=float, default=None, help=t("cli.trim_end"))
    swap.add_argument("--video-encoder", default=None, choices=list(VIDEO_ENCODERS), help=t("cli.video_encoder"))
    swap.add_argument("--video-threads", type=int, default=None, help=t("cli.video_threads"))
    swap.add_argument("--no-audio", action="store_true", help=t("cli.no_audio"))
    swap.add_argument("--filter-every", type=int, default=None, help=t("cli.filter_every"))
    swap.add_argument("--enhancer-model", default=None, choices=list(ENHANCER_MODELS), help=t("cli.enhancer_model"))
    swap.add_argument("--enhancer-blend", type=float, default=None, help=t("cli.enhancer_blend"))
    swap.add_argument("--enhancer-weight", type=float, default=None, help=t("cli.enhancer_weight"))
    _add_phase3_flags(swap)
    swap.add_argument("--job-id", default=None, help=t("cli.job_id"))
    swap.add_argument("--resume", action="store_true", help=t("cli.resume"))
    swap.add_argument("--commercial", action="store_true", help=t("cli.commercial"))

    jobs = sub.add_parser("jobs", help=t("cli.jobs_help"))
    jobs.add_argument("--config", default=_DEFAULT_CONFIG, help=t("cli.config"))
    jobs.add_argument("--lang", default=None, help=t("cli.lang"))
    job_cmds = jobs.add_subparsers(dest="job_command", required=True)
    submit = job_cmds.add_parser("submit", help=t("cli.jobs_submit"))
    submit.add_argument("--source", required=True, type=Path, help=t("cli.source"))
    submit.add_argument("--target", required=True, type=Path, help=t("cli.target"))
    submit.add_argument("--output", required=True, type=Path, help=t("cli.output"))
    submit.add_argument("--config", default=_DEFAULT_CONFIG, help=t("cli.config"))
    submit.add_argument("--lang", default=None, help=t("cli.lang"))
    submit.add_argument("--dry-run", action="store_true", help=t("cli.dry_run"))
    submit.add_argument("--processors", default=None, help=t("cli.processors"))
    submit.add_argument("--execution-provider", action="append", default=None, help=t("cli.execution_provider"))
    _add_phase3_flags(submit)
    submit.add_argument("--job-id", default=None, help=t("cli.job_id"))
    resume = job_cmds.add_parser("resume", help=t("cli.jobs_resume"))
    resume.add_argument("job_id", help=t("cli.job_id"))
    resume.add_argument("--config", default=_DEFAULT_CONFIG, help=t("cli.config"))
    resume.add_argument("--lang", default=None, help=t("cli.lang"))
    listing = job_cmds.add_parser("list", help=t("cli.jobs_list"))
    listing.add_argument("--config", default=_DEFAULT_CONFIG, help=t("cli.config"))
    listing.add_argument("--lang", default=None, help=t("cli.lang"))
    run_next = job_cmds.add_parser("run", help=t("cli.jobs_run"))
    run_next.add_argument("--config", default=_DEFAULT_CONFIG, help=t("cli.config"))
    run_next.add_argument("--lang", default=None, help=t("cli.lang"))

    live = sub.add_parser("live", help=t("cli.live_help"))
    live.add_argument("--source", required=True, type=Path, help=t("cli.source"))
    live.add_argument("--config", default=_DEFAULT_CONFIG, help=t("cli.config"))
    live.add_argument("--lang", default=None, help=t("cli.lang"))
    live.add_argument("--camera", type=int, default=None, help=t("cli.camera"))
    live.add_argument("--frames", type=int, default=0, help=t("cli.live_frames"))
    live.add_argument("--no-preview", action="store_true", help=t("cli.no_preview"))
    live.add_argument("--virtual-camera", default=None, help=t("cli.virtual_camera"))
    live.add_argument("--dry-run", action="store_true", help=t("cli.dry_run"))
    live.add_argument("--enable-compliance", action="store_true", help=t("cli.enable_compliance"))
    live.add_argument("--consent-file", type=Path, default=None, help=t("cli.consent_file"))
    live.add_argument("--acknowledge-authorized-use", action="store_true", help=t("cli.acknowledge"))
    live.add_argument("--consent-subject", default=None, help=t("cli.consent_subject"))
    live.add_argument("--consent-statement", default=None, help=t("cli.consent_statement"))
    live.add_argument("--no-watermark", action="store_true", help=t("cli.no_watermark"))
    live.add_argument("--commercial", action="store_true", help=t("cli.commercial"))
    live.add_argument("--execution-provider", action="append", default=None, help=t("cli.execution_provider"))
    _add_phase3_flags(live)

    inspect = sub.add_parser("inspect-models", help=t("cli.inspect_help"))
    inspect.add_argument("--config", default=_DEFAULT_CONFIG, help=t("cli.config"))
    inspect.add_argument("--lang", default=None, help=t("cli.lang"))

    listing = sub.add_parser("list-processors", help=t("cli.processors_help"))
    listing.add_argument("--lang", default=None, help=t("cli.lang"))

    ui = sub.add_parser("ui", help=t("cli.ui_help"))
    ui.add_argument("--config", default=_DEFAULT_CONFIG, help=t("cli.config"))
    ui.add_argument("--lang", default=None, help=t("cli.lang"))
    ui.add_argument("--host", default="127.0.0.1", help=t("cli.host"))
    ui.add_argument("--port", type=int, default=7860, help=t("cli.port"))
    ui.add_argument("--share", action="store_true", help=t("cli.share"))

    bench = sub.add_parser("benchmark", help=t("cli.benchmark_help"))
    bench.add_argument("--config", default=_DEFAULT_CONFIG, help=t("cli.config"))
    bench.add_argument("--lang", default=None, help=t("cli.lang"))
    bench.add_argument("--frames", type=int, default=32, help=t("cli.benchmark_frames"))
    bench.add_argument("--width", type=int, default=256, help=t("cli.benchmark_width"))
    bench.add_argument("--height", type=int, default=256, help=t("cli.benchmark_height"))
    bench.add_argument("--source", type=Path, default=None, help=t("cli.benchmark_source"))
    bench.add_argument("--target", type=Path, default=None, help=t("cli.benchmark_target"))
    bench.add_argument("--dry-run", action="store_true", help=t("cli.dry_run"))
    bench.add_argument("--execution-provider", action="append", default=None, help=t("cli.execution_provider"))
    bench.add_argument("--live", action="store_true", help=t("cli.benchmark_live"))

    doctor = sub.add_parser("doctor", help=t("cli.doctor_help"))
    doctor.add_argument("--config", default=_DEFAULT_CONFIG, help=t("cli.config"))
    doctor.add_argument("--lang", default=None, help=t("cli.lang"))

    download = sub.add_parser("download-models", help=t("cli.download_help"))
    download.add_argument("--config", default=_DEFAULT_CONFIG, help=t("cli.config"))
    download.add_argument("--lang", default=None, help=t("cli.lang"))
    download.add_argument("--with-gfpgan", action="store_true", help=t("cli.download_with_gfpgan"))
    download.add_argument("--force", action="store_true", help=t("cli.download_force"))
    download.add_argument("--from-dir", type=Path, default=None, help=t("cli.download_from_dir"))
    return parser


def _dispatch(namespace: argparse.Namespace) -> int:
    if namespace.command == "doctor":
        from barrett_faceswap.doctor import run_doctor

        config = load_config(namespace.config)
        for line in run_doctor(config):
            print(line)
        return 0
    if namespace.command == "download-models":
        from barrett_faceswap.fetch import download_models

        config = load_config(namespace.config)
        for line in download_models(
            config,
            with_gfpgan=namespace.with_gfpgan,
            force=namespace.force,
            from_dir=namespace.from_dir,
        ):
            print(line)
        return 0
    if namespace.command == "list-processors":
        print(t("cli.processors_header"))
        for name in list_processor_names():
            print(name)
        return 0
    if namespace.command == "ui":
        from barrett_faceswap.ui.app import launch_ui

        launch_ui(Path(namespace.config), host=namespace.host, port=namespace.port, share=namespace.share)
        return 0
    if namespace.command == "jobs":
        return _dispatch_jobs(namespace)
    if namespace.command == "live":
        return _dispatch_live(namespace)
    if namespace.command == "benchmark":
        config = load_config(namespace.config)
        runner = run_live_benchmark if namespace.live else run_benchmark
        report = runner(
            config,
            frames=namespace.frames,
            width=namespace.width,
            height=namespace.height,
            dry_run=namespace.dry_run,
            providers=namespace.execution_provider,
            source_path=namespace.source,
            target_path=namespace.target,
        )
        print(report.line())
        return 0
    config = load_config(namespace.config)
    if namespace.command == "inspect-models":
        print(t("cli.models_header"))
        for line in describe_models(config):
            print(line)
        return 0
    processors = None
    if namespace.processors:
        processors = [item.strip() for item in namespace.processors.split(",") if item.strip()]
    request = SwapRequest(
        source_path=namespace.source,
        target_path=namespace.target,
        output_path=namespace.output,
        dry_run=namespace.dry_run,
        enable_compliance=namespace.enable_compliance,
        consent_file=namespace.consent_file,
        acknowledge_authorized_use=namespace.acknowledge_authorized_use,
        consent_subject=namespace.consent_subject,
        consent_statement=namespace.consent_statement,
        no_watermark=namespace.no_watermark,
        disable_content_filter=namespace.disable_content_filter,
        content_filter_provider=namespace.content_filter_provider,
        content_filter_threshold=namespace.content_filter_threshold,
        no_block_on_minor=namespace.no_block_on_minor,
        processors=processors,
        execution_providers=namespace.execution_provider,
        target_face_index=namespace.target_face_index,
        trim_start=namespace.trim_start,
        trim_end=namespace.trim_end,
        video_encoder=namespace.video_encoder,
        video_threads=namespace.video_threads,
        keep_audio=False if namespace.no_audio else None,
        filter_every_n_frames=namespace.filter_every,
        enhancer_model=namespace.enhancer_model,
        enhancer_blend=namespace.enhancer_blend,
        enhancer_weight=namespace.enhancer_weight,
        swapper_model=namespace.swapper_model,
        pixel_boost=namespace.pixel_boost,
        many_faces=namespace.many_faces,
        reference_path=namespace.reference,
        face_map=namespace.face_map,
        blend_mode=namespace.blend_mode,
        color_transfer=namespace.color_transfer,
        video_mode=namespace.video_mode,
        eye_mode=namespace.eye_mode,
        with_enhancer=bool(namespace.with_enhancer),
        job_id=namespace.job_id,
        resume=namespace.resume,
        commercial=bool(getattr(namespace, "commercial", False)),
    )
    config = apply_request_overrides(config, request)
    result = FaceSwapPipeline(config).run(request)
    for note in result.notes:
        print(note, file=sys.stderr)
    print(t("cli.wrote", path=result.output_path))
    return 0


def _add_phase3_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--swapper-model", default=None, choices=list(SWAPPER_NAMES), help=t("cli.swapper_model"))
    parser.add_argument("--pixel-boost", type=int, default=None, help=t("cli.pixel_boost"))
    parser.add_argument("--many-faces", action="store_true", help=t("cli.many_faces"))
    parser.add_argument("--reference", type=Path, default=None, help=t("cli.reference"))
    parser.add_argument("--face-map", type=Path, default=None, help=t("cli.face_map"))
    parser.add_argument("--blend-mode", default=None, choices=list(BLEND_MODES), help=t("cli.blend_mode"))
    parser.add_argument("--color-transfer", default=None, choices=list(COLOR_TRANSFERS), help=t("cli.color_transfer"))
    parser.add_argument("--video-mode", default=None, choices=list(VIDEO_MODES), help=t("cli.video_mode"))
    parser.add_argument("--eye-mode", default=None, choices=list(EYE_MODES), help=t("cli.eye_mode"))
    parser.add_argument("--with-enhancer", action="store_true", help=t("cli.with_enhancer"))


def _dispatch_jobs(namespace: argparse.Namespace) -> int:
    import secrets

    from barrett_faceswap.jobs import JobStore, request_from_job

    config = load_config(namespace.config)
    store = JobStore(config.jobs.root)
    if namespace.job_command == "list":
        jobs = store.list_jobs()
        if not jobs:
            print(t("jobs.empty"))
            return 0
        for state in jobs:
            print(
                t(
                    "jobs.line",
                    job=state.id,
                    state=state.state,
                    frame=state.next_frame,
                    total="-" if state.total_frames is None else state.total_frames,
                )
            )
        return 0
    if namespace.job_command == "submit":
        job_id = namespace.job_id or secrets.token_hex(4)
        request = _request_from_namespace(namespace, job_id=job_id, resume=False)
        store.create(job_id, request)
        print(t("jobs.queued", job=job_id))
        return 0
    if namespace.job_command == "resume":
        state = store.load(namespace.job_id)
    else:
        state = store.next_runnable()
        if state is None:
            print(t("jobs.empty"))
            return 0
    if state.state == "done":
        print(t("jobs.already_done", job=state.id))
        return 0
    request = request_from_job(state)
    return _execute(load_config(namespace.config), request)


def _request_from_namespace(namespace: argparse.Namespace, *, job_id: str | None, resume: bool) -> SwapRequest:
    processors = None
    if getattr(namespace, "processors", None):
        processors = [item.strip() for item in namespace.processors.split(",") if item.strip()]
    return SwapRequest(
        source_path=namespace.source,
        target_path=namespace.target,
        output_path=namespace.output,
        dry_run=bool(getattr(namespace, "dry_run", False)),
        processors=processors,
        execution_providers=getattr(namespace, "execution_provider", None),
        swapper_model=namespace.swapper_model,
        pixel_boost=namespace.pixel_boost,
        many_faces=bool(namespace.many_faces),
        reference_path=namespace.reference,
        face_map=namespace.face_map,
        blend_mode=namespace.blend_mode,
        color_transfer=namespace.color_transfer,
        video_mode=namespace.video_mode,
        eye_mode=namespace.eye_mode,
        with_enhancer=bool(namespace.with_enhancer),
        job_id=job_id,
        resume=resume,
    )


def _dispatch_live(namespace: argparse.Namespace) -> int:
    from barrett_faceswap.live import LivePipeline, LoopbackWriter, iter_camera

    config = load_config(namespace.config)
    if namespace.commercial:
        config = config.model_copy(update={"execution": config.execution.model_copy(update={"commercial": True})})
    camera = config.live.camera if namespace.camera is None else namespace.camera
    output = Path("outputs") / "live-session.txt"
    request = SwapRequest(
        source_path=namespace.source,
        target_path=Path(f"camera-{camera}"),
        output_path=output,
        dry_run=namespace.dry_run,
        enable_compliance=namespace.enable_compliance,
        consent_file=namespace.consent_file,
        acknowledge_authorized_use=namespace.acknowledge_authorized_use,
        consent_subject=namespace.consent_subject,
        consent_statement=namespace.consent_statement,
        no_watermark=namespace.no_watermark,
        execution_providers=namespace.execution_provider,
        swapper_model=namespace.swapper_model,
        pixel_boost=namespace.pixel_boost,
        many_faces=namespace.many_faces,
        reference_path=namespace.reference,
        face_map=namespace.face_map,
        blend_mode=namespace.blend_mode,
        color_transfer=namespace.color_transfer,
        eye_mode=namespace.eye_mode,
        with_enhancer=bool(namespace.with_enhancer),
        commercial=namespace.commercial,
        live_session=True,
        live_frames=namespace.frames,
    )
    config = apply_request_overrides(config, request)
    writer = LoopbackWriter(namespace.virtual_camera) if namespace.virtual_camera else None
    result, _stats = LivePipeline(config).run(
        request,
        iter_camera(camera, namespace.frames),
        writer=writer,
        preview=not namespace.no_preview,
    )
    for note in result.notes:
        print(note, file=sys.stderr)
    print(t("cli.wrote", path=result.output_path))
    return 0


def _execute(config, request: SwapRequest) -> int:
    config = apply_request_overrides(config, request)
    result = FaceSwapPipeline(config).run(request)
    for note in result.notes:
        print(note, file=sys.stderr)
    print(t("cli.wrote", path=result.output_path))
    return 0


def _activate_from_argv(argv: list[str]) -> None:
    cli_lang = _peek(argv, "--lang")
    config_lang = None
    if cli_lang is None and not os.environ.get("BARRETT_FACESWAP_LANG"):
        config_lang = _peek_config_language(_peek(argv, "--config") or _DEFAULT_CONFIG)
    activate_language(cli=cli_lang, env=os.environ.get("BARRETT_FACESWAP_LANG"), config=config_lang)


def _peek(argv: list[str], flag: str) -> str | None:
    if flag in argv:
        index = argv.index(flag)
        if index + 1 < len(argv) and not argv[index + 1].startswith("-"):
            return argv[index + 1]
    prefix = flag + "="
    for item in argv:
        if item.startswith(prefix):
            return item.split("=", 1)[1]
    return None


def _peek_config_language(path: str) -> str | None:
    config_path = Path(path)
    if not config_path.is_file():
        return None
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(raw, dict):
        return None
    ui = raw.get("ui")
    if isinstance(ui, dict):
        language = ui.get("language")
        return str(language) if language else None
    return None
