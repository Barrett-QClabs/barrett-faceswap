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
from barrett_faceswap.benchmark import run_benchmark
from barrett_faceswap.config import apply_request_overrides, load_config
from barrett_faceswap.config import ENHANCER_MODELS, VIDEO_ENCODERS
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
    bench.add_argument("--dry-run", action="store_true", help=t("cli.dry_run"))
    bench.add_argument("--execution-provider", action="append", default=None, help=t("cli.execution_provider"))
    return parser


def _dispatch(namespace: argparse.Namespace) -> int:
    if namespace.command == "list-processors":
        print(t("cli.processors_header"))
        for name in list_processor_names():
            print(name)
        return 0
    if namespace.command == "ui":
        from barrett_faceswap.ui.app import launch_ui

        launch_ui(Path(namespace.config), host=namespace.host, port=namespace.port, share=namespace.share)
        return 0
    if namespace.command == "benchmark":
        config = load_config(namespace.config)
        report = run_benchmark(
            config,
            frames=namespace.frames,
            width=namespace.width,
            height=namespace.height,
            dry_run=namespace.dry_run,
            providers=namespace.execution_provider,
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
    )
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
