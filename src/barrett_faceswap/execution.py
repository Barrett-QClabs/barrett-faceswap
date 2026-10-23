"""Map friendly device names onto ONNX Runtime execution providers.

CUDA is moved to the front when the operator asked for it. CPU stays available
as a fallback. TensorRT options are attached only when that provider is
installed. A CPU build, including CI, skips it and keeps running.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from barrett_faceswap.errors import BarrettFaceswapError
from barrett_faceswap.i18n import t

logger = logging.getLogger(__name__)

_PREPARED = False
_STATUS: dict[str, object] = {}

# Pip wheels that ship the CUDA 13 / cuDNN 9 libraries onnxruntime-gpu 1.31 expects.
_CUDA_LIBS = {
    "cublas": "libcublasLt.so.13",
    "cudnn": "libcudnn.so.9",
}

PROVIDER_MAP: dict[str, str] = {
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "directml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def prefer_cuda(requested: list[str] | tuple[str, ...]) -> list[str]:
    """Put CUDA first and CPU last. Other names keep their relative order."""

    names = list(requested)
    middle = [name for name in names if name not in {"cuda", "cpu"}]
    ordered: list[str] = []
    if "cuda" in names:
        ordered.append("cuda")
    ordered.extend(middle)
    if "cpu" in names:
        ordered.append("cpu")
    return ordered


def available_provider_names() -> list[str]:
    """Return friendly names the local ONNX Runtime build can actually run."""

    ort = _onnxruntime()
    present = set(ort.get_available_providers())
    return [name for name, ort_name in PROVIDER_MAP.items() if ort_name in present]


def resolve_providers(
    requested: list[str] | tuple[str, ...],
    device_id: int,
    settings=None,
    role: str | None = None,
) -> list:
    """Build the provider list passed to ``InferenceSession``.

    ``settings`` is an ``ExecutionConfig``. When it is omitted, providers are
    selected without TensorRT cache options or CUDA graph capture.
    """

    ordered = prefer_cuda(requested)
    ort = _onnxruntime()
    present = set(ort.get_available_providers())
    selected: list = []
    seen: set[str] = set()
    for name in ordered:
        if name not in PROVIDER_MAP:
            raise BarrettFaceswapError(
                t("execution.unknown", name=name, allowed=", ".join(PROVIDER_MAP))
            )
        ort_name = PROVIDER_MAP[name]
        if ort_name not in present:
            logger.warning(t("execution.skipped", name=name, ort_name=ort_name))
            continue
        if ort_name in seen:
            continue
        selected.append(_with_options(ort_name, device_id, settings, role))
        seen.add(ort_name)
    cpu = PROVIDER_MAP["cpu"]
    if cpu in present and cpu not in seen:
        selected.append(cpu)
        seen.add(cpu)
    if not selected:
        raise BarrettFaceswapError(t("execution.none"))
    if settings is not None and settings.cuda_graph and "CUDAExecutionProvider" not in seen:
        logger.info(t("execution.cuda_graph_skipped"))
    logger.info(
        t(
            "execution.selected",
            providers=[item if isinstance(item, str) else item[0] for item in selected],
        )
    )
    return selected


def _with_options(ort_name: str, device_id: int, settings, role: str | None = None):
    if ort_name == "TensorrtExecutionProvider" and settings is not None:
        return (ort_name, tensorrt_options(device_id, settings.tensorrt, role))
    if ort_name == "CUDAExecutionProvider":
        options = {"device_id": int(device_id)}
        if settings is not None and settings.cuda_graph:
            options["enable_cuda_graph"] = "1"
        return (ort_name, options)
    if ort_name == "ROCMExecutionProvider":
        return (ort_name, {"device_id": int(device_id)})
    return ort_name


def tensorrt_options(device_id: int, tensorrt, role: str | None = None) -> dict[str, object]:
    """Provider options for an installed TensorRT runtime. Directories are created here.

    ``role`` becomes the engine-cache prefix so inswapper (``swapper``) and xseg
    (``occluder``) keep separate fp16 engines in the same directory.
    """

    engine = Path(tensorrt.engine_cache)
    timing = Path(tensorrt.timing_cache)
    engine.mkdir(parents=True, exist_ok=True)
    timing.mkdir(parents=True, exist_ok=True)
    logger.info(t("execution.tensorrt_ready", engine=engine, timing=timing))
    options: dict[str, object] = {
        "device_id": int(device_id),
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(engine),
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(timing),
        "trt_fp16_enable": bool(tensorrt.fp16),
        "trt_builder_optimization_level": int(tensorrt.builder_optimization_level),
    }
    if role:
        options["trt_engine_cache_prefix"] = role
    return options


def prepare_runtime() -> dict[str, object]:
    """Prepend NVIDIA pip library directories and preload CUDA DLLs once.

    onnxruntime-gpu 1.31 looks for CUDA 13 (``libcublasLt.so.13``) and cuDNN 9.
    Those files come from ``onnxruntime-gpu[cuda,cudnn]``. Changing
    ``LD_LIBRARY_PATH`` after process start does not reload the dynamic linker,
    so this also calls ``onnxruntime.preload_dlls()`` when that function exists.
    A CPU wheel, including CI, must not raise.
    """

    global _PREPARED
    if _PREPARED:
        return dict(_STATUS)
    _PREPARED = True
    status: dict[str, object] = {
        "preload": "unavailable",
        "error": "",
        "libraries": {name: "" for name in _CUDA_LIBS},
        "library_dirs": [],
    }
    try:
        dirs = _nvidia_lib_dirs()
        status["library_dirs"] = list(dirs)
        if dirs:
            _prepend_library_path(dirs)
        status["libraries"] = _probe_cuda_libs(dirs)
        preload, error = _preload_ort()
        status["preload"] = preload
        status["error"] = error
    except Exception as exc:
        status["preload"] = "failed"
        status["error"] = str(exc)
        logger.info(t("execution.cuda_preload_failed", error=exc))
    _STATUS.clear()
    _STATUS.update(status)
    return dict(_STATUS)


def reset_runtime_prepare() -> None:
    """Allow another preload attempt. Tests use this."""

    global _PREPARED
    _PREPARED = False
    _STATUS.clear()


def _onnxruntime():
    prepare_runtime()
    try:
        import onnxruntime
    except ImportError as exc:
        raise BarrettFaceswapError(t("execution.ort_missing")) from exc
    return onnxruntime


def _nvidia_lib_dirs() -> list[str]:
    roots: list[Path] = []
    try:
        import nvidia

        roots.append(Path(nvidia.__file__).resolve().parent)
    except Exception:
        pass
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry) / "nvidia"
        if candidate.is_dir():
            roots.append(candidate)
    found: list[str] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for lib in sorted(root.glob("*/lib")):
            if not lib.is_dir():
                continue
            text = str(lib)
            if text in seen:
                continue
            seen.add(text)
            found.append(text)
    return found


def _prepend_library_path(dirs: list[str]) -> None:
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [item for item in current.split(":") if item]
    merged: list[str] = []
    for item in list(dirs) + parts:
        if item not in merged:
            merged.append(item)
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)
    if hasattr(os, "add_dll_directory"):
        for directory in dirs:
            try:
                os.add_dll_directory(directory)  # type: ignore[attr-defined]
            except OSError:
                continue


def _probe_cuda_libs(dirs: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for name, filename in _CUDA_LIBS.items():
        path = ""
        for directory in dirs:
            candidate = Path(directory) / filename
            if candidate.is_file():
                path = str(candidate)
                break
        found[name] = path
    return found


def _preload_ort() -> tuple[str, str]:
    try:
        import onnxruntime
    except ImportError:
        logger.info(t("execution.cuda_preload_skip"))
        return "unavailable", ""
    preload = getattr(onnxruntime, "preload_dlls", None)
    if preload is None:
        logger.info(t("execution.cuda_preload_skip"))
        return "unavailable", ""
    try:
        preload()
    except Exception as exc:
        logger.info(t("execution.cuda_preload_failed", error=exc))
        return "failed", str(exc)
    logger.info(t("execution.cuda_preload_ok"))
    return "ok", ""
