"""Map friendly device names onto ONNX Runtime execution providers.

CUDA is moved to the front when the operator asked for it. CPU stays available
as a fallback. TensorRT can be named, but Phase 1 does not configure an engine
cache; that tuning is left for a later pass.
"""

from __future__ import annotations

import logging

from barrett_faceswap.errors import BarrettFaceswapError
from barrett_faceswap.i18n import t

logger = logging.getLogger(__name__)

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


def resolve_providers(requested: list[str] | tuple[str, ...], device_id: int) -> list:
    """Build the provider list passed to ``InferenceSession``."""

    ordered = prefer_cuda(requested)
    if "tensorrt" in ordered:
        logger.info(t("execution.tensorrt_deferred"))
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
        selected.append(_with_options(ort_name, device_id))
        seen.add(ort_name)
    cpu = PROVIDER_MAP["cpu"]
    if cpu in present and cpu not in seen:
        selected.append(cpu)
        seen.add(cpu)
    if not selected:
        raise BarrettFaceswapError(t("execution.none"))
    logger.info(
        t(
            "execution.selected",
            providers=[item if isinstance(item, str) else item[0] for item in selected],
        )
    )
    return selected


def _with_options(ort_name: str, device_id: int):
    if ort_name in {"CUDAExecutionProvider", "TensorrtExecutionProvider", "ROCMExecutionProvider"}:
        return (ort_name, {"device_id": int(device_id)})
    return ort_name


def _onnxruntime():
    try:
        import onnxruntime
    except ImportError as exc:
        raise BarrettFaceswapError(t("execution.ort_missing")) from exc
    return onnxruntime
