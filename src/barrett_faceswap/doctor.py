"""Report providers, CUDA libraries, and local model files.

``barrett-faceswap doctor`` is safe on a CPU checkout. It does not download
weights and it does not fail when CUDA is absent.
"""

from __future__ import annotations

from importlib import metadata

from barrett_faceswap.config import AppConfig
from barrett_faceswap.execution import available_provider_names, prepare_runtime
from barrett_faceswap.i18n import t
from barrett_faceswap.models import file_sha256

# The swap trio plus the optional files a GPU checkout usually asks about.
_DOCTOR_ROLES = ("detector", "recognizer", "swapper", "gfpgan", "occluder", "parser")


def run_doctor(config: AppConfig) -> list[str]:
    """One line per fact. The CLI prints them."""

    status = prepare_runtime()
    providers = available_provider_names()
    lines = [t("doctor.providers", providers=", ".join(providers) if providers else "-")]
    installed = installed_ort_names()
    if "onnxruntime" in installed and "onnxruntime-gpu" in installed:
        lines.append(t("doctor.both_wheels"))
    preload = str(status.get("preload") or "unavailable")
    if preload == "failed":
        label = t("doctor.preload_failed", error=status.get("error") or "-")
    elif preload == "ok":
        label = t("doctor.preload_ok")
    else:
        label = t("doctor.preload_unavailable")
    lines.append(t("doctor.preload", status=label))
    libraries = status.get("libraries")
    if isinstance(libraries, dict):
        for name in ("cublas", "cudnn"):
            path = str(libraries.get(name) or "")
            if path:
                lines.append(t("doctor.lib", name=name, path=path))
            else:
                lines.append(t("doctor.lib_missing", name=name))
    for role in _DOCTOR_ROLES:
        path = getattr(config.models, role).path
        if path.is_file():
            digest = file_sha256(path)
            state = t("doctor.present")
        else:
            digest = "-"
            state = t("doctor.missing")
        lines.append(t("doctor.model", role=role, state=state, sha256=digest, path=path))
    return lines


def installed_ort_names() -> list[str]:
    """Distribution names for the CPU and GPU ONNX Runtime wheels, if present."""

    found: list[str] = []
    for dist in metadata.distributions():
        name = str(dist.metadata["Name"] or "").lower()
        if name in {"onnxruntime", "onnxruntime-gpu"} and name not in found:
            found.append(name)
    return found
