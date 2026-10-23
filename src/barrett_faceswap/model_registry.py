"""Built-in descriptions of the ONNX weights this pipeline knows how to run.

The registry holds the license, the commercial-use flag, the alignment
template name, and the network input size. Paths and optional sha256 pins
come from the pydantic config so a checkout never needs to vendor the files.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import ModelNotFoundError
from barrett_faceswap.i18n import t


@dataclass(frozen=True)
class ModelRecord:
    """One weight the loader can hash-check and execute."""

    role: str
    filename: str
    sha256: str | None
    license_key: str
    commercial_use: bool
    alignment: str
    input_size: tuple[int, int]
    download_url: str


BUILTIN: dict[str, ModelRecord] = {
    "detector": ModelRecord(
        role="detector",
        filename="det_10g.onnx",
        sha256=None,
        license_key="models.license.insightface",
        commercial_use=False,
        alignment="scrfd",
        input_size=(640, 640),
        download_url="https://github.com/deepinsight/insightface/releases/tag/model-zoo",
    ),
    "recognizer": ModelRecord(
        role="recognizer",
        filename="w600k_r50.onnx",
        sha256=None,
        license_key="models.license.insightface",
        commercial_use=False,
        alignment="arcface_112",
        input_size=(112, 112),
        download_url="https://github.com/deepinsight/insightface/releases/tag/model-zoo",
    ),
    "swapper": ModelRecord(
        role="swapper",
        filename="inswapper_128.onnx",
        sha256=None,
        license_key="models.license.insightface",
        commercial_use=False,
        alignment="inswapper_128",
        input_size=(128, 128),
        download_url="https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx",
    ),
}


def registry_record(config: AppConfig, role: str) -> ModelRecord:
    """Merge a built-in record with the operator's sha256 pin."""

    try:
        base = BUILTIN[role]
    except KeyError as exc:
        raise ModelNotFoundError(t("models.unknown_role", role=role)) from exc
    pinned = getattr(config.models, role).sha256 or base.sha256
    if pinned == base.sha256:
        return base
    return replace(base, sha256=pinned)


def describe_models(config: AppConfig) -> list[str]:
    """Human-readable registry rows for ``inspect-models``."""

    from barrett_faceswap.models import file_sha256

    lines: list[str] = []
    for role in ("detector", "recognizer", "swapper"):
        record = registry_record(config, role)
        path = getattr(config.models, role).path
        if path.is_file():
            digest = file_sha256(path)
            state = t("models.inspect_present")
        else:
            digest = "-"
            state = t("models.inspect_missing")
        commercial = t("models.commercial_yes") if record.commercial_use else t("models.commercial_no")
        width, height = record.input_size
        lines.append(
            t(
                "models.record",
                role=role,
                state=state,
                filename=record.filename,
                alignment=record.alignment,
                input_size=f"{width}x{height}",
                commercial=commercial,
                pin=record.sha256 or "-",
                sha256=digest,
                path=path,
                license=t(record.license_key),
            )
        )
    return lines
