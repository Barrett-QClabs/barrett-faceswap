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
from barrett_faceswap.swap_catalog import CROSSFACE_FILES, SWAPPERS


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


def _swapper_records() -> dict[str, ModelRecord]:
    """Registry rows for swap graphs other than the original inswapper role."""

    records: dict[str, ModelRecord] = {}
    for spec in SWAPPERS.values():
        if spec.role == "swapper" or spec.role in records:
            continue
        records[spec.role] = ModelRecord(
            role=spec.role,
            filename=spec.filename,
            sha256=None,
            license_key=spec.license_key,
            commercial_use=spec.commercial_use,
            alignment=spec.alignment,
            input_size=(spec.native_size, spec.native_size),
            download_url=spec.download_url,
        )
    for role, filename in CROSSFACE_FILES.items():
        records[role] = ModelRecord(
            role=role,
            filename=filename,
            sha256=None,
            license_key="models.license.crossface",
            commercial_use=False,
            alignment="embedding_512",
            input_size=(1, 512),
            download_url="",
        )
    return records


_EXTRA_RECORDS = _swapper_records()

ROLE_ORDER = (
    "detector",
    "recognizer",
    "swapper",
    "hyperswap_1a",
    "hyperswap_1b",
    "hyperswap_1c",
    "simswap_256",
    "simswap_512",
    "ghost_1",
    "ghost_2",
    "ghost_3",
    "blendswap_256",
    "crossface_hyperswap",
    "crossface_simswap",
    "crossface_ghost",
    "gfpgan",
    "codeformer",
    "gpen_bfr_512",
    "gpen_bfr_1024",
    "gpen_bfr_2048",
    "restoreformer_plus_plus",
    "occluder",
    "xseg_2",
    "xseg_3",
    "sam2_refine",
    "parser",
)

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
    "gfpgan": ModelRecord(
        role="gfpgan",
        filename="gfpgan_1.4.onnx",
        sha256=None,
        license_key="models.license.gfpgan",
        commercial_use=False,
        alignment="ffhq_512",
        input_size=(512, 512),
        download_url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    ),
    "codeformer": ModelRecord(
        role="codeformer",
        filename="codeformer.onnx",
        sha256=None,
        license_key="models.license.codeformer",
        commercial_use=False,
        alignment="ffhq_512",
        input_size=(512, 512),
        download_url="https://github.com/sczhou/CodeFormer",
    ),
    "occluder": ModelRecord(
        role="occluder",
        filename="occlusion_xseg.onnx",
        sha256=None,
        license_key="models.license.occluder",
        commercial_use=False,
        alignment="xseg",
        input_size=(256, 256),
        download_url="https://github.com/iperov/DeepFaceLab",
    ),
    "parser": ModelRecord(
        role="parser",
        filename="bisenet_face_parsing.onnx",
        sha256=None,
        license_key="models.license.parser",
        commercial_use=False,
        alignment="bisenet_19",
        input_size=(512, 512),
        download_url="https://github.com/zllrunning/face-parsing.PyTorch",
    ),
    "gpen_bfr_512": ModelRecord(
        role="gpen_bfr_512",
        filename="gpen_bfr_512.onnx",
        sha256=None,
        license_key="models.license.gpen",
        commercial_use=False,
        alignment="ffhq_512",
        input_size=(512, 512),
        download_url="https://github.com/yangxy/GPEN",
    ),
    "gpen_bfr_1024": ModelRecord(
        role="gpen_bfr_1024",
        filename="gpen_bfr_1024.onnx",
        sha256=None,
        license_key="models.license.gpen",
        commercial_use=False,
        alignment="ffhq_1024",
        input_size=(1024, 1024),
        download_url="https://github.com/yangxy/GPEN",
    ),
    "gpen_bfr_2048": ModelRecord(
        role="gpen_bfr_2048",
        filename="gpen_bfr_2048.onnx",
        sha256=None,
        license_key="models.license.gpen",
        commercial_use=False,
        alignment="ffhq_2048",
        input_size=(2048, 2048),
        download_url="https://github.com/yangxy/GPEN",
    ),
    "restoreformer_plus_plus": ModelRecord(
        role="restoreformer_plus_plus",
        filename="restoreformer_plus_plus.onnx",
        sha256=None,
        license_key="models.license.restoreformer",
        commercial_use=False,
        alignment="ffhq_512",
        input_size=(512, 512),
        download_url="https://github.com/wzhouxiff/RestoreFormerPlusPlus",
    ),
    "xseg_2": ModelRecord(
        role="xseg_2",
        filename="xseg_2.onnx",
        sha256=None,
        license_key="models.license.occluder",
        commercial_use=False,
        alignment="xseg",
        input_size=(256, 256),
        download_url="https://github.com/facefusion/facefusion-assets",
    ),
    "xseg_3": ModelRecord(
        role="xseg_3",
        filename="xseg_3.onnx",
        sha256=None,
        license_key="models.license.occluder",
        commercial_use=False,
        alignment="xseg",
        input_size=(256, 256),
        download_url="https://github.com/facefusion/facefusion-assets",
    ),
    "sam2_refine": ModelRecord(
        role="sam2_refine",
        filename="sam2_refine.onnx",
        sha256=None,
        license_key="models.license.sam2",
        commercial_use=False,
        alignment="mask_refine",
        input_size=(256, 256),
        download_url="",
    ),
}
BUILTIN.update(_EXTRA_RECORDS)


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
    for role in ROLE_ORDER:
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
