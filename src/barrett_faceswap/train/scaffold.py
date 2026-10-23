"""Dataset check, a training plan, and the ONNX export contract.

This module does not download data, does not open a training loop, and does
not copy another project's trainer. It checks a manifest you already have
and prints the steps a commercial training run would follow.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from barrett_faceswap.errors import ConfigError
from barrett_faceswap.i18n import t

COMMERCIAL_LICENSE = "commercial-ok"


class SampleRecord(BaseModel):
    """One pair in a local manifest. Paths are not fetched."""

    model_config = ConfigDict(extra="ignore")

    identity: str
    source: str
    target: str
    license: str


class DatasetSpec(BaseModel):
    """A local description of a training set."""

    model_config = ConfigDict(extra="ignore")

    name: str
    resolution: int = 256
    license: str
    samples: list[SampleRecord] = Field(default_factory=list)

    def commercial(self) -> bool:
        if self.license != COMMERCIAL_LICENSE:
            return False
        return all(sample.license == COMMERCIAL_LICENSE for sample in self.samples)


def validate_dataset(path: Path) -> DatasetSpec:
    """Read a YAML manifest. Refuse a license that is not ``commercial-ok``."""

    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(t("train.missing", path=file_path))
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        spec = DatasetSpec.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(t("train.bad_spec", error=exc)) from exc
    if spec.resolution not in {256, 512}:
        raise ConfigError(t("train.resolution", size=spec.resolution))
    if not spec.commercial():
        raise ConfigError(t("train.bad_license", license=spec.license))
    return spec


def training_plan(spec: DatasetSpec) -> list[str]:
    """Ordered steps. Calling this does not train."""

    return [
        t("train.step_check", name=spec.name, count=len(spec.samples), size=spec.resolution),
        t("train.step_batches"),
        t("train.step_identity"),
        t("train.step_generator"),
        t("train.step_export"),
        t("train.step_register"),
    ]


def export_contract() -> str:
    """The ONNX this repository would run after you export your own weights."""

    return t("train.export_contract")
