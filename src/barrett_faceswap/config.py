"""Pydantic schema for ``configs/default.yaml``.

The CLI loads this same model. Relative paths resolve from the working directory.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from barrett_faceswap.errors import ConfigError
from barrett_faceswap.i18n import t

FILTER_PROVIDERS = ("none", "stub", "nudenet", "custom")
EXECUTION_PROVIDERS = ("cuda", "tensorrt", "rocm", "coreml", "directml", "cpu")
ENHANCER_MODELS = ("gfpgan_1.4", "codeformer")
VIDEO_ENCODERS = ("libx264", "h264_nvenc")


def resolve_config_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class UiConfig(_Model):
    language: str = "en"

    @field_validator("language", mode="before")
    @classmethod
    def _language(cls, value: object) -> str:
        from barrett_faceswap.i18n import normalize_language

        normalized = normalize_language(str(value))
        if normalized is None:
            raise ValueError(t("config.bad_language", value=value))
        return normalized


class ModelFile(_Model):
    path: Path
    sha256: str | None = None

    @field_validator("path", mode="before")
    @classmethod
    def _path(cls, value: object) -> Path:
        return resolve_config_path(value)

    @field_validator("sha256", mode="before")
    @classmethod
    def _sha(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"", "none", "null"}:
            return None
        if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
            raise ValueError(t("config.bad_sha"))
        return text


def _optional_model(relative: str) -> ModelFile:
    return ModelFile(path=relative)


class ModelsConfig(_Model):
    detector: ModelFile
    recognizer: ModelFile
    swapper: ModelFile
    gfpgan: ModelFile = Field(default_factory=lambda: _optional_model("models/gfpgan_1.4.onnx"))
    codeformer: ModelFile = Field(default_factory=lambda: _optional_model("models/codeformer.onnx"))
    occluder: ModelFile = Field(default_factory=lambda: _optional_model("models/occlusion_xseg.onnx"))
    parser: ModelFile = Field(default_factory=lambda: _optional_model("models/bisenet_face_parsing.onnx"))


class ExecutionConfig(_Model):
    providers: list[str] = Field(default_factory=lambda: ["cuda", "cpu"])
    device_id: int = 0

    @field_validator("providers")
    @classmethod
    def _providers(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError(t("config.providers_empty"))
        unknown = [item for item in value if item not in EXECUTION_PROVIDERS]
        if unknown:
            raise ValueError(
                t(
                    "config.providers_unknown",
                    unknown=", ".join(unknown),
                    allowed=", ".join(EXECUTION_PROVIDERS),
                )
            )
        return list(value)


class DetectConfig(_Model):
    det_size: int = 640
    score_threshold: float = 0.5
    nms_threshold: float = 0.4

    @field_validator("det_size")
    @classmethod
    def _det_size(cls, value: int) -> int:
        if value <= 0 or value % 32 != 0:
            raise ValueError(t("config.det_size"))
        return value


class SwapConfig(_Model):
    resolution: int = 128


class BlendConfig(_Model):
    mask_blur: int = 15


class EnhancerConfig(_Model):
    """GFPGAN 1.4 by default. CodeFormer is selected with ``model: codeformer``."""

    model: str = "gfpgan_1.4"
    blend: float = 0.8
    weight: float = 0.5

    @field_validator("model")
    @classmethod
    def _model(cls, value: str) -> str:
        if value not in ENHANCER_MODELS:
            raise ValueError(t("config.enhancer_model", allowed=", ".join(ENHANCER_MODELS)))
        return value

    @field_validator("blend", "weight")
    @classmethod
    def _unit(cls, value: float) -> float:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(t("config.unit_interval"))
        return float(value)


class MasksConfig(_Model):
    """Occlusion and parsing run only when the ONNX file is present."""

    occlusion: bool = True
    parser: bool = True
    occlusion_invert: bool = False
    regions: list[str] = Field(default_factory=lambda: list(_default_regions()))

    @field_validator("regions")
    @classmethod
    def _regions(cls, value: list[str]) -> list[str]:
        from barrett_faceswap.parsing import PARSING_REGIONS

        if not value:
            raise ValueError(t("config.regions_empty"))
        unknown = [item for item in value if item not in PARSING_REGIONS]
        if unknown:
            raise ValueError(
                t(
                    "config.regions_unknown",
                    unknown=", ".join(unknown),
                    allowed=", ".join(PARSING_REGIONS),
                )
            )
        return [str(item) for item in value]


class VideoConfig(_Model):
    """Disk frame pipeline. ``libx264`` is the portable default."""

    encoder: str = "libx264"
    threads: int = 4
    filter_every_n_frames: int = 30
    keep_audio: bool = True
    crf: int = 18

    @field_validator("encoder")
    @classmethod
    def _encoder(cls, value: str) -> str:
        if value not in VIDEO_ENCODERS:
            raise ValueError(t("config.encoder", allowed=", ".join(VIDEO_ENCODERS)))
        return value

    @field_validator("threads")
    @classmethod
    def _threads(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError(t("config.threads"))
        return int(value)

    @field_validator("filter_every_n_frames")
    @classmethod
    def _every(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError(t("config.filter_every"))
        return int(value)

    @field_validator("crf")
    @classmethod
    def _crf(cls, value: int) -> int:
        if not 0 <= int(value) <= 51:
            raise ValueError(t("config.crf"))
        return int(value)


class ConsentConfig(_Model):
    required: bool = True


class WatermarkConfig(_Model):
    enabled: bool = True
    text: str | None = None


class ContentFilterConfig(_Model):
    enabled: bool = True
    provider: str = "stub"
    threshold: float = 0.6
    block_on_minor: bool = True
    custom_entrypoint: str | None = None

    @field_validator("provider")
    @classmethod
    def _provider(cls, value: str) -> str:
        if value not in FILTER_PROVIDERS:
            raise ValueError(t("config.provider_enum", allowed=", ".join(FILTER_PROVIDERS)))
        return value

    @field_validator("threshold")
    @classmethod
    def _threshold(cls, value: float) -> float:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(t("config.threshold"))
        return float(value)


class AuditConfig(_Model):
    enabled: bool = True
    path: Path = Field(default_factory=lambda: resolve_config_path("outputs/audit/audit.jsonl"))

    @field_validator("path", mode="before")
    @classmethod
    def _path(cls, value: object) -> Path:
        return resolve_config_path(value)


class ComplianceConfig(_Model):
    """Production profile.

    ``enabled`` is the master switch. Nested fields apply only after an
    operator turns enforcement on. The shipped default is off so local
    dry-runs are not blocked.
    """

    enabled: bool = False
    consent: ConsentConfig = Field(default_factory=ConsentConfig)
    watermark: WatermarkConfig = Field(default_factory=WatermarkConfig)
    content_filter: ContentFilterConfig = Field(default_factory=ContentFilterConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)


class AppConfig(_Model):
    """Full project config. Constructed by the CLI and by library callers."""

    ui: UiConfig = Field(default_factory=UiConfig)
    models: ModelsConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)
    swap: SwapConfig = Field(default_factory=SwapConfig)
    blend: BlendConfig = Field(default_factory=BlendConfig)
    enhancer: EnhancerConfig = Field(default_factory=EnhancerConfig)
    masks: MasksConfig = Field(default_factory=MasksConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    processors: list[str] = Field(default_factory=lambda: ["face_swapper"])
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)

    @field_validator("processors")
    @classmethod
    def _processors(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError(t("config.processors_empty"))
        return [str(item) for item in value]


def _default_regions() -> tuple[str, ...]:
    from barrett_faceswap.parsing import DEFAULT_PARSE_REGIONS

    return DEFAULT_PARSE_REGIONS


def apply_request_overrides(config: AppConfig, request: object) -> AppConfig:
    """Copy CLI overrides onto the config the processors read.

    ``request`` is a ``SwapRequest``. The annotation stays loose so this module
    does not import the types module at load time.
    """

    enhancer = config.enhancer
    video = config.video
    model = getattr(request, "enhancer_model", None)
    blend = getattr(request, "enhancer_blend", None)
    weight = getattr(request, "enhancer_weight", None)
    encoder = getattr(request, "video_encoder", None)
    threads = getattr(request, "video_threads", None)
    every = getattr(request, "filter_every_n_frames", None)
    keep_audio = getattr(request, "keep_audio", None)
    if model is not None:
        enhancer = enhancer.model_copy(update={"model": model})
    if blend is not None:
        enhancer = enhancer.model_copy(update={"blend": blend})
    if weight is not None:
        enhancer = enhancer.model_copy(update={"weight": weight})
    if encoder is not None:
        video = video.model_copy(update={"encoder": encoder})
    if threads is not None:
        video = video.model_copy(update={"threads": threads})
    if every is not None:
        video = video.model_copy(update={"filter_every_n_frames": every})
    if keep_audio is not None:
        video = video.model_copy(update={"keep_audio": keep_audio})
    return config.model_copy(update={"enhancer": enhancer, "video": video})


def load_config(path: str | Path) -> AppConfig:
    """Read and validate a YAML config file."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(t("config.not_found", path=config_path))
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(t("config.parse_error", path=config_path, error=exc)) from exc
    if not isinstance(raw, dict):
        raise ConfigError(t("config.not_mapping", path=config_path))
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(t("config.validation", detail=exc)) from exc
