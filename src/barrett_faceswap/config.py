"""Pydantic schema for ``configs/default.yaml``.

The CLI loads this same model. Relative paths resolve from the working directory.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

from barrett_faceswap.errors import ConfigError
from barrett_faceswap.i18n import t

FILTER_PROVIDERS = ("none", "stub", "nudenet", "custom")
EXECUTION_PROVIDERS = ("cuda", "tensorrt", "rocm", "coreml", "directml", "cpu")
ENHANCER_MODELS = (
    "gfpgan_1.4",
    "codeformer",
    "gpen_bfr_512",
    "gpen_bfr_1024",
    "gpen_bfr_2048",
    "restoreformer_plus_plus",
)
VIDEO_ENCODERS = ("libx264", "h264_nvenc")


def resolve_config_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class UiConfig(_Model):
    language: str = "zh-CN"

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
    hyperswap_1a: ModelFile = Field(default_factory=lambda: _optional_model("models/hyperswap_1a_256.onnx"))
    hyperswap_1b: ModelFile = Field(default_factory=lambda: _optional_model("models/hyperswap_1b_256.onnx"))
    hyperswap_1c: ModelFile = Field(default_factory=lambda: _optional_model("models/hyperswap_1c_256.onnx"))
    simswap_256: ModelFile = Field(default_factory=lambda: _optional_model("models/simswap_256.onnx"))
    simswap_512: ModelFile = Field(default_factory=lambda: _optional_model("models/simswap_unofficial_512.onnx"))
    ghost_1: ModelFile = Field(default_factory=lambda: _optional_model("models/ghost_1_256.onnx"))
    ghost_2: ModelFile = Field(default_factory=lambda: _optional_model("models/ghost_2_256.onnx"))
    ghost_3: ModelFile = Field(default_factory=lambda: _optional_model("models/ghost_3_256.onnx"))
    crossface_hyperswap: ModelFile = Field(default_factory=lambda: _optional_model("models/crossface_hyperswap.onnx"))
    crossface_simswap: ModelFile = Field(default_factory=lambda: _optional_model("models/crossface_simswap.onnx"))
    crossface_ghost: ModelFile = Field(default_factory=lambda: _optional_model("models/crossface_ghost.onnx"))
    blendswap_256: ModelFile = Field(default_factory=lambda: _optional_model("models/blendswap_256.onnx"))
    gpen_bfr_512: ModelFile = Field(default_factory=lambda: _optional_model("models/gpen_bfr_512.onnx"))
    gpen_bfr_1024: ModelFile = Field(default_factory=lambda: _optional_model("models/gpen_bfr_1024.onnx"))
    gpen_bfr_2048: ModelFile = Field(default_factory=lambda: _optional_model("models/gpen_bfr_2048.onnx"))
    restoreformer_plus_plus: ModelFile = Field(
        default_factory=lambda: _optional_model("models/restoreformer_plus_plus.onnx")
    )
    xseg_2: ModelFile = Field(default_factory=lambda: _optional_model("models/xseg_2.onnx"))
    xseg_3: ModelFile = Field(default_factory=lambda: _optional_model("models/xseg_3.onnx"))
    sam2_refine: ModelFile = Field(default_factory=lambda: _optional_model("models/sam2_refine.onnx"))
    landmark: ModelFile = Field(default_factory=lambda: _optional_model("models/2dfan4.onnx"))


class TensorRTConfig(_Model):
    """Used only when the TensorRT execution provider is actually installed."""

    engine_cache: Path = Field(default_factory=lambda: resolve_config_path(".cache/tensorrt/engines"))
    timing_cache: Path = Field(default_factory=lambda: resolve_config_path(".cache/tensorrt/timing"))
    fp16: bool = False
    builder_optimization_level: int = 3

    @field_validator("engine_cache", "timing_cache", mode="before")
    @classmethod
    def _cache(cls, value: object) -> Path:
        return resolve_config_path(value)

    @field_validator("builder_optimization_level")
    @classmethod
    def _level(cls, value: int) -> int:
        if not 0 <= int(value) <= 5:
            raise ValueError(t("config.builder_level"))
        return int(value)


class ExecutionConfig(_Model):
    providers: list[str] = Field(default_factory=lambda: ["cuda", "cpu"])
    device_id: int = 0
    precision: str = "fp32"
    cuda_graph: bool = False
    commercial: bool = False
    tensorrt: TensorRTConfig = Field(default_factory=TensorRTConfig)

    @field_validator("precision")
    @classmethod
    def _precision(cls, value: str) -> str:
        if value not in {"fp32", "fp16"}:
            raise ValueError(t("config.precision", value=value))
        return value

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

    @field_validator("score_threshold")
    @classmethod
    def _score_threshold(cls, value: float) -> float:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(t("config.unit_interval"))
        return float(value)


class SwapConfig(_Model):
    """``model`` selects the registry swapper. ``pixel_boost`` 0 keeps ``resolution`` for inswapper."""

    model: str = "hyperswap_1a_256"
    resolution: int = 128
    pixel_boost: int = 0
    selector: str = "index"
    reference: Path | None = None
    face_map: Path | None = None
    similarity: float = 0.45
    track: bool = True
    track_iou: float = 0.3

    @field_validator("model")
    @classmethod
    def _model_name(cls, value: str) -> str:
        from barrett_faceswap.swap_catalog import SWAPPER_NAMES

        if value not in SWAPPER_NAMES:
            raise ValueError(t("config.swapper_model", model=value, allowed=", ".join(SWAPPER_NAMES)))
        return value

    @field_validator("selector")
    @classmethod
    def _selector(cls, value: str) -> str:
        from barrett_faceswap.swap_catalog import SELECTORS

        if value not in SELECTORS:
            raise ValueError(t("config.selector", selector=value, allowed=", ".join(SELECTORS)))
        return value

    @field_validator("pixel_boost")
    @classmethod
    def _boost(cls, value: int) -> int:
        if int(value) < 0:
            raise ValueError(t("swap.boost", size=value, model="-", allowed="0"))
        return int(value)

    @field_validator("similarity", "track_iou")
    @classmethod
    def _ratio(cls, value: float) -> float:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(t("config.unit_interval"))
        return float(value)

    @field_validator("reference", "face_map", mode="before")
    @classmethod
    def _optional_path(cls, value: object) -> Path | None:
        if value is None:
            return None
        text = str(value).strip()
        if text in {"", "none", "null"}:
            return None
        return resolve_config_path(text)

    @model_validator(mode="after")
    def _boost_matches_model(self) -> "SwapConfig":
        from barrett_faceswap.swap_catalog import SWAPPERS

        spec = SWAPPERS[self.model]
        if self.pixel_boost and self.pixel_boost not in spec.boost_sizes:
            allowed = ", ".join(str(size) for size in spec.boost_sizes)
            raise ValueError(t("swap.boost", size=self.pixel_boost, model=self.model, allowed=allowed))
        return self


class BlendConfig(_Model):
    mask_blur: int = 15
    mode: str = "alpha"
    color: str = "lab"
    eye_mode: str = "blend"
    eye_sharpen: float = 0.6

    @field_validator("mode")
    @classmethod
    def _mode(cls, value: str) -> str:
        from barrett_faceswap.swap_catalog import BLEND_MODES

        if value not in BLEND_MODES:
            raise ValueError(t("config.blend_mode", mode=value, allowed=", ".join(BLEND_MODES)))
        return value

    @field_validator("color")
    @classmethod
    def _color(cls, value: str) -> str:
        from barrett_faceswap.swap_catalog import COLOR_TRANSFERS

        if value not in COLOR_TRANSFERS:
            raise ValueError(t("config.color", mode=value, allowed=", ".join(COLOR_TRANSFERS)))
        return value

    @field_validator("eye_mode")
    @classmethod
    def _eye_mode(cls, value: str) -> str:
        from barrett_faceswap.swap_catalog import EYE_MODES

        if value not in EYE_MODES:
            raise ValueError(t("config.eye_mode", mode=value, allowed=", ".join(EYE_MODES)))
        return value

    @field_validator("eye_sharpen")
    @classmethod
    def _eye_sharpen(cls, value: float) -> float:
        if not 0.0 <= float(value) <= 3.0:
            raise ValueError(t("config.eye_sharpen"))
        return float(value)


class EnhancerConfig(_Model):
    """GFPGAN 1.4 by default. CodeFormer is selected with ``model: codeformer``."""

    model: str = "gfpgan_1.4"
    blend: float = 0.6
    weight: float = 0.7
    # Lower than ``blend`` so GFPGAN does not replace the target eyes.
    eye_blend: float = 0.35
    # Mix of the previous enhanced crop when the face barely moves. 0 disables it.
    temporal: float = 0.4
    temporal_motion: float = 0.04

    @field_validator("model")
    @classmethod
    def _model(cls, value: str) -> str:
        if value not in ENHANCER_MODELS:
            raise ValueError(t("config.enhancer_model", allowed=", ".join(ENHANCER_MODELS)))
        return value

    @field_validator("blend", "weight", "eye_blend", "temporal", "temporal_motion")
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
    occluder_model: str = "xseg_1"
    # Video and live recompute ONNX masks every this many frames. 1 is every frame.
    # 3 reuses the warped mask for two frames. A landmark jump still recomputes.
    stride: int = 3
    # Mean landmark shift / face width. Above this, the previous mask is dropped.
    stride_motion: float = 0.04
    regions: list[str] = Field(default_factory=lambda: list(_default_regions()))

    @field_validator("stride")
    @classmethod
    def _stride(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError(t("config.mask_stride"))
        return int(value)

    @field_validator("occluder_model")
    @classmethod
    def _occluder_model(cls, value: str) -> str:
        from barrett_faceswap.swap_catalog import OCCLUDER_MODELS

        if value not in OCCLUDER_MODELS:
            raise ValueError(t("config.occluder_model", model=value, allowed=", ".join(OCCLUDER_MODELS)))
        return value

    @field_validator("stride_motion")
    @classmethod
    def _stride_motion(cls, value: float) -> float:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(t("config.unit_interval"))
        return float(value)

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


class LandmarkConfig(_Model):
    """``auto`` runs 2DFAN4 when the file exists and replaces five points on profile or low-score faces."""

    refine: str = "auto"
    min_score: float = 0.65
    profile_ratio: float = 0.22

    @field_validator("refine")
    @classmethod
    def _refine(cls, value: str) -> str:
        allowed = ("auto", "always", "off")
        if value not in allowed:
            raise ValueError(t("config.landmark_refine", mode=value, allowed=", ".join(allowed)))
        return value

    @field_validator("min_score", "profile_ratio")
    @classmethod
    def _positive_ratio(cls, value: float) -> float:
        if float(value) < 0.0:
            raise ValueError(t("config.expression_range", field="landmarks"))
        return float(value)


class ExpressionConfig(_Model):
    """EAR/MAR-gated paste. Closed eyes take more of the target. An open mouth stays on the swap."""

    enabled: bool = True
    eye_base: float = 0.05
    eye_boost: float = 0.85
    mouth_base: float = 0.0
    mouth_boost: float = 0.55
    eye_dilate: int = 3
    mouth_dilate: int = 1
    eye_expand: float = 1.35
    mouth_expand: float = 1.15
    ear_open: float = 0.24
    ear_closed: float = 0.15
    mar_open: float = 0.045
    mar_closed: float = 0.012
    blur: int = 9

    @field_validator(
        "eye_base",
        "eye_boost",
        "mouth_base",
        "mouth_boost",
        "eye_expand",
        "mouth_expand",
        "ear_open",
        "ear_closed",
        "mar_open",
        "mar_closed",
    )
    @classmethod
    def _nonneg(cls, value: float, info: ValidationInfo) -> float:
        if float(value) < 0.0:
            raise ValueError(t("config.expression_range", field=info.field_name))
        return float(value)

    @field_validator("eye_dilate", "mouth_dilate", "blur")
    @classmethod
    def _nonneg_int(cls, value: int, info: ValidationInfo) -> int:
        if int(value) < 0:
            raise ValueError(t("config.expression_range", field=info.field_name))
        return int(value)


class SunglassesConfig(_Model):
    """Fuller identity swap, then an optional mild mouth. Eyes under the lenses stay off."""

    enabled: bool = True
    box_inset: float = 0.04
    expression: bool = True
    mouth_scale: float = 0.45

    @field_validator("box_inset")
    @classmethod
    def _inset(cls, value: float) -> float:
        if not 0.0 <= float(value) <= 0.45:
            raise ValueError(t("config.sunglasses_inset"))
        return float(value)

    @field_validator("mouth_scale")
    @classmethod
    def _mouth_scale(cls, value: float) -> float:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(t("config.unit_interval"))
        return float(value)


class VideoConfig(_Model):
    """Disk frame pipeline by default. ``mode: pipe`` keeps frames off disk."""

    encoder: str = "libx264"
    mode: str = "disk"
    threads: int = 4
    filter_every_n_frames: int = 30
    keep_audio: bool = True
    crf: int = 18
    max_face_gap: int = 10
    smooth: str = "one_euro"

    @field_validator("mode")
    @classmethod
    def _mode(cls, value: str) -> str:
        from barrett_faceswap.swap_catalog import VIDEO_MODES

        if value not in VIDEO_MODES:
            raise ValueError(t("config.video_mode", mode=value, allowed=", ".join(VIDEO_MODES)))
        return value

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

    @field_validator("max_face_gap")
    @classmethod
    def _gap(cls, value: int) -> int:
        if int(value) < 0:
            raise ValueError(t("config.max_face_gap"))
        return int(value)

    @field_validator("smooth")
    @classmethod
    def _smooth(cls, value: str) -> str:
        from barrett_faceswap.swap_catalog import SMOOTH_MODES

        if value not in SMOOTH_MODES:
            raise ValueError(t("config.smooth", mode=value, allowed=", ".join(SMOOTH_MODES)))
        return value


class ConsentConfig(_Model):
    required: bool = True


class WatermarkConfig(_Model):
    enabled: bool = True
    text: str | None = None
    # Optional TTF/OTF for labels the built-in glyphs do not cover.
    font_path: Path | None = None

    @field_validator("font_path", mode="before")
    @classmethod
    def _font(cls, value: object) -> Path | None:
        if value is None:
            return None
        text = str(value).strip()
        if text in {"", "none", "null"}:
            return None
        return resolve_config_path(text)


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


class LiveConfig(_Model):
    """Webcam loop. ``queue_size`` 1 keeps the newest frame and drops the rest."""

    queue_size: int = 1
    detect_every: int = 3
    preview: bool = True
    camera: int = 0

    @field_validator("queue_size", "detect_every")
    @classmethod
    def _positive(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError(t("config.live_positive"))
        return int(value)


class VoiceConfig(_Model):
    """Optional zero-shot voice conversion after a video face swap.

    ``enabled`` is the product default. The Gradio toggle still starts off and
    turns on only after the uploaded target video has an audio stream. A run
    converts speech only when that toggle (or ``--voice-reference``) is on, a
    reference clip is present, and the target has audio. A silent target stays
    face-only.
    """

    enabled: bool = True
    engine: str = "openvoice"
    checkpoint_dir: Path = Field(default_factory=lambda: resolve_config_path("models/openvoice_v2"))
    # 1 replaces the target timbre. 0 keeps the original track and skips the model.
    strength: float = 1.0
    # OpenVoice posterior noise. 0.3 is the upstream default. Higher is more random.
    tau: float = 0.3
    # auto uses CUDA when PyTorch can see a GPU, including an RTX 5090 on a cu128 build.
    device: str = "auto"
    # Upstream can embed a wavmark payload. Leave this off so compliance stays the only watermark.
    watermark: bool = False

    @field_validator("checkpoint_dir", mode="before")
    @classmethod
    def _checkpoint_dir(cls, value: object) -> Path:
        return resolve_config_path(value)

    @field_validator("engine")
    @classmethod
    def _engine(cls, value: str) -> str:
        if value != "openvoice":
            raise ValueError(t("config.voice_engine", value=value, allowed="openvoice"))
        return value

    @field_validator("device")
    @classmethod
    def _device(cls, value: str) -> str:
        if value not in {"auto", "cuda", "cpu"}:
            raise ValueError(t("config.voice_device", value=value))
        return value

    @field_validator("strength", "tau")
    @classmethod
    def _unit(cls, value: float) -> float:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(t("config.unit_interval"))
        return float(value)


class LibraryConfig(_Model):
    """On-disk source profiles. ``dir`` is relative to the working directory."""

    dir: Path = Field(default_factory=lambda: resolve_config_path("library"))
    voice_seconds: float = 24.0

    @field_validator("dir", mode="before")
    @classmethod
    def _dir(cls, value: object) -> Path:
        return resolve_config_path(value)

    @field_validator("voice_seconds")
    @classmethod
    def _voice_seconds(cls, value: float) -> float:
        if float(value) <= 0.0:
            raise ValueError(t("config.library_voice_seconds"))
        return float(value)


class JobsConfig(_Model):
    """Directory that stores job specs and in-progress frames."""

    root: Path = Field(default_factory=lambda: resolve_config_path("jobs"))

    @field_validator("root", mode="before")
    @classmethod
    def _root(cls, value: object) -> Path:
        return resolve_config_path(value)


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
    landmarks: LandmarkConfig = Field(default_factory=LandmarkConfig)
    expression: ExpressionConfig = Field(default_factory=ExpressionConfig)
    sunglasses: SunglassesConfig = Field(default_factory=SunglassesConfig)
    library: LibraryConfig = Field(default_factory=LibraryConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)
    jobs: JobsConfig = Field(default_factory=JobsConfig)
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
    swap = config.swap
    blend = config.blend
    model = getattr(request, "enhancer_model", None)
    enhancer_blend = getattr(request, "enhancer_blend", None)
    weight = getattr(request, "enhancer_weight", None)
    encoder = getattr(request, "video_encoder", None)
    threads = getattr(request, "video_threads", None)
    every = getattr(request, "filter_every_n_frames", None)
    keep_audio = getattr(request, "keep_audio", None)
    if model is not None:
        enhancer = enhancer.model_copy(update={"model": model})
    if enhancer_blend is not None:
        enhancer = enhancer.model_copy(update={"blend": enhancer_blend})
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
    video_mode = getattr(request, "video_mode", None)
    if video_mode:
        video = video.model_copy(update={"mode": video_mode})
    swap_model = getattr(request, "swapper_model", None)
    if swap_model:
        swap = swap.model_copy(update={"model": swap_model})
    pixel_boost = getattr(request, "pixel_boost", None)
    # 0 is the native size. A truthy check would ignore that and keep a larger config.
    if pixel_boost is not None:
        swap = swap.model_copy(update={"pixel_boost": int(pixel_boost)})
    if getattr(request, "face_map", None):
        swap = swap.model_copy(update={"selector": "map", "face_map": request.face_map})
    elif getattr(request, "many_faces", False):
        swap = swap.model_copy(update={"selector": "many"})
    elif getattr(request, "reference_path", None):
        swap = swap.model_copy(update={"selector": "reference", "reference": request.reference_path})
    elif getattr(request, "selector", None):
        swap = swap.model_copy(update={"selector": request.selector})
    blend_mode = getattr(request, "blend_mode", None)
    if blend_mode:
        blend = blend.model_copy(update={"mode": blend_mode})
    color = getattr(request, "color_transfer", None)
    if color:
        blend = blend.model_copy(update={"color": color})
    eye_mode = getattr(request, "eye_mode", None)
    if eye_mode:
        blend = blend.model_copy(update={"eye_mode": eye_mode})
    execution = config.execution
    if getattr(request, "commercial", False):
        execution = execution.model_copy(update={"commercial": True})
    masks = config.masks
    occluder_model = getattr(request, "occluder_model", None)
    if occluder_model:
        masks = masks.model_copy(update={"occluder_model": occluder_model})
    detect = config.detect
    # 0 is a legal threshold. A truthy check would keep the config value.
    detect_score = getattr(request, "detect_score", None)
    if detect_score is not None:
        try:
            detect = DetectConfig.model_validate(
                {**detect.model_dump(), "score_threshold": float(detect_score)}
            )
        except ValidationError as exc:
            raise ConfigError(t("config.unit_interval")) from exc
    return config.model_copy(
        update={
            "enhancer": enhancer,
            "video": video,
            "swap": swap,
            "blend": blend,
            "execution": execution,
            "masks": masks,
            "detect": detect,
        }
    )


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
