"""Declarative swapper backends.

Each entry names the ONNX contract this repository implements: template,
normalization, embedding projection, native size, and the pixel-boost sizes
that tile cleanly. Paths and hash pins stay in the config. Nothing here
downloads a file or vendors another project's source.
"""

from __future__ import annotations

from dataclasses import dataclass

from barrett_faceswap.errors import ConfigError
from barrett_faceswap.i18n import t


@dataclass(frozen=True)
class SwapperSpec:
    """One selectable swapper graph."""

    name: str
    role: str
    filename: str
    native_size: int
    template: str
    normalize: str
    embedding: str
    crossface_role: str | None
    boost_sizes: tuple[int, ...]
    license_key: str
    commercial_use: bool
    download_url: str
    alignment: str


def _boosts(native: int) -> tuple[int, ...]:
    return tuple(size for size in (128, 256, 512, 768, 1024) if size >= native and size % native == 0)


def _spec(
    name: str,
    *,
    role: str,
    filename: str,
    native_size: int,
    template: str,
    normalize: str,
    embedding: str,
    crossface_role: str | None,
    license_key: str,
    download_url: str,
    alignment: str,
) -> SwapperSpec:
    return SwapperSpec(
        name=name,
        role=role,
        filename=filename,
        native_size=native_size,
        template=template,
        normalize=normalize,
        embedding=embedding,
        crossface_role=crossface_role,
        boost_sizes=_boosts(native_size),
        license_key=license_key,
        commercial_use=False,
        download_url=download_url,
        alignment=alignment,
    )


_INSWAPPER = "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx"
_HYPER = "https://github.com/facefusion/facefusion-assets"
_SIM = "https://github.com/neuralchen/SimSwap"
_GHOST = "https://github.com/ai-forever/ghost"

SWAPPERS: dict[str, SwapperSpec] = {
    "inswapper_128": _spec(
        "inswapper_128",
        role="swapper",
        filename="inswapper_128.onnx",
        native_size=128,
        template="inswapper",
        normalize="zero_one",
        embedding="emap",
        crossface_role=None,
        license_key="models.license.insightface",
        download_url=_INSWAPPER,
        alignment="inswapper_128",
    ),
    "hyperswap_1a_256": _spec(
        "hyperswap_1a_256",
        role="hyperswap_1a",
        filename="hyperswap_1a_256.onnx",
        native_size=256,
        template="arcface",
        normalize="minus_one_one",
        embedding="crossface",
        crossface_role="crossface_hyperswap",
        license_key="models.license.hyperswap",
        download_url=_HYPER,
        alignment="arcface_256",
    ),
    "hyperswap_1b_256": _spec(
        "hyperswap_1b_256",
        role="hyperswap_1b",
        filename="hyperswap_1b_256.onnx",
        native_size=256,
        template="arcface",
        normalize="minus_one_one",
        embedding="crossface",
        crossface_role="crossface_hyperswap",
        license_key="models.license.hyperswap",
        download_url=_HYPER,
        alignment="arcface_256",
    ),
    "hyperswap_1c_256": _spec(
        "hyperswap_1c_256",
        role="hyperswap_1c",
        filename="hyperswap_1c_256.onnx",
        native_size=256,
        template="arcface",
        normalize="minus_one_one",
        embedding="crossface",
        crossface_role="crossface_hyperswap",
        license_key="models.license.hyperswap",
        download_url=_HYPER,
        alignment="arcface_256",
    ),
    "simswap_256": _spec(
        "simswap_256",
        role="simswap_256",
        filename="simswap_256.onnx",
        native_size=256,
        template="arcface",
        normalize="minus_one_one",
        embedding="crossface",
        crossface_role="crossface_simswap",
        license_key="models.license.simswap",
        download_url=_SIM,
        alignment="arcface_256",
    ),
    "simswap_unofficial_512": _spec(
        "simswap_unofficial_512",
        role="simswap_512",
        filename="simswap_unofficial_512.onnx",
        native_size=512,
        template="arcface",
        normalize="minus_one_one",
        embedding="crossface",
        crossface_role="crossface_simswap",
        license_key="models.license.simswap",
        download_url=_SIM,
        alignment="arcface_512",
    ),
    "ghost_1_256": _spec(
        "ghost_1_256",
        role="ghost_1",
        filename="ghost_1_256.onnx",
        native_size=256,
        template="arcface",
        normalize="minus_one_one",
        embedding="crossface",
        crossface_role="crossface_ghost",
        license_key="models.license.ghost",
        download_url=_GHOST,
        alignment="arcface_256",
    ),
    "ghost_2_256": _spec(
        "ghost_2_256",
        role="ghost_2",
        filename="ghost_2_256.onnx",
        native_size=256,
        template="arcface",
        normalize="minus_one_one",
        embedding="crossface",
        crossface_role="crossface_ghost",
        license_key="models.license.ghost",
        download_url=_GHOST,
        alignment="arcface_256",
    ),
    "ghost_3_256": _spec(
        "ghost_3_256",
        role="ghost_3",
        filename="ghost_3_256.onnx",
        native_size=256,
        template="arcface",
        normalize="minus_one_one",
        embedding="crossface",
        crossface_role="crossface_ghost",
        license_key="models.license.ghost",
        download_url=_GHOST,
        alignment="arcface_256",
    ),
}

SWAPPER_NAMES: tuple[str, ...] = tuple(SWAPPERS)

# Small ONNX graphs that map a 512-d ArcFace vector into a swapper's space.
CROSSFACE_FILES: dict[str, str] = {
    "crossface_hyperswap": "crossface_hyperswap.onnx",
    "crossface_simswap": "crossface_simswap.onnx",
    "crossface_ghost": "crossface_ghost.onnx",
}

SELECTORS = ("index", "many", "reference", "map")
BLEND_MODES = ("alpha", "poisson")
COLOR_TRANSFERS = ("none", "mean_std", "lab", "histogram")
VIDEO_MODES = ("disk", "pipe")
EYE_MODES = ("swap", "blend", "target")
SMOOTH_MODES = ("one_euro", "ema", "none")


def swapper_by_name(name: str) -> SwapperSpec:
    """Return a known swapper or raise a config error."""

    try:
        return SWAPPERS[name]
    except KeyError as exc:
        raise ConfigError(
            t("config.swapper_model", model=name, allowed=", ".join(SWAPPER_NAMES))
        ) from exc


def output_size(spec: SwapperSpec, *, pixel_boost: int, resolution: int) -> int:
    """Crop side length. ``pixel_boost`` wins. Inswapper still honors ``resolution``."""

    if int(pixel_boost) > 0:
        if int(pixel_boost) not in spec.boost_sizes:
            allowed = ", ".join(str(size) for size in spec.boost_sizes)
            raise ConfigError(t("swap.boost", size=pixel_boost, model=spec.name, allowed=allowed))
        return int(pixel_boost)
    if spec.embedding == "emap":
        size = int(resolution)
        if size <= 0:
            raise ConfigError(t("swap.boost", size=size, model=spec.name, allowed=str(spec.native_size)))
        return size
    return spec.native_size
