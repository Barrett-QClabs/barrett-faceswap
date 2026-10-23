"""Refuse weights that the registry does not mark as commercially usable.

The switch is opt-in. The default build does not set it, and every weight
shipped as a filename in this repository stays ``commercial_use: false``.
A later model you train and register yourself can flip that flag.
"""

from __future__ import annotations

from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import ConfigError
from barrett_faceswap.i18n import t
from barrett_faceswap.model_registry import BUILTIN
from barrett_faceswap.swap_catalog import swapper_by_name


def roles_for_config(config: AppConfig, processors: list[str] | None = None) -> list[str]:
    """Roles a swap would open. Enhancer roles are included only when selected."""

    spec = swapper_by_name(config.swap.model)
    roles = ["detector", "recognizer", spec.role]
    if spec.crossface_role:
        roles.append(spec.crossface_role)
    names = list(processors if processors is not None else config.processors)
    if "face_enhancer" in names:
        roles.append("codeformer" if config.enhancer.model == "codeformer" else "gfpgan")
    if config.masks.occlusion:
        roles.append("occluder")
    if config.masks.parser:
        roles.append("parser")
    if config.landmarks.refine != "off":
        roles.append("landmark")
    return roles


def refused_roles(roles: list[str]) -> list[str]:
    """Roles whose registry row is missing or not marked commercially usable."""

    blocked = []
    for role in roles:
        record = BUILTIN.get(role)
        if record is None or not record.commercial_use:
            blocked.append(role)
    return blocked


def guard_commercial(config: AppConfig, processors: list[str] | None = None) -> None:
    """Raise when a commercial build would load a non-commercial weight."""

    if not config.execution.commercial:
        return
    blocked = refused_roles(roles_for_config(config, processors))
    if blocked:
        raise ConfigError(t("commercial.refused", roles=", ".join(blocked)))
