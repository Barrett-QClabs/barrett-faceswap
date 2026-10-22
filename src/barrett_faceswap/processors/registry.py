"""Name-to-factory map for frame processors."""

from __future__ import annotations

from collections.abc import Callable

from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import ProcessorError
from barrett_faceswap.i18n import t
from barrett_faceswap.processors.base import Processor

ProcessorFactory = Callable[[AppConfig, list[str]], Processor]

_REGISTRY: dict[str, ProcessorFactory] = {}


def register_processor(name: str) -> Callable[[ProcessorFactory], ProcessorFactory]:
    """Register a factory. The factory receives the app config and provider list."""

    def decorator(factory: ProcessorFactory) -> ProcessorFactory:
        _REGISTRY[name] = factory
        return factory

    return decorator


def create_processor(name: str, config: AppConfig, providers: list[str]) -> Processor:
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ProcessorError(t("processor.unknown", name=name, known=known)) from exc
    return factory(config, providers)


def available_processors() -> list[str]:
    return sorted(_REGISTRY)
