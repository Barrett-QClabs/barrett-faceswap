"""Locate ONNX files, hash-check them, and cache sessions.

The cache key is ``(path, sha256, requested providers, device id)``. CUDA is
preferred when it appears in the request; TensorRT is accepted but not tuned.
Weights are never downloaded here.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import ModelHashError, ModelNotFoundError
from barrett_faceswap.execution import resolve_providers
from barrett_faceswap.i18n import t
from barrett_faceswap.model_registry import registry_record

logger = logging.getLogger(__name__)

_SESSION_CACHE: dict[tuple, object] = {}


def clear_session_cache() -> None:
    """Drop cached sessions. Tests use this to isolate provider keys."""

    _SESSION_CACHE.clear()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_hash(path: Path, expected: str | None, role: str) -> str:
    """Hash ``path``. When ``expected`` is set, refuse a mismatch."""

    actual = file_sha256(path)
    if expected:
        if actual != expected.lower():
            raise ModelHashError(
                t(
                    "models.hash_mismatch",
                    role=role,
                    path=path,
                    expected=expected.lower(),
                    actual=actual,
                )
            )
        logger.info(t("models.hash_ok", role=role))
    else:
        logger.info(t("models.hash_unpinned", role=role, path=path, sha256=actual))
    return actual


class ModelStore:
    """Session cache for the detector, recognizer, and swapper."""

    def __init__(self, config: AppConfig, providers: list[str] | tuple[str, ...] | None = None) -> None:
        self.config = config
        self.providers = list(providers or config.execution.providers)
        self.device_id = int(config.execution.device_id)

    def paths(self) -> dict[str, Path]:
        return {
            "detector": self.config.models.detector.path,
            "recognizer": self.config.models.recognizer.path,
            "swapper": self.config.models.swapper.path,
        }

    def missing(self) -> list[tuple[str, Path]]:
        return [(role, path) for role, path in self.paths().items() if not path.is_file()]

    def ensure_present(self) -> None:
        """Raise if any required weight is absent. Does not download anything."""

        missing = self.missing()
        if not missing:
            return
        listing = "\n".join(f"  - {role}: {path}" for role, path in missing)
        raise ModelNotFoundError(t("models.missing", listing=listing))

    def session(self, role: str, *, factory=None):
        """Open or reuse a session. ``factory`` is a test hook."""

        if role not in self.paths():
            raise ModelNotFoundError(t("models.unknown_role", role=role))
        path = self.paths()[role]
        if not path.is_file():
            self.ensure_present()
        record = registry_record(self.config, role)
        digest = verify_model_hash(path, record.sha256, role)
        key = (str(path.resolve()), digest, tuple(self.providers), self.device_id)
        cached = _SESSION_CACHE.get(key)
        if cached is not None:
            return cached
        ort_providers = resolve_providers(self.providers, self.device_id)
        if factory is None:
            session = _onnxruntime().InferenceSession(str(path), providers=ort_providers)
        else:
            session = factory(path, ort_providers, self.device_id)
        _SESSION_CACHE[key] = session
        return session


def _onnxruntime():
    import onnxruntime

    return onnxruntime
