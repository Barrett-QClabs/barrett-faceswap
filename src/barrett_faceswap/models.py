"""Locate ONNX files, hash-check them, and cache sessions.

A session is opened once per file identity (path, size, mtime), providers,
and device. Later calls return that session and do not hash the file again.
CUDA is preferred when it appears in the request; TensorRT is accepted but
not tuned. Weights are never downloaded here.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import ModelHashError, ModelNotFoundError
from barrett_faceswap.execution import resolve_providers
from barrett_faceswap.i18n import t
from barrett_faceswap.model_registry import ROLE_ORDER, registry_record

logger = logging.getLogger(__name__)

_SESSION_CACHE: dict[tuple, object] = {}
_CACHE_LOCK = threading.Lock()

# The swap path needs these three. Enhancer and mask roles are optional.
CORE_ROLES = ("detector", "recognizer", "swapper")


def clear_session_cache() -> None:
    """Drop cached sessions and hash results. Tests use this to isolate keys."""

    with _CACHE_LOCK:
        _SESSION_CACHE.clear()
        _HASH_CACHE.clear()


_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def file_sha256(path: Path) -> str:
    """SHA-256 of ``path``, reused while the size and mtime stay the same.

    ``session()`` used to hash the whole weight on every frame. A 500 MB
    inswapper file dominated the swap. The stat key makes a repeat lookup
    skip the read. A replaced file has a new mtime and is hashed again.
    """

    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    with _CACHE_LOCK:
        cached = _HASH_CACHE.get(key)
    if cached is not None:
        return cached
    value = _file_sha256_uncached(path)
    with _CACHE_LOCK:
        _HASH_CACHE[key] = value
    return value


def _file_sha256_uncached(path: Path) -> str:
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
    """Session cache for detector, swap, enhancer, and mask graphs."""

    def __init__(self, config: AppConfig, providers: list[str] | tuple[str, ...] | None = None) -> None:
        self.config = config
        self.providers = list(providers or config.execution.providers)
        self.device_id = int(config.execution.device_id)

    def paths(self) -> dict[str, Path]:
        return {role: weight_path(self.config, role) for role in ROLE_ORDER}

    def missing(self, roles: tuple[str, ...] | None = None) -> list[tuple[str, Path]]:
        selected = self.paths() if roles is None else {role: self.paths()[role] for role in roles}
        return [(role, path) for role, path in selected.items() if not path.is_file()]

    def ensure_present(self, roles: tuple[str, ...] | None = None) -> None:
        """Raise if a required weight is absent. Does not download anything.

        The default set is the swap trio. Pass a narrower tuple for the enhancer.
        """

        missing = self.missing(roles if roles is not None else CORE_ROLES)
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
            self.ensure_present((role,))
        record = registry_record(self.config, role)
        graph = bool(self.config.execution.cuda_graph)
        stat = path.stat()
        key = (
            str(path.resolve()),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            tuple(self.providers),
            self.device_id,
            graph,
            self.config.execution.precision,
        )
        with _CACHE_LOCK:
            cached = _SESSION_CACHE.get(key)
            if cached is not None:
                return cached
        verify_model_hash(path, record.sha256, role)
        with _CACHE_LOCK:
            cached = _SESSION_CACHE.get(key)
            if cached is not None:
                return cached
            ort_providers = resolve_providers(self.providers, self.device_id, self.config.execution)
            if factory is None:
                session = _onnxruntime().InferenceSession(str(path), providers=ort_providers)
            else:
                session = factory(path, ort_providers, self.device_id)
            _SESSION_CACHE[key] = session
            return session


def weight_path(config: AppConfig, role: str) -> Path:
    """fp32 path, or the sibling ``*_fp16.onnx`` when that file is present.

    A missing fp16 file falls back to fp32 so a CPU checkout still loads.
    """

    base = getattr(config.models, role).path
    if config.execution.precision != "fp16":
        return base
    candidate = base.with_name(f"{base.stem}_fp16{base.suffix}")
    if candidate.is_file():
        return candidate
    logger.info(t("execution.fp16_fallback", role=role, path=candidate))
    return base


def _onnxruntime():
    import onnxruntime

    return onnxruntime
