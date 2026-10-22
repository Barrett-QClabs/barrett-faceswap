"""Append-only local audit log.

Each event is one JSON object on its own line. The writer opens the file in
append mode and does not rewrite earlier lines. Enforcement-off runs do not
call this module.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix
    fcntl = None

from barrett_faceswap.errors import ComplianceError
from barrett_faceswap.i18n import t


def hash_file(path: Path) -> str | None:
    """SHA-256 of a file, or ``None`` when the path is missing."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_event(path: Path, event: dict) -> None:
    """Append one JSON line. Creates parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    line = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    try:
        with path.open("a", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise ComplianceError(t("audit.append_failed", path=path, error=exc)) from exc
