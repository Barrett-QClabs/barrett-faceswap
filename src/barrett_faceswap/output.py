"""Exclusive output allocation and private sibling work files."""
import os
import tempfile
from pathlib import Path


def reserve_output(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        candidate = path if index == 0 else path.with_name(f"{path.stem}-{index}{path.suffix}")
        sidecar = candidate.with_suffix(candidate.suffix + ".meta.json")
        if sidecar.exists() or sidecar.is_symlink():
            index += 1
            continue
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            index += 1
            continue
        os.close(fd)
        return candidate


def temporary_output(path: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix=".barrett-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    return Path(name)
