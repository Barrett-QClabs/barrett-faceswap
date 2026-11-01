"""Bound the lifetime of face mapper thumbnails and reference crops."""
import atexit
import tempfile
from pathlib import Path


class MapWorkspace:
    def __init__(self):
        self._temporary = None
        atexit.register(self.cleanup)

    def reset(self) -> Path:
        self.cleanup()
        self._temporary = tempfile.TemporaryDirectory(prefix="barrett-map-")
        return Path(self._temporary.name)

    def cleanup(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
