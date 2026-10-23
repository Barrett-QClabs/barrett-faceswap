"""Compose box, parsing, and occlusion masks for one aligned crop.

Model files are optional. A missing occlusion graph keeps the all-ones stub.
A missing parser uses a feathered landmark hull when the caller passes crop
landmarks, and the ellipse otherwise. Both still go through :func:`merge_min`.
"""

from __future__ import annotations

from barrett_faceswap.config import AppConfig
from barrett_faceswap.face_parsing import run_parser
from barrett_faceswap.masks import composite_mask, occlusion_mask, parser_stub_mask
from barrett_faceswap.models import ModelStore
from barrett_faceswap.occlusion import run_occlusion


class MaskStack:
    """Lazy ONNX masks for the swap paste. ``load`` runs on the calling thread."""

    def __init__(self, config: AppConfig, store: ModelStore | None) -> None:
        self.config = config
        self.store = store
        self._occlusion = None
        self._parser = None
        self._ready = False

    def load(self) -> None:
        """Open mask sessions when the files exist. Missing files stay on the stub."""

        if self._ready:
            return
        self._ready = True
        if self.store is None:
            return
        settings = self.config.masks
        if settings.occlusion and self.store.paths()["occluder"].is_file():
            self._occlusion = self.store.session("occluder")
        if settings.parser and self.store.paths()["parser"].is_file():
            self._parser = self.store.session("parser")

    def compose(self, crop_bgr, size: int, blur: int, landmarks=None):
        """Box, region, and occlusion, merged by the per-pixel minimum.

        ``landmarks`` are the five template points in the aligned crop. They
        are used only when the parsing file is absent.
        """

        self.load()
        settings = self.config.masks
        if self._occlusion is not None:
            occlusion = run_occlusion(
                self._occlusion,
                crop_bgr,
                size,
                blur,
                invert=settings.occlusion_invert,
            )
        else:
            path = None if self.store is None else str(self.store.paths()["occluder"])
            occlusion = occlusion_mask(size, path)
        if self._parser is not None:
            region = run_parser(self._parser, crop_bgr, list(settings.regions), size, blur)
        else:
            path = None if self.store is None else str(self.store.paths()["parser"])
            region = parser_stub_mask(size, blur, path, landmarks=landmarks)
        return composite_mask(size, blur, occlusion=occlusion, region=region)
