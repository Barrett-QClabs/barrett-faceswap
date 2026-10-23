"""Detect, align, swap, and paste one or more faces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from barrett_faceswap.align import align_face, swap_template
from barrett_faceswap.blend import apply_color, paste_back, paste_poisson
from barrett_faceswap.config import AppConfig
from barrett_faceswap.detect import FaceDetector, select_face
from barrett_faceswap.errors import NoFaceError
from barrett_faceswap.facesel import FaceMapEntry, FaceTracker, assign_slots, load_face_map
from barrett_faceswap.i18n import t
from barrett_faceswap.masking import MaskStack
from barrett_faceswap.models import ModelStore
from barrett_faceswap.pixel_boost import run_tiled
from barrett_faceswap.processors.registry import register_processor
from barrett_faceswap.swap import GraphSwapper, IdentitySwapper, embed_arcface
from barrett_faceswap.swap_catalog import output_size, swapper_by_name
from barrett_faceswap.types import Face, ProcessContext
from barrett_faceswap.vision import read_image


@dataclass
class _Slot:
    latent: np.ndarray
    match: np.ndarray
    source_score: float = 0.0
    target_index: int | None = None


class FaceSwapper:
    """Default processor. Dry-runs skip the model store entirely."""

    name = "face_swapper"

    def __init__(self, config: AppConfig, providers: list[str]) -> None:
        self.config = config
        self.providers = providers
        self.store: ModelStore | None = None
        self.masks: MaskStack | None = None
        self.tracker = FaceTracker(iou_threshold=float(config.swap.track_iou))
        self._slots: list[_Slot] | None = None
        self._spec = swapper_by_name(config.swap.model)

    def pre_check(self, ctx: ProcessContext) -> None:
        if ctx.dry_run:
            ctx.notes.append(t("processor.swapper_dry_run", model=self._spec.name))
            return
        self.store = ModelStore(self.config, self.providers)
        roles = ["detector", "recognizer", self._spec.role]
        if self._spec.crossface_role:
            roles.append(self._spec.crossface_role)
        self.store.ensure_present(tuple(roles))
        self.masks = MaskStack(self.config, self.store)
        self.masks.load()

    def process_frame(self, frame: np.ndarray, ctx: ProcessContext) -> np.ndarray:
        if ctx.dry_run:
            return frame
        if self.store is None:
            self.pre_check(ctx)
        assert self.store is not None
        faces = list(ctx.known_faces) if ctx.known_faces is not None else self._detect(frame)
        ordered = sorted(faces, key=lambda face: _area(face), reverse=True)
        slots = self._identities(ctx)
        pairs = self._pairs(frame, ordered, slots, ctx)
        if not pairs:
            raise NoFaceError(t("swap.no_match", threshold=self.config.swap.similarity))
        result = frame
        for face, slot in pairs:
            result = self._swap_one(result, face, slot, ctx)
        if ctx.log_every_frame:
            ctx.notes.append(
                t(
                    "processor.swapper_done",
                    source_score=f"{pairs[0][1].source_score:.3f}",
                    target_score=f"{pairs[0][0].score:.3f}",
                    index=ctx.target_face_index,
                    faces=len(pairs),
                    model=self._spec.name,
                )
            )
        return result

    def prepare_source(self, ctx: ProcessContext) -> None:
        """Embed the source once. Later frames reuse the cached slots."""

        if ctx.dry_run or self._slots is not None:
            return
        if self.store is None:
            self.pre_check(ctx)
        self._identities(ctx)

    def detect_faces(self, frame: np.ndarray):
        """Run the detector. The live loop calls this on a stride."""

        if self.store is None:
            return []
        return self._detect(frame)

    def post_process(self) -> None:
        return None

    def _detect(self, image: np.ndarray) -> list[Face]:
        assert self.store is not None
        detector = FaceDetector(
            self.store.session("detector"),
            det_size=self.config.detect.det_size,
            score_threshold=self.config.detect.score_threshold,
            nms_threshold=self.config.detect.nms_threshold,
        )
        return detector.detect(image)

    def _backend(self):
        assert self.store is not None
        if self._spec.embedding == "emap":
            return IdentitySwapper(self.store, self._spec.native_size)
        return GraphSwapper(self.store, self._spec)

    def _identities(self, ctx: ProcessContext) -> list[_Slot]:
        if self._slots is not None:
            return self._slots
        backend = self._backend()
        entries = self._map_entries(ctx)
        slots: list[_Slot] = []
        for entry in entries:
            image = ctx.source_bgr if not entry.source else read_image(Path(entry.source))
            arc, source_score = self._embed_image(backend, image)
            match = arc
            if entry.reference:
                match, _reference_score = self._embed_image(backend, read_image(Path(entry.reference)))
            latent = arc if self._spec.embedding == "emap" else backend.project(arc)
            slots.append(
                _Slot(
                    latent=latent,
                    match=match,
                    source_score=source_score,
                    target_index=entry.target_index,
                )
            )
        self._slots = slots
        return slots

    def _map_entries(self, ctx: ProcessContext) -> list[FaceMapEntry]:
        del ctx
        if self.config.swap.selector == "map" and self.config.swap.face_map is not None:
            return load_face_map(str(self.config.swap.face_map))
        reference = None
        if self.config.swap.selector == "reference" and self.config.swap.reference is not None:
            reference = str(self.config.swap.reference)
        return [FaceMapEntry(source="", target_index=None, reference=reference)]

    def _embed_image(self, backend, image: np.ndarray) -> tuple[np.ndarray, float]:
        face = select_face(self._detect(image), 0)
        if self._spec.embedding == "emap":
            vector = backend.embed(image, face.landmarks)
        else:
            assert self.store is not None
            vector = embed_arcface(self.store, image, face.landmarks)
        return vector, float(face.score)

    def _pairs(
        self,
        frame: np.ndarray,
        ordered: list[Face],
        slots: list[_Slot],
        ctx: ProcessContext,
    ) -> list[tuple[Face, _Slot]]:
        selector = self.config.swap.selector
        if selector == "index":
            return [(select_face(ordered, ctx.target_face_index), slots[0])]
        if selector == "many":
            return [(face, slots[0]) for face in ordered]
        tracks = None
        if self.config.swap.track:
            tracks = self.tracker.update([face.bbox for face in ordered])
        pairs: list[tuple[Face, _Slot]] = []
        pending: list[int] = []
        taken: set[int] = set()
        for index, face in enumerate(ordered):
            track = tracks[index] if tracks is not None else None
            if track is not None and track.slot is not None and track.slot < len(slots):
                pairs.append((face, slots[track.slot]))
                taken.add(index)
            else:
                pending.append(index)
        for slot_index, slot in enumerate(slots):
            if slot.target_index is None:
                continue
            if slot.target_index < 0 or slot.target_index >= len(ordered):
                raise NoFaceError(t("detect.bad_index", index=slot.target_index, last=len(ordered) - 1))
            face_index = slot.target_index
            if face_index in taken:
                continue
            pairs.append((ordered[face_index], slot))
            taken.add(face_index)
            if tracks is not None:
                tracks[face_index].slot = slot_index
        open_slots = [index for index, slot in enumerate(slots) if slot.target_index is None]
        open_faces = [index for index in pending if index not in taken]
        if open_slots and open_faces:
            vectors = [self._arc_of(frame, ordered[index]) for index in open_faces]
            slot_vectors = [slots[index].match for index in open_slots]
            chosen = assign_slots(
                vectors,
                slot_vectors,
                float(self.config.swap.similarity),
                one_to_one=True,
            )
            for local_index, slot_local in enumerate(chosen):
                if slot_local is None:
                    continue
                face_index = open_faces[local_index]
                slot_index = open_slots[slot_local]
                pairs.append((ordered[face_index], slots[slot_index]))
                if tracks is not None:
                    tracks[face_index].slot = slot_index
        return pairs

    def _arc_of(self, frame: np.ndarray, face: Face) -> np.ndarray:
        assert self.store is not None
        return embed_arcface(self.store, frame, face.landmarks)

    def _swap_one(self, frame: np.ndarray, face: Face, slot: _Slot, ctx: ProcessContext) -> np.ndarray:
        del ctx
        side = output_size(
            self._spec,
            pixel_boost=self.config.swap.pixel_boost,
            resolution=self.config.swap.resolution,
        )
        template = swap_template(self._spec.template, side)
        crop, matrix = align_face(frame, face.landmarks, side, template=template)
        backend = self._backend()

        def _infer(tile: np.ndarray) -> np.ndarray:
            return backend.forward(tile, slot.latent)

        swapped = run_tiled(crop, self._spec.native_size, _infer)
        if self.masks is None:
            assert self.store is not None
            self.masks = MaskStack(self.config, self.store)
            self.masks.load()
        mask = self.masks.compose(crop, side, self.config.blend.mask_blur, landmarks=template)
        colored = apply_color(self.config.blend.color, swapped, crop, mask)
        if self.config.blend.mode == "poisson":
            return paste_poisson(frame, colored, matrix, mask)
        return paste_back(frame, colored, matrix, mask)


def _area(face: Face) -> float:
    x1, y1, x2, y2 = [float(value) for value in face.bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@register_processor("face_swapper")
def _build(config: AppConfig, providers: list[str]) -> FaceSwapper:
    return FaceSwapper(config, providers)
