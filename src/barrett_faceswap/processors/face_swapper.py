"""Detect, align, swap, and paste one or more faces."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from barrett_faceswap.align import align_face, swap_template
from barrett_faceswap.blend import apply_color, paste_back, paste_poisson
from barrett_faceswap.config import AppConfig
from barrett_faceswap.detect import FaceDetector, select_face
from barrett_faceswap.errors import NoFaceError
from barrett_faceswap.expression import preserve_expression
from barrett_faceswap.eyes import eye_openness, restore_eyes
from barrett_faceswap.landmarks import LandmarkRefiner
from barrett_faceswap.facesel import FaceMapEntry, FaceTracker, assign_slots, load_face_map
from barrett_faceswap.i18n import t
from barrett_faceswap.masking import MaskStack
from barrett_faceswap.models import ModelStore
from barrett_faceswap.pixel_boost import run_tiled
from barrett_faceswap.processors.registry import register_processor
from barrett_faceswap.recover import map_points, search_faces
from barrett_faceswap.preprocess import normalize_crop
from barrett_faceswap.sunglasses import apply_sunglass_mask, sunglass_plan
from barrett_faceswap.swap import GraphSwapper, IdentitySwapper, embed_arcface
from barrett_faceswap.swap_catalog import output_size, swapper_by_name
from barrett_faceswap.types import Face, ProcessContext

logger = logging.getLogger(__name__)
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
        self._detector_obj: FaceDetector | None = None
        self._backend_impl = None
        self._landmarker: LandmarkRefiner | None = None
        self._sunglass_noted = False
        self.stage_seconds = {
            "detect": 0.0,
            "align": 0.0,
            "infer": 0.0,
            "mask": 0.0,
            "color": 0.0,
            "paste": 0.0,
        }

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
        if ctx.known_faces is not None:
            faces = list(ctx.known_faces)
        else:
            faces = self._detect(frame)
        if not faces:
            if ctx.allow_missing_face:
                return frame
            raise NoFaceError(t("detect.no_face"))
        ordered = sorted(faces, key=lambda face: _area(face), reverse=True)
        slots = self._identities(ctx)
        pairs = self._pairs(frame, ordered, slots, ctx)
        if not pairs:
            if ctx.allow_missing_face:
                return frame
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
        if self.masks is not None:
            self.masks.reset()

    def find_faces(self, image: np.ndarray) -> tuple[list[Face], str]:
        """Video entry. Retries a miss, and does not raise when the frame is empty."""

        if self.store is None:
            return [], "miss"
        started = time.perf_counter()
        try:
            return search_faces(self._detector(), image)
        finally:
            self.stage_seconds["detect"] += time.perf_counter() - started

    def _refine(self, frame: np.ndarray, face: Face) -> Face:
        if self.config.landmarks.refine == "off" or self.store is None:
            return face
        if self._landmarker is None:
            self._landmarker = LandmarkRefiner(self.store, self.config)
        return self._landmarker.refine(frame, face)

    def _detector(self) -> FaceDetector:
        if self._detector_obj is None:
            assert self.store is not None
            self._detector_obj = FaceDetector(
                self.store.session("detector"),
                det_size=self.config.detect.det_size,
                score_threshold=self.config.detect.score_threshold,
                nms_threshold=self.config.detect.nms_threshold,
            )
        return self._detector_obj

    def _detect(self, image: np.ndarray) -> list[Face]:
        started = time.perf_counter()
        try:
            return self._detector().detect(image)
        finally:
            self.stage_seconds["detect"] += time.perf_counter() - started

    def _backend(self):
        if self._backend_impl is None:
            assert self.store is not None
            if self._spec.embedding == "emap":
                self._backend_impl = IdentitySwapper(self.store, self._spec.native_size)
            else:
                self._backend_impl = GraphSwapper(self.store, self._spec)
        return self._backend_impl

    def _identities(self, ctx: ProcessContext) -> list[_Slot]:
        if self._slots is not None:
            return self._slots
        backend = self._backend()
        entries = self._map_entries(ctx)
        slots: list[_Slot] = []
        for entry in entries:
            use_saved = entry.embedding and self._spec.embedding != "source_image"
            if use_saved:
                arc = _unit_embedding(np.asarray(entry.embedding, dtype=np.float32))
                source_score = 1.0
            else:
                image = ctx.source_bgr if not entry.source else read_image(Path(entry.source))
                hint = ctx.source_face if not entry.source else None
                prepared = ctx.source_embedding if not entry.source else None
                if prepared is not None:
                    arc = _unit_embedding(prepared)
                    source_score = 1.0
                else:
                    arc, source_score = self._embed_image(backend, image, hint)
            if entry.match:
                match = _unit_embedding(np.asarray(entry.match, dtype=np.float32))
            elif entry.reference:
                match, _reference_score = self._embed_image(backend, read_image(Path(entry.reference)))
            else:
                match = arc
            if self._spec.embedding in {"emap", "source_image"}:
                latent = arc
            else:
                latent = backend.project(arc)
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

    def _embed_image(self, backend, image: np.ndarray, face: Face | None = None) -> tuple[np.ndarray, float]:
        if face is None:
            face = select_face(self._detect(image), 0)
        if self._spec.embedding == "source_image":
            side = int(self._spec.source_size)
            template = swap_template("arcface", side)
            crop, _matrix = align_face(image, face.landmarks[:5], side, template=template)
            return normalize_crop(crop, self._spec), float(face.score)
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
        side = output_size(
            self._spec,
            pixel_boost=self.config.swap.pixel_boost,
            resolution=self.config.swap.resolution,
        )
        template = swap_template(self._spec.template, side)
        started = time.perf_counter()
        face = self._refine(frame, face)
        points = np.asarray(face.landmarks, dtype=np.float32).reshape(-1, 2)
        crop, matrix = align_face(frame, points[:5], side, template=template)
        self.stage_seconds["align"] += time.perf_counter() - started
        backend = self._backend()

        def _infer(tile: np.ndarray) -> np.ndarray:
            return backend.forward(tile, slot.latent)

        started = time.perf_counter()
        swapped = run_tiled(crop, self._spec.native_size, _infer)
        self.stage_seconds["infer"] += time.perf_counter() - started
        if self.masks is None:
            assert self.store is not None
            self.masks = MaskStack(self.config, self.store)
            self.masks.load()
        started = time.perf_counter()
        mask = self.masks.compose(
            crop,
            side,
            self.config.blend.mask_blur,
            landmarks=template,
            frame_points=points[:5],
            matrix=matrix,
            sequential=ctx.reuse_masks,
            face_width=max(float(face.bbox[2]) - float(face.bbox[0]), 1.0),
        )
        self.stage_seconds["mask"] += time.perf_counter() - started
        plan = sunglass_plan(
            self.config.sunglasses,
            crop,
            template,
            labels=getattr(self.masks, "labels", None),
        )
        if plan.active:
            mask = apply_sunglass_mask(mask, side, self.config.blend.mask_blur, template, plan.box_inset)
            self._note_sunglasses(ctx)
        started = time.perf_counter()
        colored = apply_color(self.config.blend.color, swapped, crop, mask)
        expressed = None
        use_expression = self.config.expression.enabled and face.landmarks_68 is not None
        if plan.active and not plan.expression:
            use_expression = False
        if use_expression:
            expressed = preserve_expression(
                colored,
                crop,
                face.landmarks_68,
                matrix,
                self.config.expression,
                include_eyes=not plan.active,
                mouth_scale=plan.mouth_scale if plan.active else 1.0,
            )
        if plan.active:
            if expressed is not None:
                colored, _ignored = expressed
            ctx.eye_openness = 1.0
        elif expressed is None:
            eye_points = map_points(points, matrix) if points.shape[0] >= 106 else template
            openness = eye_openness(eye_points)
            ctx.eye_openness = openness
            colored = restore_eyes(
                self.config.blend.eye_mode,
                colored,
                crop,
                eye_points,
                self.config.blend.mask_blur,
                self.config.blend.eye_sharpen,
                mask=getattr(self.masks, "eye_mask", None),
                openness=openness,
            )
        else:
            colored, openness = expressed
            ctx.eye_openness = openness
        self.stage_seconds["color"] += time.perf_counter() - started
        started = time.perf_counter()
        if self.config.blend.mode == "poisson":
            pasted = paste_poisson(frame, colored, matrix, mask)
        else:
            pasted = paste_back(frame, colored, matrix, mask)
        self.stage_seconds["paste"] += time.perf_counter() - started
        return pasted

    def _note_sunglasses(self, ctx: ProcessContext) -> None:
        if self._sunglass_noted:
            return
        self._sunglass_noted = True
        message = t("sunglasses.path")
        if ctx.log_every_frame:
            ctx.notes.append(message)
        else:
            logger.info(message)


def _unit_embedding(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise NoFaceError(t("identity.no_face"))
    return array / norm


def _area(face: Face) -> float:
    x1, y1, x2, y2 = [float(value) for value in face.bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@register_processor("face_swapper")
def _build(config: AppConfig, providers: list[str]) -> FaceSwapper:
    return FaceSwapper(config, providers)
