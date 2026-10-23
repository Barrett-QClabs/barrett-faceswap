"""SCRFD-style face detection on an ONNX graph.

The decoder follows the published SCRFD distance-to-box layout used by
InsightFace ``det_10g`` exports: three stride heads, a score, a 4-distance
box, and 10 keypoint distances. Anchor points sit on the cell origin. This
module is an interoperability layer for that tensor contract. It is not a
copy of the InsightFace training or Python package.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from barrett_faceswap.errors import DetectorError, NoFaceError
from barrett_faceswap.i18n import t
from barrett_faceswap.types import Face

logger = logging.getLogger(__name__)


class FaceDetector:
    """Run the configured detector and return faces in original-image pixels."""

    def __init__(self, session, det_size: int, score_threshold: float, nms_threshold: float) -> None:
        self.session = session
        self.det_size = int(det_size)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)

    def detect(self, image_bgr: np.ndarray) -> list[Face]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise DetectorError(t("detect.need_bgr"))
        height, width = image_bgr.shape[:2]
        det_w, det_h = _input_size(self.session, self.det_size)
        resized = cv2.resize(image_bgr, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})
        faces = decode_scrfd(
            outputs,
            det_size=det_w,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            origin_size=(width, height),
        )
        if not faces:
            raise NoFaceError(t("detect.no_face"))
        return faces


def select_face(faces: list[Face], index: int) -> Face:
    """Pick a face from the largest-first ordering."""

    ordered = sorted(faces, key=face_area, reverse=True)
    if index < 0 or index >= len(ordered):
        raise NoFaceError(t("detect.bad_index", index=index, last=len(ordered) - 1))
    return ordered[index]


def face_area(face: Face) -> float:
    x1, y1, x2, y2 = [float(value) for value in face.bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def decode_scrfd(
    outputs: list[np.ndarray],
    det_size: int,
    score_threshold: float,
    nms_threshold: float,
    origin_size: tuple[int, int] | None = None,
) -> list[Face]:
    """Decode SCRFD heads into faces.

    ``origin_size`` is ``(width, height)`` of the image the caller cares about.
    When it is set, boxes are scaled from the square network input and clipped.
    """

    heads = group_scrfd_outputs(outputs)
    boxes: list[list[float]] = []
    scores: list[float] = []
    points: list[np.ndarray] = []
    for score_map, distance_map, kps_map in heads:
        count = int(score_map.shape[0])
        num_anchors, stride = _stride_for(count, det_size)
        anchors = _anchor_grid(count, stride, num_anchors)
        values = score_map.reshape(-1).astype(np.float32)
        if values.size and (float(values.max()) > 1.0 or float(values.min()) < 0.0):
            values = 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))
        distances = distance_map.astype(np.float32) * float(stride)
        keypoints = kps_map.astype(np.float32) * float(stride)
        for index in range(count):
            score = float(values[index])
            if score < score_threshold:
                continue
            ax, ay = float(anchors[index, 0]), float(anchors[index, 1])
            left, top, right, bottom = [float(item) for item in distances[index]]
            box = [ax - left, ay - top, ax + right, ay + bottom]
            raw = keypoints[index]
            landmarks = np.stack(
                [
                    [ax + float(raw[0]), ay + float(raw[1])],
                    [ax + float(raw[2]), ay + float(raw[3])],
                    [ax + float(raw[4]), ay + float(raw[5])],
                    [ax + float(raw[6]), ay + float(raw[7])],
                    [ax + float(raw[8]), ay + float(raw[9])],
                ],
                axis=0,
            ).astype(np.float32)
            boxes.append(box)
            scores.append(score)
            points.append(landmarks)
    if not boxes:
        return []
    keep = nms(np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32), nms_threshold)
    faces: list[Face] = []
    scale_x = 1.0
    scale_y = 1.0
    limit: tuple[int, int] | None = None
    if origin_size is not None:
        origin_w, origin_h = origin_size
        scale_x = float(origin_w) / float(det_size)
        scale_y = float(origin_h) / float(det_size)
        limit = (origin_w, origin_h)
    for index in keep:
        box = np.asarray(boxes[index], dtype=np.float32)
        landmarks = points[index].copy()
        box[0::2] *= scale_x
        box[1::2] *= scale_y
        landmarks[:, 0] *= scale_x
        landmarks[:, 1] *= scale_y
        if limit is not None:
            box[0::2] = np.clip(box[0::2], 0, limit[0] - 1)
            box[1::2] = np.clip(box[1::2], 0, limit[1] - 1)
            landmarks[:, 0] = np.clip(landmarks[:, 0], 0, limit[0] - 1)
            landmarks[:, 1] = np.clip(landmarks[:, 1], 0, limit[1] - 1)
        faces.append(Face(bbox=box, score=scores[index], landmarks=landmarks))
    faces.sort(key=lambda face: face.score, reverse=True)
    return faces


def group_scrfd_outputs(
    outputs: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pair score, box, and keypoint tensors by matching their row counts."""

    scores: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    keypoints: list[np.ndarray] = []
    for output in outputs:
        columns = _as_columns(output)
        width = int(columns.shape[1])
        if width == 1:
            scores.append(columns)
        elif width == 4:
            boxes.append(columns)
        elif width == 10:
            keypoints.append(columns)
        else:
            logger.debug(t("detect.debug_ignore", shape=getattr(output, "shape", None)))
    if not scores or not (len(scores) == len(boxes) == len(keypoints)):
        raise DetectorError(
            t(
                "detect.not_scrfd",
                scores=len(scores),
                boxes=len(boxes),
                keypoints=len(keypoints),
            )
        )
    order = sorted(range(len(scores)), key=lambda item: scores[item].shape[0], reverse=True)
    grouped: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    box_by_count: dict[int, list[np.ndarray]] = {}
    kps_by_count: dict[int, list[np.ndarray]] = {}
    for tensor in boxes:
        box_by_count.setdefault(int(tensor.shape[0]), []).append(tensor)
    for tensor in keypoints:
        kps_by_count.setdefault(int(tensor.shape[0]), []).append(tensor)
    for index in order:
        score = scores[index]
        count = int(score.shape[0])
        try:
            box = box_by_count[count].pop(0)
            kps = kps_by_count[count].pop(0)
        except (KeyError, IndexError) as exc:
            raise DetectorError(t("detect.unpaired", count=count)) from exc
        grouped.append((score, box, kps))
    return grouped


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS. Returns indices into ``boxes``."""

    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while len(order):
        current = int(order[0])
        kept.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        overlap = _iou(boxes[current], boxes[rest])
        order = rest[overlap < iou_threshold]
    return kept


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_others = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(0.0, others[:, 3] - others[:, 1])
    union = area_box + area_others - intersection
    return intersection / np.maximum(union, 1e-6)


def _stride_for(count: int, det_size: int) -> tuple[int, int]:
    for num_anchors in (2, 1):
        if count % num_anchors != 0:
            continue
        cells = count // num_anchors
        side = int(round(cells**0.5))
        if side * side == cells and side > 0 and det_size % side == 0:
            return num_anchors, det_size // side
    raise DetectorError(t("detect.stride", count=count, det_size=det_size))


def _anchor_grid(count: int, stride: int, num_anchors: int) -> np.ndarray:
    cells = count // num_anchors
    side = int(round(cells**0.5))
    ys, xs = np.mgrid[0:side, 0:side]
    centers = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32) * float(stride)
    if num_anchors > 1:
        centers = np.repeat(centers, num_anchors, axis=0)
    if centers.shape[0] != count:
        raise DetectorError(t("detect.anchor_grid"))
    return centers


def _as_columns(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise DetectorError(t("detect.rank", shape=getattr(array, "shape", None)))
    return values


def _input_size(session, fallback: int) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape
    if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        return int(shape[3]), int(shape[2])
    return fallback, fallback
