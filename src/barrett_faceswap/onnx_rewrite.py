"""Make ConvTranspose padding something the CUDA execution provider can run.

ONNX Runtime drops a ConvTranspose node to the CPU when its padding is
asymmetric (``pads`` begin and end differ on an axis). xseg does this, and
each fallback copies the tensor between GPU and CPU. The rewrite sets those
pads to zero, which is symmetric, and slices the extra border back off. The
values match the original node. Symmetric padding is left alone.
"""

from __future__ import annotations

import logging

import numpy as np

from barrett_faceswap.i18n import t

logger = logging.getLogger(__name__)

_REWRITE_ROLES = frozenset({"occluder", "xseg_2", "xseg_3", "parser"})


def should_rewrite(role: str) -> bool:
    return role in _REWRITE_ROLES


def _skip(role: str, exc: Exception) -> None:
    detail = str(exc).replace("{", "(").replace("}", ")")
    logger.info(t("models.convtranspose_skip", role=role, error=detail))


def convtranspose_cuda_bytes(payload: bytes, *, role: str = "model") -> tuple[bytes, int]:
    """Return model bytes and how many ConvTranspose nodes were rewritten.

    A graph with no asymmetric padding is returned unchanged. A parser failure
    also returns the original bytes so a mask file still loads.
    """

    try:
        import onnx
        from onnx import helper, numpy_helper
    except Exception as exc:
        _skip(role, exc)
        return payload, 0
    try:
        model = onnx.load_from_string(payload)
    except Exception as exc:
        _skip(role, exc)
        return payload, 0
    count = _rewrite_model(model, helper, numpy_helper)
    if count == 0:
        return payload, 0
    return model.SerializeToString(), count


def _rewrite_model(model, helper, numpy_helper) -> int:
    count = 0
    rebuilt: list = []
    for node in model.graph.node:
        rebuilt.append(node)
        if node.op_type != "ConvTranspose" or len(node.output) != 1:
            continue
        pads = _attr_ints(node, "pads")
        if pads is None or len(pads) != 4:
            continue
        height_begin, width_begin, height_end, width_end = (int(value) for value in pads)
        if height_begin == height_end and width_begin == width_end:
            continue
        _set_ints(node, "pads", [0, 0, 0, 0])
        _set_string(node, "auto_pad", "NOTSET")
        original = node.output[0]
        widened = f"{original}__cuda_pad"
        node.output[0] = widened
        stem = node.name or original
        starts = f"{stem}__slice_starts"
        ends = f"{stem}__slice_ends"
        axes = f"{stem}__slice_axes"
        end_h = np.iinfo(np.int64).max if height_end == 0 else -height_end
        end_w = np.iinfo(np.int64).max if width_end == 0 else -width_end
        _initializer(model, numpy_helper, starts, np.asarray([height_begin, width_begin], dtype=np.int64))
        _initializer(model, numpy_helper, ends, np.asarray([end_h, end_w], dtype=np.int64))
        _initializer(model, numpy_helper, axes, np.asarray([2, 3], dtype=np.int64))
        rebuilt.append(
            helper.make_node(
                "Slice",
                [widened, starts, ends, axes],
                [original],
                name=f"{stem}__slice_pad",
            )
        )
        count += 1
    if count:
        del model.graph.node[:]
        model.graph.node.extend(rebuilt)
    return count


def _attr_ints(node, name: str) -> list[int] | None:
    for attr in node.attribute:
        if attr.name == name and attr.ints:
            return list(attr.ints)
    return None


def _set_ints(node, name: str, values: list[int]) -> None:
    for attr in node.attribute:
        if attr.name == name:
            del attr.ints[:]
            attr.ints.extend(values)
            return
    from onnx import helper

    node.attribute.append(helper.make_attribute(name, values))


def _set_string(node, name: str, value: str) -> None:
    for attr in node.attribute:
        if attr.name == name:
            attr.s = value.encode()
            return
    from onnx import helper

    node.attribute.append(helper.make_attribute(name, value))


def _initializer(model, numpy_helper, name: str, values: np.ndarray) -> None:
    tensor = numpy_helper.from_array(np.asarray(values), name)
    model.graph.initializer.append(tensor)
