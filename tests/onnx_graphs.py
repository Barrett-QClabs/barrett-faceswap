"""Tiny ONNX graphs for tests. They are not face models."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_constant_onnx(path: Path, input_shape: list[int], output: np.ndarray) -> None:
    """Write a graph that ignores its input and returns ``output``."""

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    values = np.asarray(output, dtype=np.float32)
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(values.shape))
    node = helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(values))
    graph = helper.make_graph([node], "constant", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def write_swap_identity(path: Path, size: int) -> None:
    """Image identity with a required 512-d latent input. Not a face model."""

    import onnx
    from onnx import TensorProto, helper

    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, size, size])
    latent = helper.make_tensor_value_info("latent", TensorProto.FLOAT, [1, 512])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, size, size])
    node = helper.make_node("Identity", ["image"], ["output"])
    graph = helper.make_graph([node], "swap-identity", [image, latent], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def write_vector_identity(path: Path) -> None:
    """512-d identity. Stands in for a crossface converter."""

    import onnx
    from onnx import TensorProto, helper

    incoming = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 512])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 512])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "vector-identity", [incoming], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def write_identity_with_weight(path: Path, size: int) -> None:
    """Image identity plus a required scalar weight input."""

    import onnx
    from onnx import TensorProto, helper

    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, size, size])
    weight = helper.make_tensor_value_info("weight", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, size, size])
    node = helper.make_node("Identity", ["image"], ["output"])
    graph = helper.make_graph([node], "identity", [image, weight], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
