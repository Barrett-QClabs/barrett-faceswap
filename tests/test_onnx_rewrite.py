"""ConvTranspose padding rewrite, without a GPU."""

from __future__ import annotations

import numpy as np

from barrett_faceswap.onnx_rewrite import convtranspose_cuda_bytes


def _convtranspose_bytes() -> bytes:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 7, 7])
    node = helper.make_node(
        "ConvTranspose",
        ["input", "W"],
        ["output"],
        name="up",
        pads=[0, 0, 1, 1],
        strides=[2, 2],
    )
    graph = helper.make_graph(
        [node],
        "up",
        [value],
        [out],
        initializer=[numpy_helper.from_array(weight, "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def test_asymmetric_convtranspose_matches_after_the_slice() -> None:
    import onnx
    import onnxruntime

    original = _convtranspose_bytes()
    rewritten, count = convtranspose_cuda_bytes(original)
    assert count == 1
    model = onnx.load_from_string(rewritten)
    conv = next(node for node in model.graph.node if node.op_type == "ConvTranspose")
    pads = next(attr.ints for attr in conv.attribute if attr.name == "pads")
    assert list(pads) == [0, 0, 0, 0]
    assert any(node.op_type == "Slice" for node in model.graph.node)
    image = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    before = onnxruntime.InferenceSession(original, providers=["CPUExecutionProvider"])
    after = onnxruntime.InferenceSession(rewritten, providers=["CPUExecutionProvider"])
    left = before.run(None, {"input": image})[0]
    right = after.run(None, {"input": image})[0]
    assert left.shape == right.shape
    assert np.allclose(left, right, atol=1e-5)
    _again, second = convtranspose_cuda_bytes(rewritten)
    assert second == 0
