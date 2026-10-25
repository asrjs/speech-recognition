"""Focused checks for large-model external-data export behavior."""

import gc
import tempfile
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_whisper import (
    _DEFAULT_EXTERNAL_DATA_THRESHOLD,
    _resolve_external_data_policy,
)


def test_external_data_policy_keeps_explicit_modes_and_externalizes_large_auto():
    assert _resolve_external_data_policy("auto", 123, large_model=False) == (False, 123)
    assert _resolve_external_data_policy("auto", 123, large_model=True) == (True, 0)
    assert _resolve_external_data_policy(
        "auto", _DEFAULT_EXTERNAL_DATA_THRESHOLD, large_model=True
    ) == (True, 0)
    assert _resolve_external_data_policy("always", 123, large_model=False) == (True, 0)
    assert _resolve_external_data_policy("never", 123, large_model=True) == (False, 123)


def test_external_initializer_storage_matches_inline_cpu_runtime():
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    weight = numpy_helper.from_array(np.array([1.25, -2.0], dtype=np.float32), "weight")
    constant_value = numpy_helper.from_array(
        np.array([0.5, 3.0], dtype=np.float32), "constant_value"
    )
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "weight"], ["sum"]),
            helper.make_node("Constant", [], ["constant"], value=constant_value),
            helper.make_node("Add", ["sum", "constant"], ["y"]),
        ],
        "external-data-storage-test",
        [x_info],
        [y_info],
        initializer=[weight],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="asrjs-external-data-policy-test",
    )
    model.ir_version = 9

    with tempfile.TemporaryDirectory() as temp_dir:
        inline_path = Path(temp_dir) / "inline.onnx"
        external_path = Path(temp_dir) / "external.onnx"
        onnx.save_model(model, str(inline_path))
        onnx.save_model(
            model,
            str(external_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="external.onnx.data",
            size_threshold=0,
            convert_attribute=False,
        )

        external_graph = onnx.load(str(external_path), load_external_data=False)
        assert external_graph.graph.initializer[0].data_location == TensorProto.EXTERNAL
        constant_attribute = next(
            attr.t for node in external_graph.graph.node for attr in node.attribute
            if attr.name == "value"
        )
        assert constant_attribute.data_location != TensorProto.EXTERNAL
        assert constant_attribute.raw_data

        providers = ["CPUExecutionProvider"]
        inline_session = ort.InferenceSession(str(inline_path), providers=providers)
        external_session = ort.InferenceSession(str(external_path), providers=providers)
        audio = np.array([2.0, 4.0], dtype=np.float32)
        inline_result = inline_session.run(["y"], {"x": audio})[0]
        external_result = external_session.run(["y"], {"x": audio})[0]
        np.testing.assert_array_equal(external_result, inline_result)
        del inline_session, external_session
        gc.collect()
