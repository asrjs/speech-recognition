from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from tools.split_variadic_concat import (
    move_late_producers_before_consumers,
    split_variadic_concat_nodes,
)


class SplitVariadicConcatTests(unittest.TestCase):
    def test_ordered_tree_preserves_values_and_output_name(self) -> None:
        inputs = [
            helper.make_tensor_value_info(f"input_{i}", TensorProto.FLOAT, [1, 1])
            for i in range(24)
        ]
        output = helper.make_tensor_value_info("joined", TensorProto.FLOAT, [24, 1])
        graph = helper.make_graph(
            [helper.make_node("Concat", [item.name for item in inputs], ["joined"], name="wide_concat", axis=0)],
            "concat-regression",
            inputs,
            [output],
        )
        original = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17)],
            ir_version=10,
        )
        candidate = onnx.ModelProto()
        candidate.CopyFrom(original)

        counts = split_variadic_concat_nodes(
            candidate,
            node_names=("wide_concat",),
            max_inputs=6,
        )
        onnx.checker.check_model(candidate)

        concat_nodes = [node for node in candidate.graph.node if node.op_type == "Concat"]
        self.assertEqual(counts, {"wide_concat": 5})
        self.assertEqual(max(len(node.input) for node in concat_nodes), 6)
        self.assertEqual(concat_nodes[-1].output[0], "joined")

        feeds = {
            f"input_{i}": np.array([[i]], dtype=np.float32)
            for i in range(24)
        }
        original_result = ort.InferenceSession(
            original.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(["joined"], feeds)[0]
        candidate_result = ort.InferenceSession(
            candidate.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(["joined"], feeds)[0]
        np.testing.assert_array_equal(candidate_result, original_result)
        np.testing.assert_array_equal(candidate_result[:, 0], np.arange(24, dtype=np.float32))

    def test_missing_target_fails_closed(self) -> None:
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"], name="not-a-concat")],
            "missing-target",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        with self.assertRaisesRegex(ValueError, "target Concat nodes not found"):
            split_variadic_concat_nodes(model, node_names=("wide_concat",))

    def test_minimal_move_repairs_late_export_cast(self) -> None:
        graph = helper.make_graph(
            [
                helper.make_node("Identity", ["middle"], ["y"], name="consumer"),
                helper.make_node("Identity", ["x"], ["independent"], name="unrelated"),
                helper.make_node("Identity", ["x"], ["middle"], name="late-producer"),
            ],
            "late-producer-regression",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        moved = move_late_producers_before_consumers(model)
        onnx.checker.check_model(model)
        self.assertEqual([node.name for node in model.graph.node], ["late-producer", "consumer", "unrelated"])
        self.assertEqual(moved, ["late-producer:middle"])


if __name__ == "__main__":
    unittest.main()
