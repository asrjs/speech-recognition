"""Weight-free guards and real native constant delivery, not ASR acceptance."""
import copy
import unittest

import numpy as np
import onnx
from onnx import helper, TensorProto

from browser_head_replay import tensor
from frozen_encoder_replay import SCOPE, make_graph, verify_delivery


def fixture():
    inputs = [helper.make_tensor_value_info("audio_signal", TensorProto.FLOAT, [1, 2, "features"]),
              helper.make_tensor_value_info("length", TensorProto.INT64, [1])]
    outputs = [helper.make_tensor_value_info("outputs", TensorProto.FLOAT, [1, 3, "frames"]),
               helper.make_tensor_value_info("encoded_lengths", TensorProto.INT64, [1])]
    source = helper.make_model(helper.make_graph([], "source_contract_only", inputs, outputs),
                               opset_imports=[helper.make_opsetid("", 17)])
    source.ir_version = 8
    stages = {"features": tensor(np.ones((1, 2, 4), dtype=np.float32)),
              "features_length": tensor(np.array([3], dtype=np.int64)),
              "encoder": tensor(np.array([[[0., -0.], [1.25, -2.5], [3., 4.]]], dtype=np.float32)),
              "encoder_length": tensor(np.array([2], dtype=np.int64))}
    cfg = {"preprocessor": {"features": 2}, "joint": {"jointnet": {"encoder_hidden": 3}}}
    return source, stages, cfg


class FrozenBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.source, self.stages, self.cfg = fixture()

    def make(self):
        return make_graph(self.source, self.stages, self.cfg, "synthetic")

    def test_preserves_contract_and_never_mutates_source(self):
        original = self.source.SerializeToString()
        model, _, _ = self.make()
        self.assertEqual(self.source.SerializeToString(), original)
        self.assertEqual(list(model.graph.input), list(self.source.graph.input))
        self.assertEqual(list(model.graph.output), list(self.source.graph.output))
        self.assertEqual([node.op_type for node in model.graph.node], ["Constant", "Constant"])
        self.assertEqual(len(model.graph.initializer), 0)
        props = {item.key: item.value for item in model.metadata_props}
        self.assertEqual(props["asrjs.scope"], SCOPE)
        self.assertEqual(props["asrjs.inputs"], "IGNORED")
        onnx.checker.check_model(model)

    def test_native_delivery_is_bitwise_including_signed_zero(self):
        model, feeds, expected = self.make()
        checks = verify_delivery(model.SerializeToString(), feeds, expected)
        self.assertEqual([check["input"] for check in checks], ["captured_features", "zero_features"])
        self.assertTrue(all(item["bitwise_exact"] for check in checks for item in check["outputs"].values()))
        self.assertEqual(checks[0]["outputs"], checks[1]["outputs"])

    def test_serialization_is_deterministic(self):
        self.assertEqual(self.make()[0].SerializeToString(), self.make()[0].SerializeToString())

    def test_name_dtype_rank_and_duplicate_contracts_fail(self):
        for mutation in (lambda source: setattr(source.graph.input[0], "name", "wrong"),
                         lambda source: setattr(source.graph.output[0].type.tensor_type, "elem_type", TensorProto.FLOAT16),
                         lambda source: source.graph.input[0].type.tensor_type.shape.dim.pop(),
                         lambda source: source.graph.input.append(copy.deepcopy(source.graph.input[0]))):
            with self.subTest(mutation=mutation):
                self.source, self.stages, self.cfg = fixture()
                mutation(self.source)
                with self.assertRaises(ValueError):
                    self.make()

    def test_static_dimensions_are_not_relaxed(self):
        self.source.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 5
        with self.assertRaisesRegex(ValueError, "static shape"):
            self.make()

    def test_invalid_lengths_fail(self):
        for key, value in (("features_length", 0), ("features_length", 5),
                           ("encoder_length", -1), ("encoder_length", 3)):
            with self.subTest(key=key, value=value):
                self.source, self.stages, self.cfg = fixture()
                self.stages[key]["data"] = [value]
                with self.assertRaises(ValueError):
                    self.make()

    def test_nonfinite_dtype_and_data_count_fail(self):
        for mutation in (lambda stages: stages["encoder"]["data"].__setitem__(0, float("nan")),
                         lambda stages: stages["encoder"].update(dtype="float64"),
                         lambda stages: stages["features"]["data"].pop(),
                         lambda stages: stages["encoder_length"].update(data=[True])):
            with self.subTest(mutation=mutation):
                self.source, self.stages, self.cfg = fixture()
                mutation(self.stages)
                with self.assertRaises(ValueError):
                    self.make()

    def test_channel_layout_and_sample_id_fail(self):
        for cfg, sample_id in (({**self.cfg, "preprocessor": {"features": 3}}, "x"), (self.cfg, "")):
            with self.subTest(cfg=cfg, sample_id=sample_id), self.assertRaises(ValueError):
                make_graph(self.source, self.stages, cfg, sample_id)

    def test_native_comparison_rejects_signed_zero_difference(self):
        model, feeds, expected = self.make()
        expected["outputs"][0, 0, 1] = 0.
        with self.assertRaisesRegex(ValueError, "delivery differs"):
            verify_delivery(model.SerializeToString(), feeds, expected)


if __name__ == "__main__":
    unittest.main()
