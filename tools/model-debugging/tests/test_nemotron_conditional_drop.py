from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from build_nemotron_conditional_drop import build_candidate, sha256  # noqa: E402


def make_tiny_encoder(path: Path) -> None:
    inputs = [
        helper.make_tensor_value_info(
            "/pre_encode/out/Add_output_0", TensorProto.FLOAT, [1, 65, 1]
        ),
        helper.make_tensor_value_info("length", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("cache_last_channel_len", TensorProto.INT64, [1]),
    ]
    outputs = [
        helper.make_tensor_value_info("/Slice_output_0", TensorProto.FLOAT, [1, None, 1]),
        helper.make_tensor_value_info("/Sub_output_0", TensorProto.INT64, [1]),
    ]
    initializers = [
        numpy_helper.from_array(np.asarray([2], dtype=np.int64), name="starts"),
        numpy_helper.from_array(np.asarray([65], dtype=np.int64), name="ends"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="axes"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="steps"),
        numpy_helper.from_array(np.asarray(2, dtype=np.int64), name="static_two"),
    ]
    nodes = [
        helper.make_node(
            "Slice",
            ["/pre_encode/out/Add_output_0", "starts", "ends", "axes", "steps"],
            ["/Slice_output_0"],
            name="/Slice",
        ),
        helper.make_node("Sub", ["length", "static_two"], ["/Sub_output_0"], name="/Sub"),
    ]
    graph = helper.make_graph(nodes, "tiny-nemotron-encoder", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save_model(model, str(path))


class NemotronConditionalDropRegression(unittest.TestCase):
    def test_candidate_uses_zero_drop_only_without_channel_cache(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nemotron-conditional-drop-") as temp:
            root = Path(temp)
            source = root / "source.onnx"
            output_dir = root / "candidate"
            make_tiny_encoder(source)
            source_hash = sha256(source)

            receipt = build_candidate(source, output_dir)

            self.assertEqual(receipt["status"], "CANDIDATE_GRAPH_CHECKED_NO_NUMERICAL_CLAIM")
            self.assertEqual(receipt["sourceGraphSha256"], source_hash)
            self.assertEqual(sha256(source), source_hash)
            self.assertEqual(receipt["otherOriginalNodesModified"], False)
            self.assertEqual(receipt["externalData"], [])
            self.assertEqual(
                (output_dir / "config.json").read_text(encoding="utf-8"),
                '{\n  "format": "asrjs.nemotron-rnnt",\n  "schemaVersion": 1,\n'
                '  "family": "nemotron-rnnt",\n  "encoderWindowing": {\n'
                '    "firstChunkFrames": 49,\n    "shiftFrames": 56,\n'
                '    "leftContextFrames": 9\n  }\n}\n',
            )

            options = ort.SessionOptions()
            original = ort.InferenceSession(
                str(source), sess_options=options, providers=["CPUExecutionProvider"]
            )
            candidate = ort.InferenceSession(
                str(output_dir / "encoder.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self.assertEqual(original.get_providers(), ["CPUExecutionProvider"])
            self.assertEqual(candidate.get_providers(), ["CPUExecutionProvider"])
            audio = np.arange(65, dtype=np.float32).reshape(1, 65, 1)

            first_feed = {
                "/pre_encode/out/Add_output_0": audio,
                "length": np.asarray([49], dtype=np.int64),
                "cache_last_channel_len": np.asarray([0], dtype=np.int64),
            }
            original_first, original_first_length = original.run(None, first_feed)
            candidate_first, candidate_first_length = candidate.run(None, first_feed)
            self.assertEqual(original_first.shape[1], 63)
            self.assertEqual(candidate_first.shape[1], 65)
            np.testing.assert_array_equal(candidate_first[0, :, 0], audio[0, :, 0])
            self.assertEqual(int(original_first_length[0]), 47)
            self.assertEqual(int(candidate_first_length[0]), 49)

            continuation_feed = {
                **first_feed,
                "length": np.asarray([65], dtype=np.int64),
                "cache_last_channel_len": np.asarray([56], dtype=np.int64),
            }
            original_next = original.run(None, continuation_feed)
            candidate_next = candidate.run(None, continuation_feed)
            np.testing.assert_array_equal(candidate_next[0], original_next[0])
            np.testing.assert_array_equal(candidate_next[1], original_next[1])

            with self.assertRaises(FileExistsError):
                build_candidate(source, output_dir)


if __name__ == "__main__":
    unittest.main()
