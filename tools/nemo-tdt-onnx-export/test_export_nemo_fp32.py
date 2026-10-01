"""Small exporter safety/contract tests; these do not certify model parity."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np

from export_nemo_fp32 import audit_graph, create_output_directory, require_tdt_model, sha256


class ExportSafetyTests(unittest.TestCase):
    def test_new_directory_cannot_overwrite_even_empty_previous_directory(self):
        with TemporaryDirectory() as root:
            path = create_output_directory(Path(root) / "export")
            with self.assertRaises(FileExistsError):
                create_output_directory(path)

    def test_hash_is_content_based(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "source"
            path.write_bytes(b"abc")
            self.assertEqual(sha256(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_model_family_guard_does_not_use_brand_or_dimensions(self):
        model = SimpleNamespace(cfg=SimpleNamespace(loss=SimpleNamespace(loss_name="tdt")),
                                encoder=SimpleNamespace(export_cache_support=False),
                                decoder=SimpleNamespace(blank_as_pad=True),
                                list_export_subnets=lambda: ["encoder", "decoder_joint"])
        require_tdt_model(model)
        for owner, field, bad_value in [(model.cfg.loss, "loss_name", "rnnt"),
                                         (model.encoder, "export_cache_support", True),
                                         (model.decoder, "blank_as_pad", False)]:
            original = getattr(owner, field)
            setattr(owner, field, bad_value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                require_tdt_model(model)
            setattr(owner, field, original)
        model.prompt_kernel = object()
        with self.assertRaises(ValueError):
            require_tdt_model(model)

    def save_graph(self, directory, dtype=TensorProto.FLOAT, external=False):
        path = Path(directory) / "graph.onnx"
        tensor = numpy_helper.from_array(np.ones(2, dtype=np.float32), name="bias")
        graph = helper.make_graph([helper.make_node("Add", ["x", "bias"], ["y"])], "safety-test",
                                  [helper.make_tensor_value_info("x", dtype, ["batch", 2])],
                                  [helper.make_tensor_value_info("y", dtype, ["batch", 2])], [tensor])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.save_model(model, str(path), save_as_external_data=external,
                        all_tensors_to_one_file=True, location="graph.data", size_threshold=0)
        return path

    def test_graph_contract_and_external_file_hash_are_audited(self):
        with TemporaryDirectory() as root:
            result = audit_graph(self.save_graph(root, external=True), {"x": ("FLOAT", 2)}, {"y": ("FLOAT", 2)})
            self.assertEqual(result["inputs"]["x"]["shape"], ["batch", 2])
            self.assertEqual(result["external_data"][0]["bytes"], 8)
            self.assertEqual(len(result["external_data"][0]["sha256"]), 64)

    def test_constant_attribute_external_data_is_audited(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "constant.onnx"
            value = numpy_helper.from_array(np.ones(2, dtype=np.float32), name="constant-weight")
            graph = helper.make_graph([
                helper.make_node("Constant", [], ["bias"], value=value),
                helper.make_node("Add", ["x", "bias"], ["y"]),
            ], "attribute-test", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            onnx.save_model(model, str(path), save_as_external_data=True,
                            location="constant.data", size_threshold=0, convert_attribute=True)
            result = audit_graph(path, {"x": ("FLOAT", 1)}, {"y": ("FLOAT", 1)})
            self.assertEqual(len(result["external_data"]), 1)
            self.assertEqual(result["external_data"][0]["bytes"], 8)

    def test_wrong_names_dtypes_and_ranks_fail(self):
        with TemporaryDirectory() as root:
            path = self.save_graph(root)
            for expected in [{"other": ("FLOAT", 2)}, {"x": ("INT32", 2)}, {"x": ("FLOAT", 3)}]:
                with self.subTest(expected=expected), self.assertRaises(ValueError):
                    audit_graph(path, expected, {"y": ("FLOAT", 2)})

    def test_external_data_cannot_escape_graph_directory_or_be_missing(self):
        with TemporaryDirectory() as root:
            path = self.save_graph(root, external=True)
            for location in ["../outside.data", "absent.data"]:
                model = onnx.load(str(path), load_external_data=False)
                for entry in model.graph.initializer[0].external_data:
                    if entry.key == "location":
                        entry.value = location
                onnx.save_model(model, str(path))
                with self.subTest(location=location), self.assertRaises(ValueError):
                    audit_graph(path, {"x": ("FLOAT", 2)}, {"y": ("FLOAT", 2)})

    def test_external_data_out_of_bounds_range_fails(self):
        with TemporaryDirectory() as root:
            path = self.save_graph(root, external=True)
            model = onnx.load(str(path), load_external_data=False)
            for entry in model.graph.initializer[0].external_data:
                if entry.key == "length":
                    entry.value = "1000"
            onnx.save_model(model, str(path))
            with self.assertRaisesRegex(ValueError, "range invalid"):
                audit_graph(path, {"x": ("FLOAT", 2)}, {"y": ("FLOAT", 2)})


if __name__ == "__main__":
    unittest.main()
