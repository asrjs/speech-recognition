"""Tiny model-neutral artifact tests; no large weights or ASR inference."""

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx_artifacts import external_ranges, iter_tensors
from repack_onnx_external_data import copy_range, range_hash, repack, repoint_tensor


def save_fixture(root):
    root = Path(root)
    root.mkdir(exist_ok=True)
    # Deliberate slices with unused prefix/suffix bytes, shared source file.
    values = np.array([1, 2, 3, 4], dtype=np.float32)
    (root / "weights.bin").write_bytes(b"HEAD" + values.tobytes() + b"TAIL")
    bias = TensorProto(name="bias", data_type=TensorProto.FLOAT, dims=[2], data_location=TensorProto.EXTERNAL)
    constant = TensorProto(name="constant", data_type=TensorProto.FLOAT, dims=[2], data_location=TensorProto.EXTERNAL)
    repoint_tensor(bias, "weights.bin", 4, 8)
    repoint_tensor(constant, "weights.bin", 12, 8)
    inline = numpy_helper.from_array(np.array([5, 6], dtype=np.float32), name="inline")
    graph = helper.make_graph([
        helper.make_node("Constant", [], ["c"], value=constant),
        helper.make_node("Add", ["x", "bias"], ["a"]),
        helper.make_node("Add", ["a", "c"], ["b"]),
        helper.make_node("Add", ["b", "inline"], ["y"]),
    ], "sliced", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])], [bias, inline])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = root / "graph.onnx"
    path.write_bytes(model.SerializeToString())
    return path


class ArtifactTests(unittest.TestCase):
    def test_repack_preserves_initializer_attribute_inline_and_slice_values(self):
        with TemporaryDirectory() as root:
            source = save_fixture(Path(root) / "source")
            original_bytes = source.read_bytes()
            destination = Path(root) / "candidate"
            receipt = repack(source, destination, range_hash(source))
            self.assertEqual(receipt["status"], "REPACKED_BYTE_IDENTICAL_UNVALIDATED")
            self.assertEqual(receipt["external_tensor_count"], 2)
            self.assertEqual(receipt["source_external_file_count"], 1)
            self.assertEqual(receipt["external_data"]["bytes"], 16)
            self.assertEqual(source.read_bytes(), original_bytes)
            left = onnx.load(str(source), load_external_data=True)
            right = onnx.load(str(destination / source.name), load_external_data=True)
            self.assertEqual([t.name for t in iter_tensors(left)], [t.name for t in iter_tensors(right)])
            for expected, actual in zip(iter_tensors(left), iter_tensors(right)):
                np.testing.assert_array_equal(numpy_helper.to_array(actual), numpy_helper.to_array(expected))
            packed = onnx.load(str(destination / source.name), load_external_data=False)
            ranges = list(external_ranges(packed, destination / source.name))
            self.assertEqual([r.offset for r in ranges], [0, 8])
            self.assertEqual({r.location for r in ranges}, {"graph.onnx.data"})
            self.assertEqual(json.loads((destination / "repack-receipt.json").read_text())["status"], receipt["status"])

    def test_recursive_traversal_covers_nested_graph_sparse_tensor_and_function(self):
        tensor = helper.make_tensor("nested", TensorProto.FLOAT, [1], [1])
        subgraph = helper.make_graph([helper.make_node("Constant", [], ["v"], value=tensor)], "sub", [], [])
        graph = helper.make_graph([helper.make_node("If", ["condition"], ["y"], then_branch=subgraph)], "outer", [], [])
        model = helper.make_model(graph)
        sparse = model.graph.sparse_initializer.add()
        sparse.values.CopyFrom(helper.make_tensor("sparse-values", TensorProto.FLOAT, [1], [1]))
        sparse.indices.CopyFrom(helper.make_tensor("sparse-indices", TensorProto.INT64, [1], [0]))
        function = model.functions.add()
        function.node.add().CopyFrom(helper.make_node("Constant", [], ["f"], value=helper.make_tensor("function", TensorProto.FLOAT, [1], [2])))
        self.assertEqual({t.name for t in iter_tensors(model)}, {"nested", "sparse-values", "sparse-indices", "function"})

    def test_preexisting_directory_and_wrong_hash_preserve_all_files(self):
        with TemporaryDirectory() as root:
            source = save_fixture(Path(root) / "source")
            destination = Path(root) / "candidate"
            destination.mkdir()
            sentinel = destination / "user-owned"
            sentinel.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                repack(source, destination, range_hash(source))
            self.assertEqual(sentinel.read_bytes(), b"preserve")
            fresh = Path(root) / "wrong-hash"
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                repack(source, fresh, "0" * 64)
            self.assertFalse(fresh.exists())

    def test_invalid_external_ranges_fail_before_candidate_creation(self):
        cases = [
            ("location", "../outside.bin"), ("location", "missing.bin"),
            ("offset", "-1"), ("length", "0"), ("length", "1000"), ("offset", "1000"),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value), TemporaryDirectory() as root:
                source = save_fixture(Path(root) / "source")
                model = onnx.load(str(source), load_external_data=False)
                # Attribute, not initializer: this was the missed audit boundary.
                tensor = model.graph.node[0].attribute[0].t
                for entry in tensor.external_data:
                    if entry.key == key:
                        entry.value = value
                source.write_bytes(model.SerializeToString())
                destination = Path(root) / "candidate"
                with self.assertRaises(ValueError):
                    repack(source, destination, range_hash(source))
                self.assertFalse(destination.exists())

    def test_empty_sidecar_and_duplicate_metadata_are_rejected(self):
        for duplicate in [False, True]:
            with self.subTest(duplicate=duplicate), TemporaryDirectory() as root:
                source = save_fixture(Path(root) / "source")
                model = onnx.load(str(source), load_external_data=False)
                if duplicate:
                    entry = model.graph.initializer[0].external_data.add()
                    entry.key, entry.value = "location", "weights.bin"
                else:
                    (source.parent / "weights.bin").write_bytes(b"")
                with self.assertRaises(ValueError):
                    list(external_ranges(model, source))

    def test_alias_ranges_are_copied_separately_without_dedup_semantic_changes(self):
        with TemporaryDirectory() as root:
            source = save_fixture(Path(root) / "source")
            model = onnx.load(str(source), load_external_data=False)
            repoint_tensor(model.graph.node[0].attribute[0].t, "weights.bin", 4, 8)
            source.write_bytes(model.SerializeToString())
            receipt = repack(source, Path(root) / "candidate", range_hash(source))
            self.assertEqual(receipt["ranges"][0]["sha256"], receipt["ranges"][1]["sha256"])
            self.assertEqual(receipt["external_data"]["bytes"], 16)

    def test_companions_are_explicit_verified_and_cannot_escape_or_collide(self):
        with TemporaryDirectory() as root:
            source = save_fixture(Path(root) / "source")
            (source.parent / "vocab.txt").write_bytes(b"indexed vocabulary")
            receipt = repack(source, Path(root) / "candidate", range_hash(source), ["vocab.txt"])
            self.assertEqual(receipt["companions"][0]["sha256"], range_hash(source.parent / "vocab.txt"))
            for name in ["../vocab.txt", "graph.onnx", "absent", "vocab.txt/vocab.txt"]:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    repack(source, Path(root) / "invalid", range_hash(source), [name])
            self.assertFalse((Path(root) / "invalid").exists())

    def test_copy_detects_truncation_instead_of_silent_short_write(self):
        with TemporaryDirectory() as root:
            source = Path(root) / "short"
            source.write_bytes(b"abc")
            with self.assertRaisesRegex(ValueError, "Unexpected EOF"):
                copy_range(source, 0, 4, io.BytesIO())

    def test_large_offsets_are_exact_decimal_strings(self):
        tensor = TensorProto(data_location=TensorProto.EXTERNAL)
        repoint_tensor(tensor, "large.data", 2**31 + 7, 2**32 + 11)
        info = {entry.key: entry.value for entry in tensor.external_data}
        self.assertEqual(info["offset"], "2147483655")
        self.assertEqual(info["length"], "4294967307")


if __name__ == "__main__":
    unittest.main()
