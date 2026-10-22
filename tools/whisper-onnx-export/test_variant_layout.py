"""Test exporter path orchestration without importing torch/ONNX or model weights.

The real organizer function is compiled from its AST; conversion is a test double.
These tests establish naming/layout only, not numerical FP16 parity.
"""
import ast
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock


def organizer_namespace():
    source = Path(__file__).with_name("export_whisper.py")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_organize_variant_dirs")
    namespace = {"Path": Path, "shutil": shutil, "_write_publish_readme": Mock(), "_validate_variant_graphs": Mock()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class VariantLayoutTests(unittest.TestCase):
    def run_variant(self, variant="fp16", dtype="float32"):
        namespace = organizer_namespace()
        converted = []
        def convert(directory, names, **options):
            converted.append((list(names), options))
            for name in names:
                self.assertTrue((directory / name.replace(".fp16", "")).exists())
                (directory / name).write_bytes(b"fp16-graph")
                (directory / (name + ".data")).write_bytes(b"fp16-weights")
            return True
        namespace["convert_fp16_safe"] = convert
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            graphs = ["encoder_model.onnx", "decoder_step.onnx"]
            for name in graphs:
                (root / name).write_bytes(b"fp32-graph")
                (root / (name + ".data")).write_bytes(b"fp32-weights")
            (root / "manifest.json").write_text('{"graphs": ["encoder_model.onnx", "decoder_step.onnx"]}')
            (root / "config.json").write_text("{}")
            with contextlib.redirect_stdout(io.StringIO()):
                namespace["_organize_variant_dirs"](root, model_id="test/model", variant=variant, fp16=True, int8=False,
                    dtype=dtype, use_external_data=True, external_data_one_file=True,
                    external_data_threshold=1024, validate_path_only=True, quantize="none", graph_names=graphs)
            folder = root / variant
            self.assertTrue((folder / "manifest.json").exists())
            self.assertTrue((folder / "config.json").exists())
            if variant == "fp16" and dtype != "float16":
                self.assertEqual(converted, [(graphs, {"use_external_data": True, "all_tensors_to_one_file": True,
                    "size_threshold": 1024, "validate_path": True})])
                for name in graphs:
                    self.assertEqual((folder / name).read_bytes(), b"fp16-graph")
                    self.assertEqual((folder / (name + ".data")).read_bytes(), b"fp16-weights")
                self.assertEqual(sorted(p.name for p in folder.glob("*.onnx")), sorted(graphs))
            else:
                self.assertEqual(converted, [])
                for name in graphs:
                    self.assertEqual((folder / name).read_bytes(), b"fp32-graph")
            if dtype == "float16":
                namespace["_validate_variant_graphs"].assert_called_once_with(folder, graphs, "FP16")

    def test_post_export_fp16_keeps_manifest_graph_names(self):
        self.run_variant()

    def test_export_time_fp16_skips_post_conversion(self):
        self.run_variant(dtype="float16")

    def test_fp32_does_not_convert(self):
        self.run_variant(variant="fp32")


if __name__ == "__main__":
    unittest.main()
