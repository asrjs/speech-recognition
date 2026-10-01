"""Weight-free output safety regressions; never model parity evidence."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from artifact_outputs import require_new_outputs, write_new_text

ROOT = Path(__file__).resolve().parent


def load_script(relative, **dependency_stubs):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in dependency_stubs}
    sys.modules.update(dependency_stubs)
    try:
        with patch.object(sys, "path", [str(path.parent), *sys.path]):
            spec.loader.exec_module(module)
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module


class ReceiptCallerTests(unittest.TestCase):
    def test_all_callers_refuse_existing_receipts_before_reading_or_acquiring_inputs(self):
        official = load_script("model-debugging/reference/hf-parakeet-port/python-nemo-inspect.py",
            torch=SimpleNamespace(), **{"nemo.collections.asr.models": SimpleNamespace(ASRModel=object)})
        diagnostic = load_script("model-debugging/reference/hf-parakeet-port/compare_js_feature_boundaries.py")
        acquire = load_script("model-debugging/reference/orukeet/acquire_source.py")
        audit = load_script("model-debugging/reference/orukeet/audit_source.py", acquire_source=acquire)
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "prior.json"
            original = b'{"prior":true}\n'
            for module, args in (
                (official, ["--audio", "missing.wav", "--stack-timeout", "0"]),
                (diagnostic, ["--audio", "missing.wav", "--source-config", "missing.json",
                    "--official-capture", "missing-capture.json", "--sample-id", "control"]),
                (acquire, []),
                (audit, ["--nemo", "missing.nemo", "--v3-vocab", "missing-vocab.txt"]),
            ):
                receipt.write_bytes(original)
                with self.subTest(caller=module.__name__), patch.object(sys, "argv",
                        [module.__file__, *args, "--output", str(receipt)]), patch.object(
                        acquire, "HfApi", side_effect=AssertionError("No network in safety tests")):
                    with self.assertRaises(FileExistsError):
                        module.main()
                    self.assertEqual(receipt.read_bytes(), original)

    def test_official_capture_rejects_aliased_outputs_before_inference(self):
        official = load_script("model-debugging/reference/hf-parakeet-port/python-nemo-inspect.py",
            torch=SimpleNamespace(), **{"nemo.collections.asr.models": SimpleNamespace(ASRModel=object)})
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "new.json")
            with patch.object(sys, "argv", [official.__file__, "--output", output, "--stage-output", output]):
                with self.assertRaisesRegex(ValueError, "distinct"):
                    official.main()
            self.assertFalse(Path(output).exists())

    def test_acquisition_refuses_a_receipt_in_the_planned_source_directory(self):
        acquire = load_script("model-debugging/reference/orukeet/acquire_source.py")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            with patch.object(sys, "argv", [acquire.__file__, "--source-dir", str(source),
                    "--output", str(source / "config.json")]), patch.object(acquire, "HfApi",
                    side_effect=AssertionError("No network in safety tests")):
                with self.assertRaisesRegex(ValueError, "source directory"):
                    acquire.main()
            self.assertFalse(source.exists())

    def test_acquisition_failure_updates_only_its_new_owned_receipt(self):
        acquire = load_script("model-debugging/reference/orukeet/acquire_source.py")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            with patch.object(sys, "argv", [acquire.__file__, "--output", str(output)]), patch.object(
                    acquire, "HfApi", side_effect=AssertionError("No network in safety tests")):
                with self.assertRaisesRegex(AssertionError, "No network"):
                    acquire.main()
            import json
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "FAIL")


class NewOutputTests(unittest.TestCase):
    def test_preflight_does_not_create_outputs_or_parent_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new" / "receipt.json"
            require_new_outputs([output, None])
            self.assertFalse(output.parent.exists())
            write_new_text(output, "new receipt")
            self.assertEqual(output.read_text(encoding="utf-8"), "new receipt")

    def test_existing_and_raced_targets_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            require_new_outputs([output])
            output.write_bytes(b"another owner")
            with self.assertRaises(FileExistsError):
                write_new_text(output, "replacement")
            with self.assertRaises(FileExistsError):
                require_new_outputs([output])
            self.assertEqual(output.read_bytes(), b"another owner")

    def test_normalized_aliases_and_future_input_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            alias = Path(directory) / "subdir" / ".." / "receipt.json"
            with self.assertRaisesRegex(ValueError, "distinct"):
                require_new_outputs([output, alias])
            with self.assertRaisesRegex(ValueError, "protected input"):
                require_new_outputs([output], protected_inputs=[alias])
            self.assertFalse(output.exists())

    def test_exclusive_gzip_mode_preserves_prior_capture(self):
        import gzip
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.json.gz"
            with gzip.open(output, "xt", encoding="utf-8") as handle:
                handle.write("original capture")
            original = output.read_bytes()
            with self.assertRaises(FileExistsError), gzip.open(output, "xt", encoding="utf-8"):
                pass
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
