"""Regression tests for fail-closed Whisper publish smoke checks."""

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from io import StringIO
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "tools" / "whisper-onnx-export" / "audit_publish.py"
SPEC = importlib.util.spec_from_file_location("whisper_audit_publish", AUDIT_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class WhisperAuditSmokeGateTests(unittest.TestCase):
    def run_smoke(self, reference_json):
        checks = []

        def check(condition, message):
            checks.append((condition, message))

        with tempfile.TemporaryDirectory() as temp_dir:
            audit._run_smoke_decode(Path(temp_dir), reference_json, check)

        self.assertTrue(checks)
        self.assertFalse(checks[-1][0])
        return checks[-1][1]

    def test_missing_reference_argument_fails_smoke(self):
        message = self.run_smoke(None)
        self.assertIn("no reference JSON", message)

    def test_missing_reference_file_fails_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.json"
            checks = []
            audit._run_smoke_decode(
                Path(temp_dir), missing, lambda ok, msg: checks.append((ok, msg))
            )
        self.assertEqual(len(checks), 1)
        self.assertFalse(checks[0][0])
        self.assertIn("not found", checks[0][1])

    def test_malformed_reference_fails_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "reference.json"
            reference.write_text("{invalid", encoding="utf-8")
            checks = []
            audit._run_smoke_decode(
                Path(temp_dir), reference, lambda ok, msg: checks.append((ok, msg))
            )
        self.assertEqual(len(checks), 1)
        self.assertFalse(checks[0][0])
        self.assertIn("failed to load reference JSON", checks[0][1])

    def test_ort_session_creation_failure_fails_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "reference.json"
            reference.write_text("{}", encoding="utf-8")
            checks = []
            with patch.object(audit.ort, "InferenceSession", side_effect=RuntimeError("bad graph")):
                audit._run_smoke_decode(
                    Path(temp_dir), reference, lambda ok, msg: checks.append((ok, msg))
                )
        self.assertEqual(len(checks), 1)
        self.assertFalse(checks[0][0])
        self.assertIn("ORT session creation failed", checks[0][1])

    def test_requested_smoke_without_reference_is_aggregated_as_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "manifest.json").write_text(
                json.dumps({"format": "whisper-browser-self-export-v1", "artifacts": {}}),
                encoding="utf-8",
            )
            (root / "config.json").touch()
            (root / "tokenizer.json").touch()
            output = StringIO()
            with patch.object(audit, "_EXPECTED_GRAPH_FILES", []), patch.object(
                audit, "_EXPECTED_DATA_FILES", []
            ), redirect_stdout(output):
                passes, failures = audit.audit_directory(root, smoke_decode=True)
        self.assertGreater(passes, 0)
        self.assertEqual(failures, 1)
        self.assertIn("no reference JSON was provided", output.getvalue())

    def test_cli_returns_nonzero_when_requested_smoke_lacks_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "manifest.json").write_text(
                json.dumps({"format": "whisper-browser-self-export-v1", "artifacts": {}}),
                encoding="utf-8",
            )
            (root / "config.json").touch()
            (root / "tokenizer.json").touch()
            with patch.object(audit, "_EXPECTED_GRAPH_FILES", []), patch.object(
                audit, "_EXPECTED_DATA_FILES", []
            ), patch.object(sys, "argv", [str(AUDIT_PATH), str(root), "--smoke"]):
                with redirect_stdout(StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        audit.main()
        self.assertEqual(raised.exception.code, 1)

    def test_structural_only_mode_does_not_run_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "manifest.json").write_text(
                json.dumps({"format": "whisper-browser-self-export-v1", "artifacts": {}}),
                encoding="utf-8",
            )
            messages = []
            audit._audit_variant_dir(
                root, lambda ok, msg: messages.append((ok, msg)), smoke_decode=False
            )
        self.assertFalse(any("Smoke" in message or "smoke" in message for _, message in messages))


if __name__ == "__main__":
    unittest.main()
