"""Runner preflight and receipt parsing; no browser/build/dependency load."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).with_name('browser-demo-smoke.py')
spec = importlib.util.spec_from_file_location('companion_smoke', TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompanionSmokeTests(unittest.TestCase):
    def test_local_model_requires_nonempty_selected_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                module.local_model_inputs(root, 'fp16')
            for name in ('encoder-model.fp16.onnx', 'decoder_joint-model.onnx', 'vocab.txt'):
                (root / name).write_bytes(b'fixture-not-a-model')
            self.assertEqual(len(module.local_model_inputs(root, 'fp16')), 3)
            (root / 'encoder-model.fp16.onnx.data').write_bytes(b'')
            with self.assertRaises(ValueError):
                module.local_model_inputs(root, 'fp16')

    def test_streamed_digest_matches_full_digest(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture'
            data = b'abcd' * (1024 * 1024)
            path.write_bytes(data)
            self.assertEqual(module.digest(path), hashlib.sha256(data).hexdigest())

    def test_build_receipt_parses_amid_logs(self):
        receipt = {'status': 'PASS_IN_MEMORY_COMPANION_BUILD', 'outputs': [{'fileName': 'index.html'}]}
        self.assertEqual(module.parse_build('vite log\n' + json.dumps(receipt, indent=2) + '\nready'), receipt)

    def test_missing_duplicate_empty_and_malformed_receipts_fail(self):
        for value in ('log only', module.BUILD_MARKER, module.BUILD_MARKER * 2,
                      json.dumps({'status': 'PASS_IN_MEMORY_COMPANION_BUILD', 'outputs': []}, indent=2)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.parse_build(value)

    def test_existing_output_is_preserved_before_build_or_browser(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'existing.json'
            output.write_bytes(b'original')
            result = subprocess.run([sys.executable, str(TOOL), '--consumer-root', str(root / 'missing'),
                '--output', str(output)], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Refusing existing output', result.stderr)
            self.assertEqual(output.read_bytes(), b'original')

    def test_requested_missing_consumer_fails_without_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'receipt.json'
            result = subprocess.run([sys.executable, str(TOOL), '--consumer-root', str(root / 'missing'),
                '--output', str(output)], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('vite.config.js', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
