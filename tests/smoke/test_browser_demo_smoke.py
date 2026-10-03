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
