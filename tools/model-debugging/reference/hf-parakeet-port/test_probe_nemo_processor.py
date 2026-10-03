"""CLI preflight regressions, independent of NeMo, weights and capture data."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).with_name('probe_nemo_processor.py')


class ProcessorPreflightTests(unittest.TestCase):
    def run_cli(self, config, capture, output, config_hash='0' * 64, onnx=False, stages=False, pairwise=False):
        command = [sys.executable, str(TOOL), '--config', str(config),
            '--expected-config-sha256', config_hash, '--capture', str(capture),
            '--expected-capture-sha256', '0' * 64, '--sample-id', 'test',
            '--output', str(output)]
        if onnx:
            command.append('--onnx-probe')
        if stages:
            command.append('--stage-probe')
        if pairwise:
            command.append('--pairwise-normalization')
        return subprocess.run(command, capture_output=True, text=True, timeout=10)

    def test_pairwise_requires_stages_before_inputs_or_imports(self):
        result = self.run_cli(Path('missing.json'), Path('missing.gz'), Path('missing-output.json'), pairwise=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--pairwise-normalization requires --stage-probe', result.stderr)

    def test_requested_pairwise_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'receipt.json'
            output.write_bytes(b'original')
            result = self.run_cli(root / 'missing.json', root / 'missing.gz', output,
                                  onnx=True, stages=True, pairwise=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Refusing existing output', result.stderr)
            self.assertEqual(output.read_bytes(), b'original')

    def test_stage_probe_requires_export_before_inputs_or_imports(self):
        result = self.run_cli(Path('missing.json'), Path('missing.gz'), Path('missing-output.json'), stages=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--stage-probe requires --onnx-probe', result.stderr)

    def test_requested_stages_preserve_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'receipt.json'
            output.write_bytes(b'original')
            result = self.run_cli(root / 'missing.json', root / 'missing.gz', output, onnx=True, stages=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Refusing existing output', result.stderr)
            self.assertEqual(output.read_bytes(), b'original')

    def test_requested_export_preserves_existing_output_before_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'receipt.json'
            output.write_bytes(b'original')
            result = self.run_cli(root / 'missing.json', root / 'missing.gz', output, onnx=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Refusing existing output', result.stderr)
            self.assertEqual(output.read_bytes(), b'original')

    def test_existing_output_is_preserved_before_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'receipt.json'
            output.write_bytes(b'original')
            result = self.run_cli(root / 'missing.json', root / 'missing.gz', output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Refusing existing output', result.stderr)
            self.assertEqual(output.read_bytes(), b'original')

    def test_config_cannot_be_output(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            config.write_bytes(b'original')
            result = self.run_cli(config, config.with_suffix('.gz'), config)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('protected input', result.stderr)
            self.assertEqual(config.read_bytes(), b'original')

    def test_config_identity_fails_before_optional_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_bytes(b'{}')
            result = self.run_cli(config, root / 'missing.gz', root / 'receipt.json')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Config identity differs', result.stderr)
            self.assertFalse((root / 'receipt.json').exists())

    def test_capture_identity_fails_before_optional_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_bytes(b'{}')
            capture = root / 'capture.gz'
            capture.write_bytes(b'not the oracle')
            result = self.run_cli(config, capture, root / 'receipt.json',
                                  hashlib.sha256(config.read_bytes()).hexdigest())
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Capture identity differs', result.stderr)
            self.assertFalse((root / 'receipt.json').exists())


if __name__ == '__main__':
    unittest.main()
