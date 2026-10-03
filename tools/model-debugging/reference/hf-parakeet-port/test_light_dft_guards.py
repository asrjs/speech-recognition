"""Light DFT CLI rejection controls, without capture/model execution."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from probe_light_dft_features import build_power_probe
from replay_packed_fft import build_probe

TOOL = Path(__file__).with_name('probe_light_dft_features.py')


class LightDftGuardTests(unittest.TestCase):
    def test_power_probe_preserves_baseline_and_changes_only_named_contract(self):
        baseline = build_probe()
        self.assertEqual(build_power_probe('fp32-rounded'), baseline)
        library = build_power_probe('library')
        restored = library.replace("'official-complex-library'", "'official-complex-rounded'")
        restored = restored.replace("referenceComplexPower !== 'library'",
                                    "referenceComplexPower !== 'fp32-rounded'")
        restored = restored.replace("referenceComplexPower: 'library'",
                                    "referenceComplexPower: 'fp32-rounded'")
        self.assertEqual(restored, baseline)
        with self.assertRaises(ValueError):
            build_power_probe('unsupported')

    def test_invalid_bank_fails_without_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, probe = root / 'receipt.json', root / 'probe.mjs'
            result = self.run_cli(root / 'missing.gz', output, probe)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Filterbank identity differs', result.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(probe.exists())

    def run_cli(self, capture, output, probe):
        return subprocess.run([sys.executable, str(TOOL), '--capture', str(capture),
            '--expected-capture-sha256', '0' * 64, '--expected-bank-sha256', '0' * 64,
            '--sample-id', 'test', '--output', str(output), '--probe-output', str(probe)],
            capture_output=True, text=True, timeout=20)

    def test_existing_outputs_are_preserved(self):
        for name in ('receipt.json', 'probe.mjs'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                existing = root / name
                existing.write_bytes(b'original')
                result = self.run_cli(root / 'missing.gz', root / 'receipt.json', root / 'probe.mjs')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Refusing existing output', result.stderr)
                self.assertEqual(existing.read_bytes(), b'original')
                other = root / ('probe.mjs' if name == 'receipt.json' else 'receipt.json')
                self.assertFalse(other.exists())

    def test_capture_alias_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / 'capture.gz'
            capture.write_bytes(b'original')
            result = self.run_cli(capture, capture, root / 'probe.mjs')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('protected input', result.stderr)
            self.assertEqual(capture.read_bytes(), b'original')
            self.assertFalse((root / 'probe.mjs').exists())


if __name__ == '__main__':
    unittest.main()
