"""Light DFT CLI rejection controls, without capture/model execution."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import onnxruntime as ort
from probe_light_dft_features import (build_power_probe, checked_coordinate, build_dft_graph,
                                     DFT_TYPES, pocketfft_control)
from replay_packed_fft import build_probe, digest

TOOL = Path(__file__).with_name('probe_light_dft_features.py')


class LightDftGuardTests(unittest.TestCase):
    def test_pocketfft_native_impulse_and_silence(self):
        frames = np.zeros((1, 2, 512), dtype=np.float32)
        frames[0, 0, 0] = 1
        original = frames.copy()
        values, metadata = pocketfft_control(frames)
        self.assertEqual(values.shape, (1, 257, 2))
        self.assertEqual(values.dtype, np.complex64)
        np.testing.assert_array_equal(values[0, :, 0], np.ones(257, dtype=np.complex64))
        np.testing.assert_array_equal(values[0, :, 1], np.zeros(257, dtype=np.complex64))
        np.testing.assert_array_equal(frames, original)
        self.assertEqual(metadata['workers'], 1)
        self.assertEqual(len(metadata['sources_sha256']), 3)
        for path, expected in metadata['sources_sha256'].items():
            self.assertEqual(digest(Path(path).read_bytes()), expected)

    def test_pocketfft_rejects_invalid_input(self):
        for frames in (np.zeros((1, 1, 512), dtype=np.float64),
                       np.zeros((1, 1, 511), dtype=np.float32),
                       np.zeros((1, 0, 512), dtype=np.float32),
                       np.full((1, 1, 512), np.nan, dtype=np.float32)):
            with self.subTest(shape=frames.shape, dtype=frames.dtype), self.assertRaises(ValueError):
                pocketfft_control(frames)

    def test_both_dft_precisions_run_real_impulse_and_silence(self):
        self.assertEqual(digest(build_dft_graph('fp32')),
                         '6d657e99a8f911a6caf20eab3547294b0003d17cb1b868f18d722d55aaede898')
        for precision, (dtype, _) in DFT_TYPES.items():
            with self.subTest(precision=precision):
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                session = ort.InferenceSession(build_dft_graph(precision), options,
                                               providers=['CPUExecutionProvider'])
                frames = np.zeros((1, 2, 512, 1), dtype=dtype)
                frames[0, 0, 0, 0] = 1
                values = session.run(['spectrum'], {'frames': frames})[0]
                self.assertEqual(values.shape, (1, 2, 257, 2))
                self.assertEqual(values.dtype, dtype)
                np.testing.assert_array_equal(values[0, 0, :, 0], np.ones(257, dtype=dtype))
                np.testing.assert_array_equal(values[0, 0, :, 1], np.zeros(257, dtype=dtype))
                np.testing.assert_array_equal(values[0, 1], np.zeros((257, 2), dtype=dtype))
        with self.assertRaises(ValueError):
            build_dft_graph('fp16')

    def test_coordinate_selects_only_valid_prefix_without_mutation(self):
        self.assertIsNone(checked_coordinate(None, 3, 7))
        source = [2, 6]
        selected = checked_coordinate(source, 3, 7)
        self.assertEqual(selected, [2, 6])
        self.assertIsNot(selected, source)
        for invalid in ([3, 0], [0, 7], [-1, 0], [0, -1], [True, 0], [0.0, 1], [0], [0, 1, 2]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                checked_coordinate(invalid, 3, 7)

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

    def test_pocketfft_fp64_rejected_before_capture_or_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, probe = root / 'receipt.json', root / 'probe.mjs'
            result = self.run_cli(root / 'missing.gz', output, probe,
                                  ['--fft-engine', 'scipy-pocketfft', '--dft-precision', 'fp64'])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('PocketFFT control supports fp32 only', result.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(probe.exists())

    def test_torch_fp64_rejected_before_capture_or_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, probe = root / 'receipt.json', root / 'probe.mjs'
            result = self.run_cli(root / 'missing.gz', output, probe,
                                  ['--fft-engine', 'torch-rfft', '--dft-precision', 'fp64'])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Torch oracle reproduction supports fp32 only', result.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(probe.exists())

    def run_cli(self, capture, output, probe, extra=()):
        return subprocess.run([sys.executable, str(TOOL), '--capture', str(capture),
            '--expected-capture-sha256', '0' * 64, '--expected-bank-sha256', '0' * 64,
            '--sample-id', 'test', '--output', str(output), '--probe-output', str(probe), *extra],
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
