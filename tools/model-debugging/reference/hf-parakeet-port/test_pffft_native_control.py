"""Optional real-binary sanity plus always-on fail-closed input controls."""
from hashlib import sha256
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from pffft_native_control import pffft_control


class PffftControlTests(unittest.TestCase):
    def test_invalid_frames_rejected_before_loading(self):
        for frames in (np.zeros((1, 1, 511), np.float32),
                       np.zeros((1, 1, 512), np.float64),
                       np.zeros((1, 0, 512), np.float32),
                       np.full((1, 1, 512), np.inf, np.float32)):
            with self.subTest(shape=frames.shape), self.assertRaises(ValueError):
                pffft_control(frames, 'missing.dll', '0' * 64)

    def test_identity_mismatch_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / 'not-a-dll'
            library.write_bytes(b'not native code')
            with self.assertRaisesRegex(ValueError, 'binary identity differs'):
                pffft_control(np.zeros((1, 1, 512), np.float32), library, '0' * 64)

    def test_requested_real_binary(self):
        library = os.environ.get('ASR_TEST_PFFFT_LIBRARY')
        if library is None:
            self.skipTest('Native PFFFT binary not requested')
        expected = os.environ.get('ASR_TEST_PFFFT_SHA256')
        self.assertIsNotNone(expected, 'Requested PFFFT test requires expected SHA256')
        frames = np.zeros((1, 3, 512), np.float32)
        frames[0, 0, 0] = 1
        frames[0, 2, 1] = 1
        original = frames.copy()
        candidate, metadata = pffft_control(frames, library, expected)
        np.testing.assert_array_equal(candidate[0, :, 0], np.ones(257, np.complex64))
        np.testing.assert_array_equal(candidate[0, :, 1], np.zeros(257, np.complex64))
        np.testing.assert_allclose(candidate[0, :, 2], np.fft.rfft(frames[0, 2]),
                                   atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(frames, original)
        self.assertEqual(candidate.dtype, np.complex64)
        self.assertEqual(candidate.shape, (1, 257, 3))
        self.assertEqual(metadata['binary_sha256'], sha256(Path(library).read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
