"""Small weight-free guards; pinned identity is from the actual librosa receipt."""
import unittest
import numpy as np
from diagnostic_mel_bank import slaney_bank


class DiagnosticMelBankTests(unittest.TestCase):
    def test_recorded_reference_identity(self):
        bank = slaney_bank(sample_rate=16000, n_fft=512, n_mels=128,
            expected_sha256='f5608e00763a20a191022fbb00a43346db5f1ec06828a97a6f4157f7d02b76c8')
        self.assertEqual(bank.shape, (128, 257))
        self.assertEqual(bank.dtype, np.float32)
        self.assertTrue(np.isfinite(bank).all())

    def test_wrong_reference_identity_fails(self):
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            slaney_bank(sample_rate=16000, n_fft=512, n_mels=128, expected_sha256='0' * 64)

    def test_invalid_recipe_fails(self):
        for override in [{'sample_rate': True}, {'n_fft': 0}, {'n_mels': 1.5},
                         {'fmin': -1}, {'fmin': False}, {'fmax': True},
                         {'fmin': [0]}, {'fmax': '8000'},
                         {'fmax': float('nan')}, {'fmax': 9000},
                         {'fmin': 1000, 'fmax': 1000}, {'expected_sha256': 'A' * 64}]:
            with self.subTest(override=override):
                recipe = dict(sample_rate=16000, n_fft=512, n_mels=128)
                recipe.update(override)
                with self.assertRaises(ValueError):
                    slaney_bank(**recipe)

    def test_independent_results(self):
        recipe = dict(sample_rate=16000, n_fft=512, n_mels=128)
        first = slaney_bank(**recipe)
        second = slaney_bank(**recipe)
        self.assertEqual(first.tobytes(), second.tobytes())
        first.fill(0)
        self.assertGreater(float(second.max()), 0)


if __name__ == '__main__':
    unittest.main()
