"""Weight-free guards for exact capture PCM selection."""
import copy
import hashlib
import unittest
import numpy as np
from compare_js_feature_boundaries import captured_pcm


def sample():
    pcm = np.asarray([0, -0.25, 0.5], dtype='<f4')
    return {'stages': {'waveform': {'dtype': 'float32', 'shape': [1, 3],
                                  'data': pcm.tolist()}},
            'model_input': {'boundary': 'preprocessor_input_signal',
                'sample_rate': 16000, 'channels': 1,
                'dtype': 'float32', 'encoding': 'f32le', 'shape': [1, 3],
                'samples': 3, 'valid_samples': 3, 'right_padding_samples': 0,
                'sha256': hashlib.sha256(pcm.tobytes()).hexdigest()}}


class CapturedPcmTests(unittest.TestCase):
    def test_exact_values_and_bytes(self):
        result = captured_pcm(sample(), 16000)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.tobytes(), np.asarray([0, -0.25, 0.5], dtype='<f4').tobytes())

    def test_identity_mismatch_fails(self):
        for key, value in [('boundary', 'encoder_input'), ('boundary', None),
                           ('sample_rate', 8000), ('channels', 2),
                           ('samples', 2), ('right_padding_samples', 1),
                           ('sha256', '0' * 64), ('shape', [3]),
                           ('valid_samples', True)]:
            with self.subTest(key=key):
                source = sample()
                source['model_input'][key] = value
                with self.assertRaises(ValueError):
                    captured_pcm(source, 16000)

    def test_malformed_values_fail(self):
        for data in [[0, 0.1, 0.5], [0, float('nan'), 0.5],
                     [0, True, 0.5], [0, 0.5]]:
            with self.subTest(data=data):
                source = copy.deepcopy(sample())
                source['stages']['waveform']['data'] = data
                with self.assertRaises(ValueError):
                    captured_pcm(source, 16000)


if __name__ == '__main__':
    unittest.main()
