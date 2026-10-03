"""Synthetic evidence guards; not real model parity."""
import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import probe_captured_filterbank as probe


class FilterbankTests(unittest.TestCase):
    def sample(self):
        values = np.ones((1, 128, 257), dtype=np.float32)
        return {'stages': {'feature_stft/filterbank': {
            'dtype': 'float32', 'shape': list(values.shape), 'data': values.ravel().tolist()}},
            'feature_stft_contract': {'filterbank_boundary': 'model.preprocessor.featurizer.fb',
                'filterbank_f32le_sha256': hashlib.sha256(values.astype('<f4').tobytes()).hexdigest()}}

    def test_actual_buffer_identity_passes(self):
        self.assertEqual(probe.captured_filterbank(self.sample()).shape, (1, 128, 257))

    def test_hash_or_boundary_substitution_rejected(self):
        for key, value in [('filterbank_boundary', 'generated-librosa'),
                           ('filterbank_f32le_sha256', '0' * 64)]:
            sample = self.sample()
            sample['feature_stft_contract'][key] = value
            with self.assertRaises(ValueError):
                probe.captured_filterbank(sample)

    def test_missing_wrong_dtype_shape_nonfinite_rejected(self):
        for mode in ('missing', 'dtype', 'shape', 'nan'):
            sample = self.sample()
            tensor = sample['stages']['feature_stft/filterbank']
            if mode == 'missing':
                del sample['stages']['feature_stft/filterbank']
            elif mode == 'dtype':
                tensor['dtype'] = 'float64'
            elif mode == 'shape':
                tensor['shape'] = [128, 257]
            else:
                tensor['data'][0] = float('nan')
            with self.assertRaises((KeyError, ValueError)):
                probe.captured_filterbank(sample)

    def test_existing_receipt_preserved_before_reading_or_node(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'receipt.json'
            output.write_text('preserve', encoding='utf-8')
            arguments = ['probe', '--capture', 'missing.gz', '--capture-sha256', '1' * 64,
                         '--weight-sha256', '2' * 64, '--sample-id', 'fixture', '--output', str(output)]
            with patch('sys.argv', arguments), patch.object(probe, 'digest') as hashing, \
                    patch.object(probe.subprocess, 'run') as node:
                with self.assertRaises(FileExistsError):
                    probe.main()
                hashing.assert_not_called()
                node.assert_not_called()
            self.assertEqual(output.read_text(encoding='utf-8'), 'preserve')


if __name__ == '__main__':
    unittest.main()
