"""Synthetic result guards, not model parity."""
import json
import subprocess
import unittest

import numpy as np
import probe_filterbank_features as probe


class FeatureMatrixTests(unittest.TestCase):
    def results(self):
        return {name: {'raw': [0.0, 1.0], 'features': [0.0, 1.0],
                       'cascade_features': [1.0, 1.0], 'power_sha256': '1' * 64}
                for name in ('baseline', 'actual_buffer', 'triangle_rounding')}

    def test_equal_matrices_keep_separate_feature_failure(self):
        result = probe.evaluate(self.results(), np.ones((1, 1, 2), dtype=np.float32))
        self.assertFalse(result['feature_vs_official']['triangle_rounding']['allclose'])
        self.assertTrue(result['cascade_feature_vs_official']['triangle_rounding']['allclose'])
        self.assertTrue(result['candidate_vs_actual']['features']['allclose'])

    def test_changed_missing_or_invalid_power_identity_rejected(self):
        for value in (None, '2' * 64, 'z' * 64):
            results = self.results()
            results['actual_buffer']['power_sha256'] = value
            with self.assertRaisesRegex(ValueError, 'Power identity'):
                probe.evaluate(results, np.ones((1, 1, 2), dtype=np.float32))

    def test_missing_condition_or_malformed_tensor_rejected(self):
        for mode in ('condition', 'length', 'nan'):
            results = self.results()
            if mode == 'condition':
                del results['baseline']
            elif mode == 'length':
                results['baseline']['raw'].pop()
            else:
                results['baseline']['features'][0] = float('nan')
            with self.assertRaises(ValueError):
                probe.evaluate(results, np.ones((1, 1, 2), dtype=np.float32))

    def test_nonfinite_cascade_cannot_hide_behind_production_features(self):
        results = self.results()
        results['actual_buffer']['cascade_features'][0] = float('nan')
        with self.assertRaises(ValueError):
            probe.evaluate(results, np.ones((1, 1, 2), dtype=np.float32))

    def test_node_synthetic_same_matrix_exact_and_same_power(self):
        source = "import {createMelFilterbank} from './dist/audio/js-mel.js'; console.log(JSON.stringify(Array.from(createMelFilterbank(128))));"
        coefficients = json.loads(subprocess.run(['node', '--input-type=module', '-e', source],
            cwd=probe.ROOT, text=True, capture_output=True, check=True, timeout=10).stdout)
        payload = {'pcm': [float(np.float32(np.sin(index * .03))) for index in range(1600)],
                   'frames': 11, 'valid': 10, 'filterbank': coefficients}
        process = subprocess.run(['node', '--input-type=module', '-e', probe.NODE_PROBE],
            cwd=probe.ROOT, input=json.dumps(payload), text=True, capture_output=True,
            check=True, timeout=10)
        result = json.loads(process.stdout)
        self.assertEqual(result['baseline'], result['actual_buffer'])
        self.assertEqual(len({value['power_sha256'] for value in result.values()}), 1)


if __name__ == '__main__':
    unittest.main()
