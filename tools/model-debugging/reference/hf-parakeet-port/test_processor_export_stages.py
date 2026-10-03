"""Stage comparator guards, independent of optional inference engines."""
import unittest
import numpy as np
from processor_export_stages import NAMES, compare_stages


class StageComparatorTests(unittest.TestCase):
    def values(self):
        return [np.array([2], dtype=np.int64 if name == 'features_lens' else np.float32)
                for name in NAMES]

    def test_identical_stages_are_exact(self):
        metrics = compare_stages(self.values(), self.values())
        self.assertEqual(set(metrics), set(NAMES))
        self.assertTrue(all(value['allclose'] and value['bytes_exact'] for value in metrics.values()))

    def test_missing_stage_fails(self):
        with self.assertRaisesRegex(ValueError, 'count differs'):
            compare_stages(self.values(), self.values()[:-1])

    def test_dtype_mismatch_fails(self):
        candidate = self.values()
        candidate[2] = candidate[2].astype(np.float64)
        with self.assertRaisesRegex(ValueError, 'shape/dtype mismatch'):
            compare_stages(self.values(), candidate)

    def test_nonfinite_stage_fails(self):
        candidate = self.values()
        candidate[3][0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            compare_stages(self.values(), candidate)

    def test_integer_lengths_require_exact_equality(self):
        reference, candidate = self.values(), self.values()
        reference[1][0] = 1000000
        candidate[1][0] = 1000001
        result = compare_stages(reference, candidate)['features_lens']
        self.assertFalse(result['allclose'])
        self.assertEqual(result['mismatch_count'], 1)


if __name__ == '__main__':
    unittest.main()
