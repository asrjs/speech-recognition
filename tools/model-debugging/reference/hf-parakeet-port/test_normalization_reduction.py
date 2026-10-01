"""Weight-free contracts for the diagnostic-only float32 reduction candidate."""
import unittest
import numpy as np
from normalization_reduction import cascade_sum_rows, normalize_candidate


class ReductionTests(unittest.TestCase):
    def test_exact_integer_sums_across_vector_and_cascade_boundaries(self):
        for width in (8, 16):
            for size in (1, 2, 15, 16, 17, 63, 64, 65, 100, 101, 128, 129, 257, 1101):
                with self.subTest(width=width, size=size):
                    values = np.ones((3, size), dtype=np.float32)
                    np.testing.assert_array_equal(cascade_sum_rows(values, width), [size]*3)

    def test_masked_normalization_does_not_mutate_inputs(self):
        values = np.asarray([[1, 2, 3, 999]], dtype=np.float32)
        before = values.tobytes()
        result, mean, std = normalize_candidate(values, 3, 16)
        self.assertEqual(values.tobytes(), before)
        self.assertEqual(float(mean[0]), 2)
        self.assertEqual(float(std[0]), float(np.float32(1)+np.float32(1e-5)))
        self.assertEqual(float(result[0, 3]), 0)
        np.testing.assert_array_equal(result[0, :3], np.float32([-1, 0, 1])/std[0])

    def test_single_valid_frame_nan_variance_becomes_zero_before_guard(self):
        result, mean, std = normalize_candidate(np.asarray([[3, 999]], dtype=np.float32), 1, 16)
        np.testing.assert_array_equal(result, [[0, 0]])
        self.assertEqual(float(mean[0]), 3)
        self.assertEqual(float(std[0]), float(np.float32(1e-5)))

    def test_measured_constant_masked_reference_discriminates_widths(self):
        # Installed Torch 2.6.0 CPU observation, not a universal ISA contract.
        raw = np.full((128, 101), -16.63553237915039, dtype=np.float32)
        result, mean, std = normalize_candidate(raw, 100, 8)
        np.testing.assert_array_equal(mean, np.full(128, -16.635530471801758, dtype=np.float32))
        np.testing.assert_array_equal(std, np.full(128, 1.1916957191715483e-5, dtype=np.float32))
        np.testing.assert_array_equal(result[:, :100], np.full((128, 100), -0.1600533276796341, dtype=np.float32))
        np.testing.assert_array_equal(result[:, 100], np.zeros(128, dtype=np.float32))
        other, _, _ = normalize_candidate(raw, 100, 16)
        self.assertFalse(np.array_equal(result, other))

    def test_invalid_contracts_fail_closed(self):
        for values in (np.ones(3, dtype=np.float32), np.ones((1, 0), dtype=np.float32),
                np.ones((1, 3), dtype=np.float64), np.asarray([[np.nan]], dtype=np.float32)):
            with self.subTest(values=values):
                with self.assertRaises(ValueError): cascade_sum_rows(values, 16)
        for width in (0, 7, True):
            with self.assertRaises(ValueError): cascade_sum_rows(np.ones((1, 3), dtype=np.float32), width)
        for length in (0, 4, True, 1.5):
            with self.assertRaises(ValueError): normalize_candidate(np.ones((1, 3), dtype=np.float32), length, 16)


if __name__ == '__main__':
    unittest.main()
