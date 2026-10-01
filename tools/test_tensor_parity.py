import unittest

import numpy as np

from tensor_parity import compare_tensors


class TensorParityTests(unittest.TestCase):
    def test_small_error_reports_explicit_tolerances_and_metrics(self):
        result = compare_tensors(np.array([1, 2], dtype=np.float32),
                                 np.array([1, 2.00001], dtype=np.float32), atol=1e-5, rtol=1e-4)
        self.assertTrue(result["allclose"])
        self.assertGreater(result["max_abs_error"], 0)
        self.assertEqual(result["mismatch_count"], 0)

    def test_divergence_is_not_truncated_and_records_first_index(self):
        result = compare_tensors(np.array([1, 2], dtype=np.float32),
                                 np.array([1, 3], dtype=np.float32), atol=0, rtol=0)
        self.assertFalse(result["allclose"])
        self.assertEqual(result["first_divergent_index"], 1)
        self.assertEqual(result["mean_abs_error"], 0.5)

    def test_malformed_inputs_fail_instead_of_returning_a_pass(self):
        base = np.array([1], dtype=np.float32)
        for bad in [np.array([1, 2], dtype=np.float32), np.array([1], dtype=np.float64),
                    np.array([np.nan], dtype=np.float32), np.array([np.inf], dtype=np.float32)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compare_tensors(base, bad, atol=1e-5, rtol=1e-4)
        with self.assertRaises(ValueError):
            compare_tensors(base[:0], base[:0], atol=1e-5, rtol=1e-4)
        with self.assertRaises(ValueError):
            compare_tensors(base, base, atol=float("nan"), rtol=0)

    def test_integer_lengths_are_exact_even_with_large_float_tolerances(self):
        result = compare_tensors(np.array([1], dtype=np.int64), np.array([2], dtype=np.int64), atol=100, rtol=100)
        self.assertFalse(result["allclose"])

    def test_int64_identity_does_not_round_through_float64(self):
        result = compare_tensors(np.array([2**53], dtype=np.int64),
                                 np.array([2**53 + 1], dtype=np.int64), atol=0, rtol=0)
        self.assertFalse(result["allclose"])
        self.assertEqual(result["max_abs_error"], 1)


if __name__ == "__main__":
    unittest.main()
