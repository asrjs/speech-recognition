"""Weight-free tests for actual-model STFT reproduction gates."""
import copy
import unittest
import numpy as np
from compare_js_feature_boundaries import validate_actual_stft as validate


def validate_actual_stft(*args, **kwargs):
    return validate(*args, capture_marker=True, **kwargs)


class ActualStftTests(unittest.TestCase):
    def setUp(self):
        self.arrays = {"input": np.ones((1, 4), dtype=np.float32),
                       "window": np.ones(2, dtype=np.float32),
                       "real": np.ones((1, 2, 3), dtype=np.float32),
                       "imag": np.zeros((1, 2, 3), dtype=np.float32)}
        self.contract = {"n_fft": 2, "hop_length": 1, "win_length": 2, "exact_pad": False}
        self.sample = {"feature_stft_contract": dict(self.contract), "stages": {
            "feature_stft/" + name: {"shape": list(value.shape), "dtype": "float32",
                                     "data": value.ravel().tolist()}
            for name, value in self.arrays.items()}}

    def test_exact_capture_passes(self):
        result = validate_actual_stft(self.sample, self.arrays, self.contract, required=True)
        self.assertEqual(result["status"], "PASS_ACTUAL_STFT_REPRODUCTION")
        self.assertTrue(all(row["bits_exact"] for row in result["boundaries"].values()))

    def test_old_capture_is_unverified_not_pass(self):
        self.assertEqual(validate({"stages": {}}, self.arrays, self.contract)["status"], "UNTESTED")
        with self.assertRaises(ValueError):
            validate_actual_stft({"stages": {}}, self.arrays, self.contract, required=True)

    def test_partial_capture_fails(self):
        del self.sample["stages"]["feature_stft/imag"]
        with self.assertRaises(ValueError):
            validate_actual_stft(self.sample, self.arrays, self.contract)

    def test_missing_false_or_truthy_marker_cannot_certify_actual_capture(self):
        for marker in (None, False, 1, "true"):
            with self.subTest(marker=marker):
                with self.assertRaises(ValueError):
                    validate(self.sample, self.arrays, self.contract, capture_marker=marker)

    def test_changed_boundary_fails_even_inside_old_tolerance(self):
        for name in self.arrays:
            with self.subTest(name=name):
                arrays = copy.deepcopy(self.arrays)
                arrays[name].flat[0] = np.nextafter(arrays[name].flat[0], np.float32(2))
                with self.assertRaises(ValueError):
                    validate_actual_stft(self.sample, arrays, self.contract)

    def test_wrong_contract_dtype_shape_count_and_nonfinite_fail(self):
        for mode in ("contract", "dtype", "shape", "count", "nan", "empty"):
            with self.subTest(mode=mode):
                sample = copy.deepcopy(self.sample)
                stage = sample["stages"]["feature_stft/input"]
                if mode == "contract": sample["feature_stft_contract"]["hop_length"] = 2
                if mode == "dtype": stage["dtype"] = "float64"
                if mode == "shape": stage["shape"] = [4]
                if mode == "count": stage["data"].pop()
                if mode == "nan": stage["data"][0] = float("nan")
                if mode == "empty": stage["shape"], stage["data"] = [1, 0], []
                with self.assertRaises(ValueError):
                    validate_actual_stft(sample, self.arrays, self.contract)

    def test_signed_zero_is_not_bit_exact(self):
        self.arrays["imag"].flat[0] = np.float32(-0.0)
        with self.assertRaises(ValueError):
            validate_actual_stft(self.sample, self.arrays, self.contract)

    def test_wrong_reproduced_dtype_and_nonfinite_fail(self):
        for values in (self.arrays["input"].astype(np.float64),
                       np.full((1, 4), np.nan, dtype=np.float32)):
            with self.subTest(dtype=values.dtype):
                arrays = dict(self.arrays, input=values)
                with self.assertRaises(ValueError):
                    validate_actual_stft(self.sample, arrays, self.contract)


if __name__ == "__main__":
    unittest.main()
