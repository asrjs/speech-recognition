"""Round-trip tests for the `thrush-ternary-v2` decoder.

The decoder has to be right before any weight is fetched, because a silent
misread would look like a bad model rather than a bad reader.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ternary import (  # noqa: E402
    ELEMENTS_PER_BYTE,
    TernaryLayout,
    dequantize,
    decode_matrix,
    pack_rows,
    unpack_rows,
)


class PackingTests(unittest.TestCase):
    def test_row_bytes_follows_the_manifest(self):
        self.assertEqual(TernaryLayout(in_features=1024, out_features=4, group_size=128).row_bytes, 205)
        self.assertEqual(TernaryLayout(in_features=1, out_features=4, group_size=128).row_bytes, 1)
        self.assertEqual(TernaryLayout(in_features=6, out_features=4, group_size=128).row_bytes, 2)

    def test_group_count_follows_the_manifest(self):
        self.assertEqual(TernaryLayout(in_features=1024, out_features=4, group_size=128).groups, 8)
        self.assertEqual(TernaryLayout(in_features=100, out_features=4, group_size=128).groups, 1)

    def test_rejects_a_block_of_the_wrong_length(self):
        with self.assertRaises(ValueError):
            unpack_rows(b"\x00" * 10, TernaryLayout(in_features=1024, out_features=4, group_size=128))


class RoundTripTests(unittest.TestCase):
    def _random_codes(self, out_features: int, in_features: int, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).integers(0, 3, size=(out_features, in_features), dtype=np.uint8)

    def test_pack_unpack_is_exact(self):
        for in_features in (1, 5, 6, 128, 130, 1024):
            codes = self._random_codes(3, in_features, in_features)
            layout = TernaryLayout(in_features=in_features, out_features=3, group_size=128)
            recovered = unpack_rows(pack_rows(codes), layout)
            np.testing.assert_array_equal(recovered, codes)

    def test_least_significant_digit_is_the_first_element(self):
        # Digits 1,0,2,0,1 with element 0 least significant: 1 + 0*3 + 2*9 + 0*27 + 1*81.
        packed = bytes([1 + 0 * 3 + 2 * 9 + 0 * 27 + 1 * 81])
        codes = unpack_rows(packed, TernaryLayout(in_features=5, out_features=1, group_size=128))
        np.testing.assert_array_equal(codes, np.array([[1, 0, 2, 0, 1]], dtype=np.uint8))

    def test_padding_digits_are_discarded(self):
        # Five base-3 digits of 1: 1 + 3 + 9 + 27 + 81.
        packed = bytes([1 + 3 + 9 + 27 + 81])
        codes = unpack_rows(packed, TernaryLayout(in_features=3, out_features=1, group_size=128))
        np.testing.assert_array_equal(codes, np.array([[1, 1, 1]], dtype=np.uint8))

    def test_rejects_codes_outside_the_alphabet(self):
        with self.assertRaises(ValueError):
            pack_rows(np.array([[0, 1, 3]], dtype=np.uint8))


class WeightRuleTests(unittest.TestCase):
    def test_weights_are_scale_times_signed_unit(self):
        codes = np.array([[2, 0, 1, 2]], dtype=np.uint8)
        scales = np.array([[0.25]], dtype=np.float32)
        weights = dequantize(codes, scales, group_size=128)
        np.testing.assert_allclose(weights, np.array([[0.25, -0.25, 0.0, 0.25]], dtype=np.float32))

    def test_group_column_selects_the_scale(self):
        codes = np.array([[2] * 8], dtype=np.uint8)
        scales = np.array([[1.0, 4.0]], dtype=np.float32)
        weights = dequantize(codes, scales, group_size=4)
        np.testing.assert_allclose(
            weights, np.array([[1, 1, 1, 1, 4, 4, 4, 4]], dtype=np.float32)
        )

    def test_rejects_mismatched_scales(self):
        with self.assertRaises(ValueError):
            dequantize(
                np.zeros((2, 128), dtype=np.uint8), np.zeros((2, 3), dtype=np.float32), group_size=128
            )

    def test_decode_matrix_matches_the_two_steps(self):
        codes = np.random.default_rng(7).integers(0, 3, (5, 130), dtype=np.uint8)
        layout = TernaryLayout(in_features=130, out_features=5, group_size=128)
        scales = np.random.default_rng(8).random((5, 2)).astype(np.float32) * 0.1
        decoded = decode_matrix(pack_rows(codes), scales, layout)
        np.testing.assert_allclose(decoded, dequantize(codes, scales, 128), rtol=0, atol=0)
        self.assertEqual(decoded.dtype, np.float32)
        self.assertEqual(decoded.shape, (5, 130))


class ManifestTests(unittest.TestCase):
    def test_matches_the_published_manifest(self):
        import json

        manifest = json.loads(
            Path(".cache/parakeet-candidates/parakeet-redux-ternary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["format"], "thrush-ternary-v2")
        packing = manifest["packing"]
        self.assertEqual(packing["base"], 3)
        self.assertEqual(packing["elements_per_byte"], ELEMENTS_PER_BYTE)
        self.assertEqual(packing["code_offset"], 1)
        self.assertEqual(manifest["quant"]["group_size"], 128)
        self.assertEqual(packing["row_bytes"], "ceil(in/5)")


if __name__ == "__main__":
    unittest.main()