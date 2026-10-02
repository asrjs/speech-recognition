"""Weight-free negative controls for exact complex oracle-injection masks."""
import unittest
import numpy as np
from complex_intervention import validate_complex_intervention as validate


class ComplexInterventionTests(unittest.TestCase):
    def setUp(self):
        self.candidate = np.ones((1, 3, 5), dtype=np.complex64)
        self.oracle = self.candidate.copy()
        self.oracle[0, 1, 2] = 2 + 3j
        self.patched = self.candidate.copy()
        self.patched[0, 1, 2] = self.oracle[0, 1, 2]

    def check(self, **kwargs):
        args = dict(candidate=self.candidate, oracle=self.oracle,
                    patched=self.patched, coordinates=[(1, 2)], valid_length=4)
        args.update(kwargs)
        return validate(**args)

    def test_exact_mask_passes_without_mutating_inputs(self):
        before = [value.tobytes() for value in (self.candidate, self.oracle, self.patched)]
        result = self.check()
        self.assertEqual(result["status"], "DIAGNOSTIC_NOT_PORT_ACCEPTANCE")
        self.assertTrue(result["outside_mask_bits_unchanged"])
        self.assertEqual(before, [value.tobytes() for value in
                                (self.candidate, self.oracle, self.patched)])

    def test_noncontiguous_layout_has_identical_receipt(self):
        values = [value.transpose(0, 2, 1).copy().transpose(0, 2, 1)
                  for value in (self.candidate, self.oracle, self.patched)]
        self.assertFalse(values[0].flags.c_contiguous)
        self.assertEqual(self.check(), self.check(candidate=values[0], oracle=values[1], patched=values[2]))

    def test_unselected_component_and_padding_changes_fail(self):
        for component in ("real", "imag"):
            for frame in (0, 4):
                with self.subTest(component=component, frame=frame):
                    patched = self.patched.copy()
                    getattr(patched, component)[0, 0, frame] += 1
                    with self.assertRaisesRegex(ValueError, "outside"):
                        self.check(patched=patched)

    def test_signed_zero_change_fails_inside_and_outside_mask(self):
        for coordinate in ((0, 0, 0), (0, 1, 2)):
            with self.subTest(coordinate=coordinate):
                self.oracle.imag[coordinate] = 0
                self.patched.imag[coordinate] = np.float32(-0.0)
                with self.assertRaises(ValueError):
                    self.check()
                self.patched.imag[coordinate] = self.candidate.imag[coordinate]

    def test_selected_component_must_match_exact_not_allclose(self):
        self.patched.real[0, 1, 2] = np.nextafter(np.float32(2), np.float32(3))
        with self.assertRaisesRegex(ValueError, "oracle"):
            self.check()

    def test_invalid_coordinates_and_valid_lengths_fail(self):
        for coordinates in ([], [(1, 4)], [(3, 0)], [(-1, 0)], [(1, 2), (1, 2)],
                            [(True, 2)], [(1, 2.0)], [(1,)], "1,2"):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                self.check(coordinates=coordinates)
        for length in (0, 6, True, 4.0):
            with self.subTest(length=length), self.assertRaises(ValueError):
                self.check(valid_length=length)

    def test_invalid_shape_dtype_and_nonfinite_fail(self):
        for values in (self.patched.astype(np.complex128), self.patched[0],
                       self.patched[:, :, :4], np.empty((1, 3, 0), np.complex64),
                       np.full(self.patched.shape, complex(np.nan, 0), np.complex64)):
            with self.subTest(shape=values.shape, dtype=values.dtype), self.assertRaises(ValueError):
                self.check(patched=values)


if __name__ == "__main__":
    unittest.main()
