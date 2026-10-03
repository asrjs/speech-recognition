import unittest

import numpy as np

from stage_attribution import (
    attribute_input_source,
    compare_stage_outputs,
    sweep_input_scale,
)


def gain_runner(gain: float):
    """A linear stage whose output is `gain * feed`."""

    def run(feeds):
        return {"out": (gain * np.asarray(feeds["x"], dtype=np.float32)).astype(np.float32)}

    return run


def scaled_runner():
    return gain_runner(1000.0)


class StageAttributionTests(unittest.TestCase):
    def test_implicates_only_the_input_that_reproduces_the_failure(self):
        reference = {"x": np.array([0.0], dtype=np.float32)}
        candidate = {"x": np.array([1e-4], dtype=np.float32)}
        result = attribute_input_source(
            reference, candidate, scaled_runner(), atol=1e-5, rtol=1e-4
        )
        self.assertEqual(result["status"], "ATTRIBUTED_TO_SINGLE_INPUT")
        self.assertTrue(result["resolved"])
        self.assertEqual(result["implicated_inputs"], ["x"])
        self.assertFalse(result["observed"]["allclose"])

    def test_candidate_value_is_not_rebuilt_by_subtraction(self):
        # reference + (candidate - reference) cancels to 0 in float32 for
        # these magnitudes, which would fabricate an UNRESOLVED result.
        reference = {"x": np.array([1e10], dtype=np.float32)}
        candidate = {"x": np.array([1.0], dtype=np.float32)}
        self.assertEqual(
            float(
                np.asarray(reference["x"])
                + (np.asarray(candidate["x"]) - np.asarray(reference["x"]))
            ),
            0.0,
            "precondition: the naive reconstruction cancels to zero",
        )
        result = attribute_input_source(
            reference, candidate, scaled_runner(), atol=1e-5, rtol=1e-4
        )
        self.assertEqual(result["status"], "ATTRIBUTED_TO_SINGLE_INPUT")
        self.assertEqual(result["implicated_inputs"], ["x"])

    def test_runs_the_component_once_per_replay_and_no_more(self):
        calls = []

        def counting(feeds):
            calls.append(float(np.asarray(feeds["a"], dtype=np.float32)[0]))
            return {"out": np.asarray(feeds["a"], dtype=np.float32).copy()}

        feeds = {
            "a": np.array([0.0], dtype=np.float32),
            "b": np.array([0.0], dtype=np.float32),
        }
        attribute_input_source(
            feeds,
            {k: v + 1.0 for k, v in feeds.items()},
            counting,
            atol=1e-5,
            rtol=1e-4,
        )
        # reference + candidate + one replay per input, exactly once each.
        self.assertEqual(len(calls), 2 + len(feeds))

    def test_supplied_oracle_baseline_keeps_the_execution_residual_visible(self):
        # Against the reference replay, scale 0 is exactly 0. Against a
        # supplied oracle it reports the runner's own distance from that
        # oracle. These are different contracts and must not be conflated.
        reference = {"x": np.array([0.0], dtype=np.float32)}
        recorded = {"x": np.array([3e-8], dtype=np.float32)}

        def oracle(feeds):
            return {"out": (np.asarray(feeds["x"], dtype=np.float32) * 1000.0).astype(np.float32)}

        against_replay = sweep_input_scale(
            reference, recorded, oracle, scales=[0.0], atol=1e-5, rtol=1e-4
        )
        self.assertEqual(against_replay["baseline"], "REFERENCE_INPUT_REPLAY")
        self.assertEqual(against_replay["rows"][0]["max_abs_error"]["out"], 0.0)

        supplied = sweep_input_scale(
            reference,
            recorded,
            oracle,
            scales=[0.0],
            atol=1e-5,
            rtol=1e-4,
            compare_against={"out": np.array([2.5e-5], dtype=np.float32)},
        )
        self.assertEqual(supplied["baseline"], "SUPPLIED_ORACLE")
        self.assertGreater(supplied["rows"][0]["max_abs_error"]["out"], 0.0)

    def test_supplied_oracle_must_cover_every_output(self):
        base = {"x": np.array([0.0], dtype=np.float32)}
        with self.assertRaises(ValueError):
            sweep_input_scale(
                base,
                base,
                gain_runner(1.0),
                scales=[1.0],
                atol=1e-5,
                rtol=1e-4,
                compare_against={"other": np.array([0.0], dtype=np.float32)},
            )

    def test_an_unrelated_input_is_not_implicated(self):
        reference = {"x": np.array([0.0], dtype=np.float32)}
        candidate = {"x": np.array([1e-9], dtype=np.float32)}
        result = attribute_input_source(
            reference, candidate, scaled_runner(), atol=1e-5, rtol=1e-4
        )
        self.assertEqual(result["status"], "BOUNDARY_PASSING")
        self.assertEqual(result["implicated_inputs"], [])
        self.assertEqual(result["inputs_tested"], ["x"])

    def test_unreproduced_failure_reports_unresolved_instead_of_guessing(self):
        # Each input alone stays under the gate, but the sum does not.
        def summing(feeds):
            total = np.asarray(feeds["a"], dtype=np.float32) + np.asarray(
                feeds["b"], dtype=np.float32
            )
            return {"out": total.astype(np.float32)}

        reference = {
            "a": np.array([0.0], dtype=np.float32),
            "b": np.array([0.0], dtype=np.float32),
        }
        candidate = {
            "a": np.array([6e-6], dtype=np.float32),
            "b": np.array([6e-6], dtype=np.float32),
        }
        result = attribute_input_source(
            reference, candidate, summing, atol=1e-5, rtol=1e-4
        )
        self.assertEqual(result["status"], "ATTRIBUTION_UNRESOLVED")
        self.assertFalse(result["resolved"])
        self.assertEqual(result["implicated_inputs"], [])
        self.assertEqual(result["inputs_tested"], ["a", "b"])

    def test_interaction_is_not_misattributed_to_one_input(self):
        # f(x, y) = x + 100*x*y. With reference (0, 0) and candidate
        # (2, 0.5) the observed result is 2 + 100 = 102. Replaying x alone
        # gives 2, which fails the gate but explains almost none of 102.
        def interacting(feeds):
            x = np.asarray(feeds["x"], dtype=np.float32)
            y = np.asarray(feeds["y"], dtype=np.float32)
            return {"out": (x + 100.0 * x * y).astype(np.float32)}

        reference = {
            "x": np.array([0.0], dtype=np.float32),
            "y": np.array([0.0], dtype=np.float32),
        }
        candidate = {
            "x": np.array([2.0], dtype=np.float32),
            "y": np.array([0.5], dtype=np.float32),
        }
        result = attribute_input_source(
            reference, candidate, interacting, atol=1e-5, rtol=1e-4
        )
        self.assertEqual(result["status"], "ATTRIBUTION_UNRESOLVED")
        self.assertFalse(result["resolved"])
        self.assertEqual(result["implicated_inputs"], [])
        x_row = result["per_input"]["x"]
        self.assertTrue(x_row["fails_gate_against_reference"])
        self.assertFalse(x_row["reproduces_observed_output"])

    def test_missing_or_mismatched_feeds_fail_instead_of_skipping(self):
        runner = gain_runner(1.0)
        base = {"x": np.array([0.0], dtype=np.float32)}
        with self.assertRaises(ValueError):
            attribute_input_source(
                base,
                {"y": np.array([0.0], dtype=np.float32)},
                runner,
                atol=1e-5,
                rtol=1e-4,
            )
        with self.assertRaises(ValueError):
            attribute_input_source(
                base,
                {"x": np.array([0.0, 1.0], dtype=np.float32)},
                runner,
                atol=1e-5,
                rtol=1e-4,
            )
        with self.assertRaises(ValueError):
            attribute_input_source(
                base,
                {"x": np.array([0.0], dtype=np.float64)},
                runner,
                atol=1e-5,
                rtol=1e-4,
            )
        with self.assertRaises(ValueError):
            attribute_input_source(
                base, base, runner, atol=1e-5, rtol=1e-4, output_names=["z"]
            )

    def test_non_finite_inputs_and_invalid_tolerances_are_rejected(self):
        base = {"x": np.array([0.0], dtype=np.float32)}
        bad = {"x": np.array([np.nan], dtype=np.float32)}
        with self.assertRaises(ValueError):
            attribute_input_source(base, bad, gain_runner(1.0), atol=1e-5, rtol=1e-4)
        for atol, rtol in [(-1.0, 1e-4), (1e-5, -1.0), (float("nan"), 1e-4)]:
            with self.assertRaises(ValueError):
                attribute_input_source(
                    base, base, gain_runner(1.0), atol=atol, rtol=rtol
                )

    def test_output_comparison_never_omits_an_output(self):
        left = {
            "a": np.array([0.0], dtype=np.float32),
            "b": np.array([0.0], dtype=np.float32),
        }
        right = {"a": np.array([0.0], dtype=np.float32)}
        with self.assertRaises(ValueError):
            compare_stage_outputs(left, right, atol=1e-5, rtol=1e-4)
        result = compare_stage_outputs(left, left, atol=1e-5, rtol=1e-4)
        self.assertEqual(sorted(result["outputs"]), ["a", "b"])
        self.assertTrue(result["allclose"])

    def test_rtol_term_is_applied_so_a_large_reference_passes(self):
        # A large-magnitude coordinate passes on rtol alone. A control that
        # divided a single measured gain by atol alone would misjudge this.
        left = {"out": np.array([-51.0289], dtype=np.float32)}
        right = {"out": np.array([-51.0277], dtype=np.float32)}
        self.assertTrue(compare_stage_outputs(left, right, atol=1e-5, rtol=1e-4)["allclose"])
        self.assertFalse(compare_stage_outputs(left, right, atol=1e-7, rtol=0.0)["allclose"])


class InputScaleSweepTests(unittest.TestCase):
    def test_reports_largest_passing_scale_as_a_directional_bracket(self):
        reference = {"x": np.array([0.0], dtype=np.float32)}
        recorded = {"x": np.array([3e-8], dtype=np.float32)}
        result = sweep_input_scale(
            reference,
            recorded,
            scaled_runner(),
            scales=[0.0, 0.25, 0.5, 1.0],
            atol=1e-5,
            rtol=1e-4,
        )
        self.assertEqual([row["scale"] for row in result["rows"]], [0.0, 0.25, 0.5, 1.0])
        self.assertTrue(result["rows"][1]["allclose"])
        self.assertFalse(result["rows"][2]["allclose"])
        self.assertEqual(result["largest_passing_scale"], 0.25)
        self.assertIn("not a per-frame budget", result["limitations"].lower())

    def test_empty_scales_and_bad_output_names_are_rejected(self):
        base = {"x": np.array([0.0], dtype=np.float32)}
        with self.assertRaises(ValueError):
            sweep_input_scale(
                base, base, scaled_runner(), scales=[], atol=1e-5, rtol=1e-4
            )
        with self.assertRaises(ValueError):
            sweep_input_scale(
                base,
                base,
                scaled_runner(),
                scales=[1.0],
                atol=1e-5,
                rtol=1e-4,
                output_names=["missing"],
            )


if __name__ == "__main__":
    unittest.main()