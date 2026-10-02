import copy
import unittest
import numpy as np

from browser_head_replay import tensor
from live_head_parity import compare_live_head
from validate_stage_replay import log_softmax


def fixture():
    def value(data, dtype="float32"):
        return tensor(np.asarray(data, dtype=dtype).reshape(1, 1, -1))
    feeds = {"encoder_outputs": value([0., .5]), "targets": value([1], "int32"),
             "target_length": value([1], "int32"), "input_states_1": value([0., 0.]),
             "input_states_2": value([0., 0.])}
    outputs = {"outputs": value([3., 1., 2., 0., 2.]), "output_states_1": value([.5, 1.]),
               "output_states_2": value([1., 2.]), "prednet_lengths": value([1], "int32")}
    expected = {**copy.deepcopy(outputs), "normalized_scores": tensor(log_softmax(np.asarray(
                outputs["outputs"]["data"], dtype=np.float32).reshape(1, 1, -1)))}
    state = {"h": outputs["output_states_1"], "c": outputs["output_states_2"]}
    run = {"tokenIds": [0], "frameIndices": [0], "tdtSteps": [1], "text": "x", "decoderState": state,
           "observerRestored": True, "headCalls": [], "traceEnabled": False}
    runs = [copy.deepcopy(run), copy.deepcopy(run)]
    runs[0].update(traceEnabled=True, headCalls=[{"index": 0, "feeds": feeds, "outputs": outputs}])
    return {"runs": runs}, {"blank_id": 2, "steps": [{"binding": {"frame": 0}, "feeds": copy.deepcopy(feeds), "expected": expected}]}


class LiveHeadTests(unittest.TestCase):
    def test_live_boundaries_and_nonperturbation_pass(self):
        result, head = fixture()
        proof = compare_live_head(result, head)
        self.assertTrue(proof["allclose"])
        self.assertEqual(proof["observed_calls"], 1)

    def test_numerical_and_exact_feed_failures(self):
        for group, name in (("feeds", "input_states_2"), ("outputs", "output_states_1"),
                            ("outputs", "outputs"), ("feeds", "targets"), ("outputs", "prednet_lengths")):
            result, head = fixture()
            result["runs"][0]["headCalls"][0][group][name]["data"][0] += 2
            self.assertFalse(compare_live_head(result, head)["allclose"])

    def test_frozen_input_signed_zero_and_control_changes_fail(self):
        result, head = fixture()
        result["runs"][0]["headCalls"][0]["feeds"]["encoder_outputs"]["data"][0] = -0.
        self.assertFalse(compare_live_head(result, head)["allclose"])
        result, head = fixture()
        result["runs"][1]["decoderState"]["h"]["data"][0] += .0000001
        # Exact f32 required for snapshots: use an actual float32 value.
        result["runs"][1]["decoderState"]["h"]["data"][0] = float(np.float32(result["runs"][1]["decoderState"]["h"]["data"][0]))
        self.assertFalse(compare_live_head(result, head)["allclose"])

    def test_missing_order_storage_and_observer_control_fail_closed(self):
        for mutate in (lambda r: r["runs"][0]["headCalls"].clear(),
                       lambda r: r["runs"][0]["headCalls"][0].update(index=True),
                       lambda r: r["runs"][0]["headCalls"][0]["outputs"].pop("output_states_2"),
                       lambda r: r["runs"][0]["headCalls"][0]["feeds"]["targets"]["data"].__setitem__(0, True),
                       lambda r: r["runs"][0]["headCalls"][0]["outputs"]["outputs"]["data"].__setitem__(0, float("nan")),
                       lambda r: r["runs"][0].update(observerRestored=False),
                       lambda r: r["runs"][1].update(traceEnabled=True)):
            result, head = fixture()
            mutate(result)
            with self.assertRaises(ValueError):
                compare_live_head(result, head)

    def test_extra_call_and_untraced_calls_fail_closed(self):
        for mutate in (
                lambda r: r["runs"][0]["headCalls"].append(copy.deepcopy(r["runs"][0]["headCalls"][0])),
                lambda r: r["runs"][1]["headCalls"].append(copy.deepcopy(r["runs"][0]["headCalls"][0]))):
            result, head = fixture()
            mutate(result)
            with self.assertRaises(ValueError):
                compare_live_head(result, head)

    def test_shape_dtype_and_non_float32_storage_fail_closed(self):
        for mutate in (
                lambda t: t.update(shape=[1, 1, 3]),
                lambda t: t.update(dtype="float64"),
                lambda t: t["data"].__setitem__(0, .1)):
            result, head = fixture()
            mutate(result["runs"][0]["headCalls"][0]["feeds"]["encoder_outputs"])
            with self.assertRaises(ValueError):
                compare_live_head(result, head)

    def test_changed_second_run_output_never_passes(self):
        for name, changed in (("tokenIds", [1]), ("frameIndices", [1]),
                              ("tdtSteps", [0]), ("text", "different")):
            result, head = fixture()
            result["runs"][1][name] = changed
            self.assertFalse(compare_live_head(result, head)["allclose"])


if __name__ == "__main__":
    unittest.main()
