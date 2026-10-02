"""Replay contract sanity tests: synthetic calls do not certify real model parity."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from validate_stage_replay import log_softmax, replay_bindings, stage, unique_frame
import validate_stage_replay as replay


def sample_fixture():
    cfg = {"joint": {"jointnet": {"encoder_hidden": 2, "joint_hidden": 2}},
           "decoder": {"vocab_size": 2, "prednet": {"pred_rnn_layers": 1, "pred_hidden": 2}}}
    stages = {}

    def put(name, data, dtype="float32"):
        value = np.asarray(data, dtype=dtype)
        stages[name] = {"shape": list(value.shape), "dtype": dtype, "data": value.ravel().tolist()}

    put("encoder", [[[1, 2], [3, 4]]])
    put("encoder_projection/0000/input_encoder_output", [[[1, 3], [2, 4]]])
    put("encoder_projection/0000/output", [[[5, 6], [7, 8]]])
    put("predictor/0000/labels", [[2]], "int64")
    put("predictor/0000/output", [[[9, 10]]])
    for name in ("input_state_h", "input_state_c", "output_state_h", "output_state_c"):
        put("predictor/0000/" + name, [[[0, 0]]])
    put("predictor_projection/0000/input_prednet_output", [[[9, 10]]])
    put("predictor_projection/0000/output", [[[11, 12]]])
    for index, row in enumerate(([5, 6], [7, 8])):
        put(f"joint/{index:04d}/input_f", [[row]])
        put(f"joint/{index:04d}/input_g", [[[11, 12]]])
    calls = [{"kind": kind, "index": index, "stage_prefix": f"{kind}/{index:04d}"}
             for kind, index in (("encoder_projection", 0), ("predictor", 0),
                                 ("predictor_projection", 0), ("joint", 0), ("joint", 1))]
    calls[1].update(labels_missing=False, state_missing=False, add_sos=False)
    return {"stages": stages, "calls": calls}, cfg


class ReplayTests(unittest.TestCase):
    def test_case_materializes_encoder_once_after_binding_validation(self):
        sample, cfg = sample_fixture()
        cfg["joint"].update(num_classes=2, num_extra_outputs=1)
        stages = sample["stages"]

        def put(name, value, dtype="float32"):
            array = np.asarray(value, dtype=dtype)
            stages[name] = {"shape": list(array.shape), "dtype": dtype,
                            "data": array.ravel().tolist()}

        put("features", [[[1, 2]]])
        put("features_length", [2], "int64")
        put("encoder_length", [2], "int64")
        scores = log_softmax(np.array([[[[1, 2, 3, 4]]]], dtype=np.float32))
        for index in range(2):
            put(f"joint/{index:04d}/output", scores)
        sample.update(sample_id="synthetic", audio={"sha256": "audio"},
                      tokens=[1], transcript="test", token_frames=[0], token_duration=[1])
        identity = {"model": "synthetic", "model_revision": "revision", "weight_sha256": "weight"}
        flags = dict.fromkeys(replay.CONTROL_KEYS, True)
        meta = {**identity, "audio_sha256": "audio", "device": "cpu", "batch_size": 1}
        _, counts = replay_bindings(sample, cfg)
        reference = {"schema_version": 1, "meta": meta, "samples": [sample],
                     "instrumentation_control": flags, "stage_order": list(stages)}
        control = {"meta": meta, "stage_capture": {"instrumentation_control": flags,
                   "stage_count": len(stages), "call_counts": counts},
                   "result": {"token_ids": [1], "text": "test", "timestamp": [0], "token_duration": [1]}}
        encoder = Mock()
        encoder.get_outputs.return_value = [SimpleNamespace(name=name) for name in ("outputs", "encoded_lengths")]
        encoder.run.return_value = [stage(stages, "encoder"), np.array([2], dtype=np.int64)]
        head = Mock()
        head.get_outputs.return_value = [SimpleNamespace(name=name) for name in
                                        ("outputs", "output_states_1", "output_states_2", "prednet_lengths")]
        head.run.side_effect = lambda _, feeds: [scores.copy(), feeds["input_states_1"].copy(),
                                                feeds["input_states_2"].copy(), np.ones(1, dtype=np.int32)]
        with patch.object(replay, "read_json", side_effect=[reference, control]), \
                patch.object(replay, "range_hash", return_value="audio"), \
                patch.object(replay, "stage", wraps=stage) as materialize:
            result = replay.validate_case(Path("capture"), Path("summary"), Path("audio"),
                                          cfg, identity, (encoder, head), 1e-5, 1e-4)
        self.assertEqual(result["status"], "RECORDED_BOUNDARIES_PASS")
        self.assertEqual(result["end_to_end_tokens_text"], "UNTESTED")
        self.assertEqual(head.run.call_count, 2)
        raw = stage(stages, "encoder")
        for index, call in enumerate(head.run.call_args_list):
            feeds = call.args[1]
            expected = {"encoder_outputs": raw[:, :, index:index+1],
                        "targets": np.array([[2]], dtype=np.int32),
                        "target_length": np.ones(1, dtype=np.int32),
                        "input_states_1": np.zeros((1, 1, 2), dtype=np.float32),
                        "input_states_2": np.zeros((1, 1, 2), dtype=np.float32)}
            self.assertEqual(set(feeds), set(expected))
            for name, value in expected.items():
                np.testing.assert_array_equal(feeds[name], value)
                self.assertEqual(feeds[name].dtype, value.dtype)
            self.assertTrue(feeds["encoder_outputs"].flags.c_contiguous)
        # One chronology check plus one reusable array in validate_case, not one per joint.
        self.assertEqual(sum(call.args[1] == "encoder" for call in materialize.call_args_list), 2)

    def test_multiple_joints_reuse_the_latest_predictor_not_a_new_search(self):
        sample, cfg = sample_fixture()
        bindings, counts = replay_bindings(sample, cfg)
        self.assertEqual([item["frame"] for item in bindings], [0, 1])
        self.assertEqual([item["predictor"] for item in bindings], ["predictor/0000"] * 2)
        self.assertEqual(counts["predictor"], 1)

    def test_wrong_call_index_sos_and_label_range_fail(self):
        for mutation in (lambda s: s["calls"][1].update(index=1),
                         lambda s: s["calls"][1].update(add_sos=True),
                         lambda s: s["stages"]["predictor/0000/labels"].update(data=[3])):
            sample, cfg = sample_fixture()
            mutation(sample)
            with self.assertRaises(ValueError):
                replay_bindings(sample, cfg)

    def test_mismatched_projected_inputs_fail_before_ort(self):
        for name in ("encoder_projection/0000/input_encoder_output",
                     "predictor_projection/0000/input_prednet_output", "joint/0000/input_g"):
            sample, cfg = sample_fixture()
            sample["stages"][name]["data"][0] += 1
            with self.subTest(name=name), self.assertRaises(ValueError):
                replay_bindings(sample, cfg)

    def test_ambiguous_or_missing_projected_row_fails(self):
        projected = np.array([[[1, 2], [1, 2]]], dtype=np.float32)
        for row in (np.array([[[1, 2]]], dtype=np.float32), np.array([[[3, 4]]], dtype=np.float32)):
            with self.assertRaises(ValueError):
                unique_frame(projected, row)

    def test_nonmonotonic_joint_frames_fail(self):
        sample, cfg = sample_fixture()
        sample["stages"]["joint/0000/input_f"]["data"] = [7, 8]
        sample["stages"]["joint/0001/input_f"]["data"] = [5, 6]
        with self.assertRaisesRegex(ValueError, "monotonic"):
            replay_bindings(sample, cfg)

    def test_reference_shape_dtype_count_and_finiteness_are_not_guessed(self):
        sample, _ = sample_fixture()
        original = sample["stages"]["encoder"]
        for change in ({"data": [1]}, {"data": [float("nan")] * 4}, {"dtype": "float64"}, {"shape": [0, 2, 2]}):
            bad = copy.deepcopy(original)
            bad.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                stage({"encoder": bad}, "encoder", "float32")

    def test_log_softmax_normalizes_all_channels_stably_not_separate_heads(self):
        raw = np.array([[[[1000, -1000, 999, 998]]]], dtype=np.float32)
        normalized = log_softmax(raw)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertAlmostEqual(float(np.exp(normalized.astype(np.float64)).sum()), 1, places=6)
        np.testing.assert_allclose(log_softmax(raw + 100), normalized, atol=1e-5, rtol=0)
        self.assertGreater(float(np.exp(normalized[..., :3]).sum()), 0)
        self.assertLess(float(np.exp(normalized[..., :3]).sum()), 1)


if __name__ == "__main__":
    unittest.main()
