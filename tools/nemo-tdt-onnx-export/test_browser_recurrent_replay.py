"""Synthetic fail-closed harness tests, not browser/model acceptance."""
import copy
import base64
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from browser_head_replay import tensor, resolve_route
from browser_recurrent_replay import pcm_fixture, final_state_fixture, evaluate_result, select_ort_module


def state(value):
    return tensor(np.full((1, 1, 2), value, dtype=np.float32))


def fixture():
    first = {"binding": {"frame": 0}, "feeds": {"targets": {"data": [2]},
             "input_states_1": state(0), "input_states_2": state(0)},
             "expected": {"normalized_scores": {"data": [0, 3, 1, 0, 2]},
                          "output_states_1": state(1), "output_states_2": state(2)}}
    second = copy.deepcopy(first)
    second["binding"]["frame"] = 1
    second["expected"].update(normalized_scores={"data": [0, 1, 3, 0, 2]},
                              output_states_1=state(3), output_states_2=state(4))
    sample = {"tokens": [1], "token_frames": [0], "token_duration": [1]}
    cfg = {"decoding": {"durations": [0, 1]}, "joint": {"num_extra_outputs": 2}}
    return sample, {"blank_id": 2, "steps": [first, second]}, cfg


class RecurrentBrowserTests(unittest.TestCase):
    def test_final_state_discards_trailing_blank_outputs(self):
        sample, head, cfg = fixture()
        self.assertEqual(final_state_fixture(sample, head, cfg), {"h": state(1), "c": state(2)})

    def test_all_blank_keeps_initial_state(self):
        sample, head, cfg = fixture()
        sample.update(tokens=[], token_frames=[], token_duration=[])
        head["steps"][0]["expected"]["normalized_scores"]["data"] = [0, 1, 3, 0, 2]
        self.assertEqual(final_state_fixture(sample, head, cfg), {"h": state(0), "c": state(0)})

    def test_changed_official_decisions_initial_state_and_duration_contract_fail(self):
        for mutate in (lambda s, h, c: s.update(token_frames=[1]),
                       lambda s, h, c: h["steps"][0]["feeds"].update(input_states_1=state(1)),
                       lambda s, h, c: c["decoding"].update(durations=[0, 2])):
            sample, head, cfg = fixture()
            mutate(sample, head, cfg)
            with self.assertRaises(ValueError):
                final_state_fixture(sample, head, cfg)

    def test_pcm_identity_length_and_f32_precision_guards(self):
        data = np.array([[0., -0., .5]], dtype=np.float32)
        sample = {"stages": {"waveform": tensor(data), "waveform_length": {"dtype": "int32", "shape": [1], "data": [3]}},
                  "model_input": {"boundary": "preprocessor_input_signal", "sample_rate": 16000, "channels": 1,
                    "dtype": "float32", "encoding": "f32le", "shape": [1, 3], "samples": 3,
                    "valid_samples": 3, "right_padding_samples": 0, "sha256": hashlib.sha256(data.tobytes()).hexdigest()}}
        encoded = pcm_fixture(sample, 16000)
        self.assertEqual(encoded["pcmSha256"], sample["model_input"]["sha256"])
        self.assertEqual(base64.b64decode(encoded["pcmF32leBase64"]), data.tobytes())
        self.assertNotIn("pcm", encoded)
        for mutate in (lambda s: s["model_input"].update(sha256="wrong"),
                       lambda s: s["stages"]["waveform_length"].update(data=[2]),
                       lambda s: s["stages"]["waveform"]["data"].__setitem__(2, .1),
                       lambda s: s["stages"]["waveform"]["data"].__setitem__(2, True)):
            changed = copy.deepcopy(sample)
            mutate(changed)
            with self.assertRaises(ValueError):
                pcm_fixture(changed, 16000)

    def result(self):
        expected = {"pcmSha256": "pcm", "tokenIds": [1], "frameIndices": [0], "tdtSteps": [1],
                    "text": "x", "decoderState": {"h": state(1), "c": state(2)}}
        result = {"backendRequested": "webgpu", "encoderBackend": "wasm", "decoderBackend": "webgpu",
                  "loaded": {"backend": "webgpu", "family": "nemo-tdt"}, "pcmSha256": "pcm",
                  "crossOriginIsolated": True, "cleanupErrors": [], "runs": [{**copy.deepcopy(expected),
                    "repeat": repeat, "gpuDispatches": 1} for repeat in range(2)]}
        return result, expected

    def test_exact_output_and_numerical_state_controls(self):
        result, expected = self.result()
        self.assertTrue(evaluate_result(result, expected, "webgpu"))
        result, expected = self.result()
        result["runs"][1]["decoderState"]["c"]["data"][0] = 4
        self.assertFalse(evaluate_result(result, expected, "webgpu"))

    def test_real_encoder_requires_actual_hybrid_metrics_and_mode(self):
        result, expected = self.result()
        expected.update(realEncoder=True, encoderBackend='webgpu')
        result.update(encoderBackend='webgpu', decoderBackend='wasm',
                      backendRequested='wasm', loaded={'backend': 'wasm', 'family': 'nemo-tdt'},
                      realEncoder='EXECUTED_NOT_COMPONENT_ACCEPTED')
        for run in result['runs']:
            run['metrics'] = {'encoderBackend': 'webgpu', 'decoderBackend': 'wasm'}
        self.assertTrue(evaluate_result(copy.deepcopy(result), expected, 'wasm'))
        for mutate in (lambda r: r.update(realEncoder='BYPASSED_FROZEN_ORACLE'),
                       lambda r: r['runs'][0].update(metrics={}),
                       lambda r: r['runs'][1]['metrics'].update(encoderBackend='wasm')):
            changed = copy.deepcopy(result)
            mutate(changed)
            with self.assertRaises(ValueError):
                evaluate_result(changed, expected, 'wasm')
        result['runs'][0]['gpuDispatches'] = 0
        self.assertFalse(evaluate_result(result, expected, 'wasm'))

    def test_failure_cleanup_zero_dispatch_and_divergent_output_never_pass(self):
        for mutate in (lambda r: r.update(cleanupErrors=["failed"]),
                       lambda r: r.update(error="failed"),
                       lambda r: r["runs"][0].update(gpuDispatches=0),
                       lambda r: r["runs"][0].update(tokenIds=[2])):
            result, expected = self.result()
            mutate(result)
            self.assertFalse(evaluate_result(result, expected, "webgpu"))

    def test_missing_or_malformed_outputs_identity_and_repeats_fail(self):
        for mutate in (lambda r: r.update(pcmSha256="wrong"),
                       lambda r: r.update(decoderBackend="wasm"),
                       lambda r: r["runs"].pop(),
                       lambda r: r["runs"][1].update(repeat=0),
                       lambda r: r["runs"][0].update(tokenIds=[True]),
                       lambda r: r["runs"][0].update(decoderState=None),
                       lambda r: r["runs"][0]["decoderState"]["h"].update(shape=[1, 2]),
                       lambda r: r["runs"][0]["decoderState"]["h"]["data"].__setitem__(0, float("nan"))):
            result, expected = self.result()
            mutate(result)
            with self.assertRaises(ValueError):
                evaluate_result(result, expected, "webgpu")

    def test_extra_routes_are_exact_allowlist_not_directory_access(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.txt"
            path.write_text("synthetic")
            self.assertEqual(resolve_route("/vocab.txt", Path("graph.onnx"), {"/vocab.txt": path}), path)
            for url in ("/other.txt", "/vendor/../fixture.txt", "/vocab.txt/other"):
                with self.assertRaises(ValueError):
                    resolve_route(url, Path("graph.onnx"), {"/vocab.txt": path})
            with self.assertRaises(ValueError):
                resolve_route("/missing", Path("graph.onnx"), {"/missing": path.with_name("missing")})


class OrtBundleSelectionTests(unittest.TestCase):
    """The import-map build must follow the providers actually needed."""

    def test_webgpu_encoder_with_wasm_decoder_selects_the_webgpu_build(self):
        # The regression: this combination previously selected the all-bundle,
        # a different ORT build, and every cross-run comparison then differed by
        # a constant 1.1316e-6.
        self.assertEqual(
            select_ort_module('wasm', 'webgpu', 'auto'), '/ort-dist/ort.webgpu.min.mjs')

    def test_webgpu_decoder_selects_the_webgpu_build(self):
        self.assertEqual(
            select_ort_module('webgpu', None, 'auto'), '/ort-dist/ort.webgpu.min.mjs')

    def test_pure_wasm_selects_the_all_bundle(self):
        self.assertEqual(
            select_ort_module('wasm', None, 'auto'), '/ort-dist/ort.all.bundle.min.mjs')

    def test_explicit_overrides_win_in_both_directions(self):
        self.assertEqual(
            select_ort_module('wasm', 'webgpu', 'all'), '/ort-dist/ort.all.bundle.min.mjs')
        self.assertEqual(
            select_ort_module('wasm', None, 'webgpu'), '/ort-dist/ort.webgpu.min.mjs')

    def test_selection_is_never_none_and_always_a_served_route(self):
        for backend, encoder, override in (
            ('wasm', None, 'auto'), ('wasm', 'webgpu', 'auto'), ('webgpu', None, 'auto'),
            ('wasm', 'webgpu', 'all'), ('wasm', None, 'webgpu'),
        ):
            with self.subTest(backend=backend, encoder=encoder, override=override):
                self.assertTrue(
                    select_ort_module(backend, encoder, override).startswith('/ort-dist/ort.'))


if __name__ == "__main__":
    unittest.main()
