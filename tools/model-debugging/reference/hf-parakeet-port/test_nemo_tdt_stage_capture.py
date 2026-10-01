"""Small simulations test instrumentation safety, not real-model parity."""

from types import SimpleNamespace

import hashlib
import unittest
import torch

from nemo_tdt_stage_capture import NemoTdtStageCapture


class PairModule(torch.nn.Module):
    def forward(self, input_signal, length):
        return input_signal + 1, length


class Predictor(torch.nn.Module):
    def predict(self, y=None, state=None, add_sos=True, batch_size=None):
        output = y.float().unsqueeze(-1) + state[0]
        return output, [state[0] + 1, state[1] + 2]


class Joint(torch.nn.Module):
    def project_encoder(self, encoder):
        return encoder * 2

    def project_prednet(self, predictor):
        return predictor * 3

    def joint_after_projection(self, encoder, predictor):
        return encoder + predictor


def make_model():
    return SimpleNamespace(
        cfg=SimpleNamespace(decoding=SimpleNamespace(strategy="greedy_batch", model_type="tdt")),
        preprocessor=PairModule(), encoder=PairModule(), decoder=Predictor(), joint=Joint(),
    )


def run(model):
    features, length = model.preprocessor(torch.ones(1, 2), torch.tensor([2]))
    encoded, _ = model.encoder(features, length)
    encoded = model.joint.project_encoder(encoded)
    output, state = model.decoder.predict(torch.tensor([[3]]), [torch.ones(1, 1, 2)] * 2,
                                         add_sos=False, batch_size=1)
    return model.joint.joint_after_projection(encoded, model.joint.project_prednet(output)), state


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.model = make_model()

    def test_method_outputs_unchanged_and_hooks_removed(self):
        model = self.model
        expected, _ = run(model)
        with NemoTdtStageCapture(model) as capture:
            actual, _ = run(model)
        capture.validate_complete()
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(list(capture.stages)[:6], [
            "waveform", "waveform_length", "features", "features_length", "encoder", "encoder_length",
        ])
        self.assertEqual(capture.stages["predictor/0000/labels"]["data"], [3])
        self.assertFalse(capture.calls[1]["add_sos"])
        self.assertNotIn("predict", model.decoder.__dict__)
        self.assertFalse(model.preprocessor._forward_hooks)
        self.assertFalse(model.preprocessor._forward_pre_hooks)
        self.assertFalse(model.encoder._forward_hooks)

    def test_keyword_waveform_capture_has_independent_float32_little_endian_identity(self):
        waveform = torch.tensor([[0.0, 0.5, -1.0, 0.25]])
        with NemoTdtStageCapture(self.model) as capture:
            actual, length = self.model.preprocessor(
                input_signal=waveform, length=torch.tensor([4]),
            )
        waveform.fill_(99)
        self.assertTrue(torch.equal(actual, torch.tensor([[1.0, 1.5, 0.0, 1.25]])))
        self.assertEqual(length.tolist(), [4])
        self.assertEqual(capture.stages["waveform"]["data"], [0.0, 0.5, -1.0, 0.25])
        self.assertEqual(capture.model_input_contract(16000), {
            "boundary": "preprocessor_input_signal",
            "sample_rate": 16000, "channels": 1, "dtype": "float32",
            "encoding": "f32le", "shape": [1, 4],
            "samples": 4, "valid_samples": 4, "right_padding_samples": 0,
            "sha256": hashlib.sha256(bytes.fromhex(
                "000000000000003f000080bf0000803e",
            )).hexdigest(),
        })
        self.assertFalse(self.model.preprocessor._forward_pre_hooks)

    def test_waveform_contract_keeps_padding_and_valid_length_separate(self):
        with NemoTdtStageCapture(self.model) as capture:
            self.model.preprocessor(torch.tensor([[0.5, -0.5, 0.0, 0.0]]), torch.tensor([2]))
        contract = capture.model_input_contract(16000)
        self.assertEqual(contract["samples"], 4)
        self.assertEqual(contract["valid_samples"], 2)
        self.assertEqual(contract["right_padding_samples"], 2)
        self.assertEqual(capture.stages["waveform"]["data"], [0.5, -0.5, 0.0, 0.0])

    def test_waveform_contract_rejects_invalid_rate_length_layout_and_dtype(self):
        cases = [
            (torch.ones(1, 4), torch.tensor([5]), 16000),
            (torch.ones(1, 4), torch.tensor([0]), 16000),
            (torch.ones(1, 4), torch.tensor([4.0]), 16000),
            (torch.ones(2, 4), torch.tensor([4, 4]), 16000),
            (torch.ones(1, 1, 4), torch.tensor([4]), 16000),
            (torch.ones(1, 4, dtype=torch.float64), torch.tensor([4]), 16000),
            (torch.ones(1, 4), torch.tensor([4]), 0),
        ]
        for waveform, length, sample_rate in cases:
            with self.subTest(shape=waveform.shape, dtype=waveform.dtype, length=length, rate=sample_rate):
                capture = NemoTdtStageCapture(self.model)
                capture.record("waveform", waveform)
                capture.record("waveform_length", length)
                with self.assertRaisesRegex(RuntimeError, "model-input"):
                    capture.model_input_contract(sample_rate)

    def test_snapshots_do_not_alias_native_mutable_tensors(self):
        value = torch.tensor([1.0, 2.0])
        capture = NemoTdtStageCapture(self.model)
        capture.record("original", value)
        value.fill_(99)
        self.assertEqual(capture.stages["original"]["data"], [1.0, 2.0])

    def test_methods_restored_on_exception(self):
        def failing_joint(encoder, predictor):
            raise RuntimeError("native failure")

        model = self.model
        model.joint.joint_after_projection = failing_joint
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            with NemoTdtStageCapture(model):
                run(model)
        self.assertIs(model.joint.joint_after_projection, failing_joint)
        self.assertNotIn("predict", model.decoder.__dict__)
        self.assertFalse(model.encoder._forward_hooks)
        self.assertFalse(model.preprocessor._forward_pre_hooks)

    def test_failed_setup_removes_partial_hooks(self):
        model = self.model
        model.joint.project_encoder = None
        with self.assertRaises(TypeError):
            with NemoTdtStageCapture(model):
                self.fail("invalid hook setup must not execute")
        self.assertNotIn("predict", model.decoder.__dict__)
        self.assertFalse(model.preprocessor._forward_hooks)
        self.assertFalse(model.preprocessor._forward_pre_hooks)

    def test_call_limit_fails_instead_of_truncating(self):
        model = self.model
        with self.assertRaisesRegex(RuntimeError, "no truncated PASS"):
            with NemoTdtStageCapture(model, max_calls=1):
                run(model)
                model.decoder.predict(torch.tensor([[3]]), [torch.ones(1, 1, 2)] * 2)
        self.assertNotIn("predict", model.decoder.__dict__)

    def test_empty_capture_cannot_be_reference(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            NemoTdtStageCapture(self.model).validate_complete()

    def test_invalid_tensors_rejected(self):
        for value in (torch.tensor([]), torch.tensor([float("nan")])):
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    NemoTdtStageCapture(self.model).record("bad", value)

    def test_non_cpu_capture_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "CPU"):
            NemoTdtStageCapture(self.model).record("bad", torch.empty(1, device="meta"))

    def test_duplicate_stage_rejected(self):
        capture = NemoTdtStageCapture(self.model)
        capture.record("features", torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            capture.record("features", torch.ones(1))

if __name__ == "__main__":
    unittest.main()
