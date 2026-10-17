"""Exercise real fp16 Qwen decoder export boundaries without downloading weights."""
from __future__ import annotations

import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch
from transformers import Qwen3Config, Qwen3Model
from transformers import Qwen3ASRConfig, Qwen3ASREncoderConfig, Qwen3ASRForConditionalGeneration

from export_qwen3_asr_fp16 import Float32DecoderStep, export_decoder, export_front
from capture_cu_seqlens import capture_schedule


class SmallQwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = Qwen3Config(
            hidden_size=64, intermediate_size=96, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=64, max_position_embeddings=256,
        )
        config._attn_implementation = "eager"
        self.config = SimpleNamespace(text_config=config)
        self.model = torch.nn.Module()
        self.model.language_model = Qwen3Model(config).half().eval()
        self.lm_head = torch.nn.Linear(64, 64, bias=False).half().eval()


class DecoderExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(17)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.model = SmallQwen().eval()
        cls.step = Float32DecoderStep(cls.model).eval()
        cls.oracle = Float32DecoderStep(copy.deepcopy(cls.model).float()).eval()
        cls.receipt = export_decoder(cls.model, cls.root)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cls.session = ort.InferenceSession(
            str(cls.root / "decoder_step_model.onnx"), options,
            providers=["CPUExecutionProvider"],
        )

    @classmethod
    def tearDownClass(cls):
        cls.session = None
        cls.directory.cleanup()

    def test_cache_zero_one_and_token_axis_roundtrip(self):
        # fp16 relative gate: against the fp32 oracle on the same weights, the
        # ONNX error must not exceed native PyTorch fp16 error. ORT CPU runs
        # many fp16 ops in fp32, so it cannot reproduce native fp16 rounding.
        generator = np.random.default_rng(91)
        errors = {"native": [], "onnx": []}
        for initial_past in (0, 1, 4):
            keys = np.zeros((2, 1, 2, initial_past, 16), dtype=np.float32)
            values = keys.copy()
            caches = {
                "native": (torch.from_numpy(keys), torch.from_numpy(values)),
                "oracle": (torch.from_numpy(keys), torch.from_numpy(values)),
            }
            for count in (2, 1, 3):
                embeds = generator.normal(0, 0.1, (1, count, 64)).astype(np.float32)
                with torch.no_grad():
                    native = self.step(torch.from_numpy(embeds), *caches["native"])
                    oracle = self.oracle(torch.from_numpy(embeds), *caches["oracle"])
                produced = self.session.run(None, {
                    "inputs_embeds": embeds, "past_key": keys, "past_value": values,
                })
                expected_length = keys.shape[3] + count
                self.assertEqual(produced[1].shape, (2, 1, 2, expected_length, 16))
                self.assertEqual(produced[2].shape, produced[1].shape)
                for reference, fp16, actual in zip(oracle, native, produced):
                    self.assertEqual(actual.dtype, np.dtype(np.float32))
                    self.assertEqual(actual.shape, tuple(reference.shape))
                    errors["native"].append(np.abs(fp16.numpy() - reference.numpy()))
                    errors["onnx"].append(np.abs(actual - reference.numpy()))
                keys, values = produced[1:]
                caches["native"] = native[1:]
                caches["oracle"] = oracle[1:]
        native_error = np.concatenate([e.ravel() for e in errors["native"]])
        onnx_error = np.concatenate([e.ravel() for e in errors["onnx"]])
        self.assertLessEqual(onnx_error.max(), native_error.max())
        self.assertLessEqual(onnx_error.mean(), native_error.mean())

    def test_wrong_feed_dtype_is_rejected(self):
        with self.assertRaisesRegex(Exception, "type|Type|float"):
            self.session.run(None, {
                "inputs_embeds": np.zeros((1, 1, 64), dtype=np.float16),
                "past_key": np.zeros((2, 1, 2, 0, 16), dtype=np.float32),
                "past_value": np.zeros((2, 1, 2, 0, 16), dtype=np.float32),
            })

    def test_missing_sidecar_fails_session_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / "decoder_step_model.onnx"
            shutil.copyfile(self.root / graph.name, graph)
            with self.assertRaisesRegex(Exception, "data|file|File"):
                ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])


class EncoderScheduleTests(unittest.TestCase):
    """Keep window topology tests fp32 to isolate them from fp16 rounding."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        audio = Qwen3ASREncoderConfig(
            d_model=32, encoder_layers=1, encoder_attention_heads=4,
            encoder_ffn_dim=64, downsample_hidden_size=4, output_dim=64,
        )
        text = Qwen3Config(
            hidden_size=64, intermediate_size=96, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=64,
        )
        config = Qwen3ASRConfig(audio_config=audio, text_config=text)
        config._attn_implementation = "eager"
        config.audio_config._attn_implementation = "eager"
        config.text_config._attn_implementation = "eager"
        cls.model = Qwen3ASRForConditionalGeneration(config).float().eval()
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.features = torch.randn(1, 128, 1100)
        mask = torch.ones(1, 1100)
        schedule = capture_schedule(cls.model, cls.features, mask)
        export_front(cls.model, cls.features, mask, schedule, cls.root)
        cls.session = ort.InferenceSession(
            str(cls.root / "audio_front_model.onnx"), providers=["CPUExecutionProvider"],
        )

    @classmethod
    def tearDownClass(cls):
        cls.session = None
        cls.directory.cleanup()

    def test_one_graph_accepts_different_frame_and_window_counts(self):
        for frames in (100, 500, 1100):
            with self.subTest(frames=frames):
                features = self.features[:, :, :frames].contiguous()
                mask = torch.ones(1, frames)
                schedule = capture_schedule(self.model, features, mask)
                with torch.no_grad():
                    reference = self.model.model.multi_modal_projector(
                        self.model.model.audio_tower(
                            features, mask, cu_seqlens=schedule,
                        ).last_hidden_state
                    ).numpy()
                produced = self.session.run(None, {
                    "input_features": features.numpy(),
                    "input_features_mask": mask.numpy(),
                    "cu_seqlens": schedule.numpy(),
                })[0]
                np.testing.assert_allclose(produced, reference, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
