"""Weight-free real-pair STFT export feasibility; NOT processor/model parity.

Run explicitly with installed Torch/ONNX/ORT: python <this-file> -v.
No source monkeypatch, model loading, graph file or dependency install.
Constants describe this bounded NeMo fixture, not a universal processor.
"""
import io
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch


class StftBoundary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(400, periodic=False))

    def forward(self, waveform):
        return torch.stft(
            waveform, n_fft=512, hop_length=160, win_length=400,
            center=True, window=self.window, return_complex=False, pad_mode="constant",
        )


class ProcessorStftExportTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        self.module = StftBoundary().eval()
        self.generator = torch.Generator().manual_seed(0)

    def waveform(self, samples):
        return torch.randn(1, samples, generator=self.generator) * 0.01

    def test_real_pair_preserves_native_complex_boundary_bytes(self):
        for samples in (1600, 3200):
            with self.subTest(samples=samples), torch.no_grad():
                x = self.waveform(samples)
                actual = self.module(x).numpy()
                expected = torch.view_as_real(torch.stft(
                    x, n_fft=512, hop_length=160, win_length=400, center=True,
                    window=self.module.window, return_complex=True, pad_mode="constant",
                )).numpy()
                self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_export_executes_dynamic_lengths_on_explicit_cpu(self):
        buffer = io.BytesIO()
        torch.onnx.export(
            self.module, (self.waveform(1600),), buffer, opset_version=17,
            dynamo=False, input_names=["waveforms"], output_names=["spectrum"],
            dynamic_axes={"waveforms": {1: "samples"}, "spectrum": {2: "frames"}},
        )
        graph = onnx.load_model_from_string(buffer.getvalue())
        onnx.checker.check_model(graph)
        self.assertIn("STFT", [node.op_type for node in graph.graph.node])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            buffer.getvalue(), sess_options=options, providers=["CPUExecutionProvider"],
        )
        self.assertEqual(session.get_providers(), ["CPUExecutionProvider"])
        for samples in (1600, 3200):
            with self.subTest(samples=samples), torch.no_grad():
                x = self.waveform(samples)
                expected = self.module(x).numpy()
                actual = session.run(None, {"waveforms": x.numpy()})[0]
                self.assertEqual(actual.shape, (1, 257, samples // 160 + 1, 2))
                self.assertEqual(actual.dtype, np.float32)
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
