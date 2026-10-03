"""Weight-free ONNX feasibility; does not establish official model parity."""
import io
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch

from processor_pairwise_normalization import PairwiseNormalization, pairwise_sum


class PairwiseNormalizationTests(unittest.TestCase):
    def setUp(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        self.addCleanup(torch.set_num_threads, old_threads)
        self.generator = torch.Generator().manual_seed(17)

    def test_pairwise_sum_preserves_odd_tail(self):
        for frames in (1, 3, 32, 67, 4061):
            with self.subTest(frames=frames):
                values = torch.ones(1, 3, frames)
                self.assertTrue(torch.equal(pairwise_sum(values), torch.full((1, 3), float(frames))))

    def test_single_valid_frame_and_masked_tail(self):
        features, mean, std = PairwiseNormalization(pad_value=-7)(torch.tensor([[[2., 99., 99.]]]), torch.tensor([1]))
        self.assertEqual(features.tolist(), [[[0., -7., -7.]]])
        self.assertEqual(mean.item(), 2.)
        self.assertTrue(torch.isfinite(std).all())

    def test_dynamic_onnx_loop_matches_native_candidate(self):
        module = PairwiseNormalization().eval()
        buffer = io.BytesIO()
        torch.onnx.export(module, (torch.randn(1, 3, 11, generator=self.generator), torch.tensor([10])),
            buffer, opset_version=17, dynamo=False, input_names=['log_mel', 'valid'],
            output_names=['features', 'mean', 'std'],
            dynamic_axes={'log_mel': {2: 'frames'}, 'features': {2: 'frames'}})
        graph = onnx.load_model_from_string(buffer.getvalue())
        onnx.checker.check_model(graph)
        self.assertIn('Loop', [node.op_type for node in graph.graph.node])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(buffer.getvalue(), options, providers=['CPUExecutionProvider'])
        self.assertEqual(session.get_providers(), ['CPUExecutionProvider'])
        for frames in (1, 3, 32, 67, 4061):
            with self.subTest(frames=frames), torch.no_grad():
                x = torch.randn(1, 3, frames, generator=self.generator)
                valid = torch.tensor([max(1, frames - 1)])
                expected = module(x, valid)
                actual = session.run(None, {'log_mel': x.numpy(), 'valid': valid.numpy()})
                for left, right in zip(expected, actual):
                    self.assertEqual(right.dtype, np.float32)
                    self.assertTrue(np.isfinite(right).all())
                    np.testing.assert_allclose(right, left.numpy(), atol=1e-5, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
