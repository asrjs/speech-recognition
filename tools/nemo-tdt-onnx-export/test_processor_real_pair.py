"""Constructor acceptance boundaries; no NeMo, weights or numerical claims."""
from types import SimpleNamespace
import unittest
import torch
from processor_real_pair import RealPairProcessor


class RealPairContractTests(unittest.TestCase):
    def processor(self, **overrides):
        values = dict(training=False, use_grads=False, exact_pad=False,
                      stft_pad_amount=None, frame_splicing=1, pad_to=0,
                      normalize='per_feature')
        values.update(overrides)
        return SimpleNamespace(featurizer=SimpleNamespace(**values))

    def test_supported_contract_keeps_official_featurizer(self):
        processor = self.processor()
        self.assertIs(RealPairProcessor(processor).featurizer, processor.featurizer)
        self.assertIsNone(RealPairProcessor(processor).diagnostic_normalizer)

    def test_diagnostic_normalizer_is_explicit_and_registered(self):
        module = torch.nn.Identity()
        adapter = RealPairProcessor(self.processor(), diagnostic_normalizer=module)
        self.assertIs(adapter.diagnostic_normalizer, module)
        self.assertIn('diagnostic_normalizer', dict(adapter.named_children()))

    def test_invalid_diagnostic_normalizer_fails(self):
        with self.assertRaisesRegex(TypeError, 'torch module'):
            RealPairProcessor(self.processor(), diagnostic_normalizer=lambda x: x)

    def test_unsupported_contracts_fail_closed(self):
        for overrides in ({'training': True}, {'use_grads': True}, {'exact_pad': True},
                          {'stft_pad_amount': 176}, {'frame_splicing': 2},
                          {'pad_to': 16}, {'pad_to': 'max'}, {'normalize': 'all_features'}):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, 'requires'):
                RealPairProcessor(self.processor(**overrides))


if __name__ == '__main__':
    unittest.main()
