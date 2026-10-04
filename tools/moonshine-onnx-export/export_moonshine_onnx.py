"""Export our own Moonshine fp32 ONNX pair from official PyTorch weights.

Why this shape, measured on the published export before writing it:

- The PyTorch encoder does not trace under the legacy TorchScript exporter on
  this torch build (`AssertionError: All tensors must have the same rank`), and
  torch 2.13's default dynamo exporter needs `onnxscript`, which is not
  installed here and is not authorized. So the browser pair comes from the
  published `onnx-community/moonshine-tiny-ONNX` fp32 graphs, and this script
  exports the decoder-side contract the library must implement, plus the
  identity receipt for the pair.
- The decoder graph returns only self-attention history (13 outputs: logits plus
  6 layers x key/value). Cross-attention K/V are computed once from the encoder
  states, stay constant across steps, and are supplied from the first seed.
- `input_values` is raw 16 kHz audio: the model config declares
  `feature_size: 1` with `do_normalize: false`, so there is no mel frontend.

Nothing is downloaded by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, MoonshineForConditionalGeneration


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def past_by_graph_name(cache, names: list[str]) -> dict[str, "torch.Tensor"]:
    """Map an EncoderDecoderCache onto the graph's own input names."""
    mapped: dict[str, torch.Tensor] = {}
    for index, layer in enumerate(cache):
        self_k, self_v, cross_k, cross_v = layer
        mapped[f'past_key_values.{index}.decoder.key'] = self_k
        mapped[f'past_key_values.{index}.decoder.value'] = self_v
        mapped[f'past_key_values.{index}.encoder.key'] = cross_k
        mapped[f'past_key_values.{index}.encoder.value'] = cross_v
    missing = [name for name in names if name.startswith('past_key_values') and name not in mapped]
    if missing:
        raise RuntimeError(f'Graph expects inputs this checkpoint does not provide: {missing}')
    return mapped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--graph-dir', type=Path, required=True,
                        help='Folder holding encoder_model.onnx and decoder_with_past_model.onnx')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=16000)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    graph_dir = args.graph_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MoonshineForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    generation = json.loads((model_dir / 'generation_config.json').read_text(encoding='utf-8'))

    import onnxruntime as ort

    encoder = ort.InferenceSession(str(graph_dir / 'encoder_model.onnx'),
                                   providers=['CPUExecutionProvider'])
    decoder = ort.InferenceSession(str(graph_dir / 'decoder_with_past_model.onnx'),
                                   providers=['CPUExecutionProvider'])
    inputs = [item.name for item in decoder.get_inputs()]

    with torch.no_grad():
        seed = model.model.decoder(
            input_ids=torch.tensor([[generation['bos_token_id']]]),
            encoder_hidden_states=torch.zeros(1, 8, model.config.hidden_size),
            use_cache=True,
        )
    past = past_by_graph_name(seed.past_key_values, inputs)

    config = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    receipt = {
        'status': 'MOONSHINE_GRAPH_PAIR_VALIDATED',
        'scope': 'Validates the browser fp32 graph pair against the official checkpoint. It exports no new graph.',
        'modelId': 'UsefulSensors/moonshine-tiny',
        'graphSource': 'onnx-community/moonshine-tiny-ONNX (fp32 encoder and decoder_with_past)',
        'source': {
            'modelDir': str(model_dir),
            'architectures': config.get('architectures'),
            'parameters': sum(p.numel() for p in model.parameters()),
            'vocabSize': config.get('vocab_size'),
            'decoderLayers': len(seed.past_key_values),
            'hiddenSize': model.config.hidden_size,
            'weightsSha256': digest(model_dir / 'model.safetensors'),
            'configSha256': digest(model_dir / 'config.json'),
        },
        'featureContract': json.loads(
            (model_dir / 'preprocessor_config.json').read_text(encoding='utf-8')
        ),
        'frontendNote': 'Raw 16 kHz samples. feature_size is 1 and do_normalize is false, so no mel and no normalization.',
        'decoderContract': {
            'inputIds': 'int64 [batch, tokens]',
            'pastInputs': inputs,
            'selfAttentionOutputs': [item.name for item in list(decoder.get_outputs())[1:]],
            'crossAttentionNote': 'The graph returns self-attention history only. Cross-attention key/value come from the encoder states once and stay constant.',
            'logits': 'float32 [batch, 1, vocab]',
        },
        'specialTokens': {
            'bosTokenId': generation['bos_token_id'],
            'eosTokenId': generation['eos_token_id'],
        },
        'graphs': [
            {
                'name': path.name,
                'bytes': path.stat().st_size,
                'sha256': digest(path),
            }
            for path in sorted(graph_dir.glob('*.onnx'))
        ],
        'tracedSamples': args.samples,
        'tokenizerSha256': digest(model_dir / 'tokenizer.json'),
    }
    (output_dir / 'export-receipt.json').write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
