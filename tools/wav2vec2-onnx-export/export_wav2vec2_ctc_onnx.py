"""Export a transformers Wav2Vec2ForCTC checkpoint to ONNX for our runtime.

Follows the same own-export discipline as tools/whisper-onnx-export and
tools/parakeet-ctc-onnx-export: we export from official PyTorch weights rather
than adopting a third-party repack, because the graph we run is the one we
control and can re-optimize later.

The graph takes raw 16 kHz samples as `input_values [batch, samples]` with the
convolutional frontend inside, matching the runtime's existing wav2vec2 and CTC
contract. Nothing is downloaded by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import Wav2Vec2ForCTC


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=16000)
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Wav2Vec2ForCTC.from_pretrained(model_dir).eval()
    samples = args.samples
    audio = torch.zeros(1, samples, dtype=torch.float32)

    class Graph(torch.nn.Module):
        """Wav2Vec2 encoder plus CTC head under the runtime input contract."""

        def __init__(self, inner: Wav2Vec2ForCTC) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_values: torch.Tensor) -> torch.Tensor:
            return self.inner(input_values).logits

    target = output_dir / 'model.onnx'
    torch.onnx.export(
        Graph(model),
        (audio,),
        str(target),
        input_names=['input_values'],
        output_names=['logits'],
        dynamic_axes={'input_values': {1: 'samples'}, 'logits': {1: 'frames'}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    vocab_path = model_dir / 'vocab.json'
    vocab = json.loads(vocab_path.read_text(encoding='utf-8'))
    receipt = {
        'status': 'EXPORTED_WAV2VEC2_CTC_ONNX',
        'scope': 'Own fp32 export from official PyTorch weights; not yet validated for browser parity.',
        'source': {
            'modelDir': str(model_dir),
            'architecture': model.config.architectures,
            'hiddenSize': model.config.hidden_size,
            'numHiddenLayers': model.config.num_hidden_layers,
            'vocabSize': model.config.vocab_size,
            'parameters': sum(p.numel() for p in model.parameters()),
            'weightsSha256': digest(model_dir / 'pytorch_model.bin'),
            'configSha256': digest(model_dir / 'config.json'),
        },
        'vocabulary': {'size': len(vocab), 'blankId': 0, 'source': vocab_path.name},
        'featureContract': json.loads(
            (model_dir / 'preprocessor_config.json').read_text(encoding='utf-8')
        ),
        'export': {
            'opset': args.opset,
            'tracedSamples': samples,
            'graph': str(target),
            'graphSha256': digest(target),
            'graphBytes': target.stat().st_size,
        },
    }
    (output_dir / 'export-receipt.json').write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())