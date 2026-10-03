"""Export a transformers ParakeetForCTC encoder+CTC head to ONNX.

The graph takes mel features and an attention mask and returns per-frame
logits, matching the CTC contract this repository's CTC decoders already use.
Emitted under an existing, ignored artifact directory; nothing is downloaded by
this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import ParakeetForCTC

ROOT = Path(__file__).resolve().parents[2]


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
    parser.add_argument('--frames', type=int, default=1400,
                        help='Mel frames to trace with; dynamic axes cover any length.')
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ParakeetForCTC.from_pretrained(model_dir, dtype=torch.float32).eval()
    mel_bins = int(model.config.encoder_config.num_mel_bins)
    frames = args.frames
    features = torch.zeros(1, frames, mel_bins, dtype=torch.float32)
    mask = torch.ones(1, frames, dtype=torch.int64)

    class Wrapper(torch.nn.Module):
        """Encoder plus CTC head under the runtime's own input contract."""

        def __init__(self, inner: ParakeetForCTC) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return self.inner(input_features=input_features, attention_mask=attention_mask).logits

    wrapper = Wrapper(model)
    target = output_dir / 'model.onnx'
    torch.onnx.export(
        wrapper,
        (features, mask),
        str(target),
        input_names=['input_features', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_features': {1: 'frames'},
            'attention_mask': {1: 'frames'},
            'logits': {1: 'out_frames'},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        # The dynamo exporter needs onnxscript, which this environment does not
        # install; the legacy TorchScript exporter has no extra dependency.
        dynamo=False,
    )

    receipt = {
        'status': 'EXPORTED_PARAKEET_CTC_ONNX',
        'scope': 'Encoder plus CTC head export; not model parity or browser acceptance.',
        'modelId': 'ctaguchi/parakeet-hf-ctc-110m-en',
        'source': {
            'modelDir': str(model_dir),
            'parameters': sum(p.numel() for p in model.parameters()),
            'vocabSize': int(model.config.vocab_size),
            'padTokenId': int(model.config.pad_token_id),
            'melBins': mel_bins,
            'subsamplingFactor': int(model.config.encoder_config.subsampling_factor),
            'modelSafetensorsSha256': digest(model_dir / 'model.safetensors'),
            'configSha256': digest(model_dir / 'config.json'),
        },
        'featureContract': json.loads((model_dir / 'processor_config.json').read_text(encoding='utf-8')),
        'export': {
            'opset': args.opset,
            'tracedFrames': frames,
            'graph': str(target.relative_to(ROOT)).replace('\\', '/'),
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