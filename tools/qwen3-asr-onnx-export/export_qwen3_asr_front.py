"""Export the Qwen3-ASR audio front as a standalone fp32 graph.

The front is `audio_tower` plus `multi_modal_projector`: 128-bin mel features in,
language-model embeddings out. Exporting it separately keeps the mel contract in
one testable piece and lets the decoder graph be checked independently, which is
the discipline the goal document requires for numerical parity.

Run with the transformers 5.x environment:
    tools/model-debugging/reference/transformers5/Scripts/python.exe \\
        tools/qwen3-asr-onnx-export/export_qwen3_asr_front.py --output-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers

OPSET = 18


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AudioFront(torch.nn.Module):
    """audio_tower plus multi_modal_projector as one graph."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.audio_tower = model.model.audio_tower
        self.projector = model.model.multi_modal_projector

    def forward(self, input_features: torch.Tensor, input_features_mask: torch.Tensor) -> torch.Tensor:
        # The encoder needs the mask to pool variable-length audio, so it is a
        # real graph input rather than a baked constant.
        encoded = self.audio_tower(input_features, input_features_mask)
        # The tower returns a pooled output object, not a tensor.
        hidden = getattr(encoded, "last_hidden_state", encoded)
        return self.projector(hidden)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/qwen3-asr-0.6b"))
    parser.add_argument("--audio", default="tools/data/fixtures/audio/jfk-short.wav")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        args.model_dir, dtype=torch.float32
    ).eval()
    processor = transformers.AutoProcessor.from_pretrained(args.model_dir)

    import wave

    with wave.open(args.audio, "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    inputs = processor.apply_transcription_request(audio=pcm, sampling_rate=16000)
    features = inputs["input_features"].to(torch.float32)
    mask = inputs.get("input_features_mask")
    if mask is None:
        mask = inputs.get("attention_mask")
    mask = mask.to(features.dtype) if mask is not None else torch.ones_like(features)

    front = AudioFront(model).eval()
    with torch.no_grad():
        reference = front(features, mask).numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "audio_front_model.onnx"
    torch.onnx.export(
        front,
        (features, mask),
        output,
        input_names=["input_features", "input_features_mask"],
        output_names=["audio_embeds"],
        dynamic_axes={
            "input_features": {0: "batch", 2: "frames"},
            "input_features_mask": {0: "batch", 2: "frames"},
            "audio_embeds": {0: "batch", 1: "frames"},
        },
        opset_version=OPSET,
        dynamo=True,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    produced = session.run(
        None,
        {"input_features": features.numpy(), "input_features_mask": mask.numpy()},
    )[0]
    difference = np.abs(reference - produced)
    receipt = {
        "artifact": output.name,
        "bytes": output.stat().st_size,
        "sha256": _digest(output),
        "opset": OPSET,
        "exporter": "torch.onnx.export(dynamo=True)",
        "input": {
            "input_features": list(features.shape),
            "input_features_mask": list(mask.shape),
        },
        "output": {"name": "audio_embeds", "shape": list(produced.shape)},
        "parity": {
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "allclose_atol_1e-4_rtol_1e-3": bool(
                np.allclose(reference, produced, atol=1e-4, rtol=1e-3)
            ),
        },
        "note": "mel contract: 128 bins, n_fft 400, hop 160, 16 kHz, matching the repository's JSMelProcessor",
    }
    (args.output_dir / "audio-front-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
