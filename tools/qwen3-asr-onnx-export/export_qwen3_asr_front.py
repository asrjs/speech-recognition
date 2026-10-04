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


def build_cu_seqlens(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Cumulative window boundaries in post-CNN units.

    cu_seqlens counts frames after the convolutional downsampling, not raw
    feature frames, so it is derived with the encoder's own helper. The runtime
    needs the same computation, so the value is recorded in the receipt rather
    than hidden inside the graph.
    """
    from transformers.models.qwen3_asr.modeling_qwen3_asr import (
        _get_feat_extract_output_lengths,
    )

    encoder = model.model.audio_tower
    mask = torch.ones(features.shape[0], features.shape[-1], dtype=features.dtype)
    feature_lens = mask.sum(-1).to(torch.long)
    chunk_lengths = mask.sum(-1).to(torch.long)
    aftercnn = _get_feat_extract_output_lengths(chunk_lengths, encoder.n_window)
    n_window_ratio = encoder.n_window_infer // (encoder.n_window * 2)
    window_aftercnn = int(aftercnn.max().item()) * n_window_ratio
    bounds = [0]
    for length in aftercnn.tolist():
        bounds += [window_aftercnn] * (length // window_aftercnn)
        remainder = length % window_aftercnn
        if remainder:
            bounds += [remainder]

    return torch.tensor(bounds, dtype=torch.int32).cumsum(-1, dtype=torch.int32)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AudioFront(torch.nn.Module):
    """audio_tower plus multi_modal_projector as one graph."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.audio_tower = model.model.audio_tower
        self.projector = model.model.multi_modal_projector

    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        # The encoder needs the mask to pool variable-length audio, so it is a
        # real graph input rather than a baked constant.
        #
        # cu_seqlens is the documented override on get_audio_cu_seqlens. Without it
        # the schedule is computed in Python from the mask, which is data-dependent
        # control flow the exporter cannot specialize. Supplying it makes the
        # schedule a graph input the runtime computes once.
        encoded = self.audio_tower(
            input_features, input_features_mask, cu_seqlens=cu_seqlens
        )
        # The tower returns a pooled output object, not a tensor.
        hidden = getattr(encoded, "last_hidden_state", encoded)
        return self.projector(hidden)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/qwen3-asr-0.6b"))
    parser.add_argument("--audio", default="tools/data/fixtures/audio/jfk-short.wav")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-trace", action="store_true",
                        help="use the legacy tracer; the window schedule becomes a baked constant")
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
        cu_seqlens = build_cu_seqlens(model, features)
        reference = front(features, mask, cu_seqlens).numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "audio_front_model.onnx"
    torch.onnx.export(
        front,
        (features, mask, cu_seqlens),
        output,
        input_names=["input_features", "input_features_mask", "cu_seqlens"],
        output_names=["audio_embeds"],
        dynamic_axes={
            "input_features": {0: "batch", 2: "frames"},
            # The mask is 2-D [batch, frames]; features are 3-D [batch, bins, frames].
            "input_features_mask": {0: "batch", 1: "frames"},
            "audio_embeds": {0: "batch", 1: "frames"},
        },
        opset_version=OPSET,
        dynamo=not args.legacy_trace,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    # The legacy tracer bakes the window schedule in, so the graph's real inputs
    # are only those it still declares.
    graph_inputs = {entry.name for entry in session.get_inputs()}
    feed = {"input_features": features.numpy(), "input_features_mask": mask.numpy()}
    if "cu_seqlens" in graph_inputs:
        feed["cu_seqlens"] = cu_seqlens.numpy()
    produced = session.run(None, feed)[0]
    difference = np.abs(reference - produced)
    receipt = {
        "artifact": output.name,
        "bytes": output.stat().st_size,
        "sha256": _digest(output),
        "opset": OPSET,
        "exporter": "torch.onnx.export(dynamo=False)" if args.legacy_trace else "torch.onnx.export(dynamo=True)",
        "legacyTrace": bool(args.legacy_trace),
        "legacyTraceLimit": "the window schedule is baked as a constant, so this graph handles only the frame count it was exported at" if args.legacy_trace else None,
        "input": {
            "input_features": list(features.shape),
            "input_features_mask": list(mask.shape),
            "cu_seqlens": list(cu_seqlens.shape),
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
        "graphInputs": sorted(graph_inputs),
        "bakedSchedule": sorted(set(graph_inputs) - {"input_features", "input_features_mask"}),
        "generalisesBeyondExportedLength": False if args.legacy_trace else None,
        "tracerWarning": "torch emitted three TracerWarnings stating the trace may not generalize to other inputs; with legacy tracing these are the window split sizes, the padding check and the causal-attention test",
    }
    (args.output_dir / "audio-front-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
