"""Check the exported cross-attention projection against PyTorch.

Runs one PyTorch decoder step with hooks on every cross-attention key and value
projection, then feeds the same encoder states into the ONNX graph and compares
all twelve tensors.
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from artifact_outputs import write_new_text  # noqa: E402


def main() -> int:
    from transformers import AutoFeatureExtractor, MoonshineForConditionalGeneration

    model = MoonshineForConditionalGeneration.from_pretrained(
        ".cache/moonshine-tiny", dtype=torch.float32
    ).eval()
    with wave.open("tools/data/fixtures/audio/jfk-short.wav", "rb") as handle:
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    batch = AutoFeatureExtractor.from_pretrained(".cache/moonshine-tiny")(
        pcm, sampling_rate=16000, return_tensors="pt"
    )

    with torch.no_grad():
        encoder_states = model.model.encoder(**batch).last_hidden_state.numpy()

    # One decoder step with hooks gives the ground-truth cross-attention tensors.
    decoder = model.model.decoder
    captured: list[torch.Tensor] = []
    handles = [
        getattr(layer.encoder_attn, role).register_forward_hook(
            lambda _module, _inputs, output: captured.append(output.detach())
        )
        for layer in decoder.layers
        for role in ("k_proj", "v_proj")
    ]
    try:
        with torch.no_grad():
            model(**batch, use_cache=True, decoder_input_ids=torch.tensor([[1]]))
    finally:
        for handle in handles:
            handle.remove()

    ordered = [
        value.reshape(1, 8, value.shape[1], 36).numpy() for value in captured
    ]
    assert len(ordered) == 12, f"captured {len(ordered)} tensors, expected 12"

    session = ort.InferenceSession(
        "models/onnx/moonshine-tiny/decoder_cross_kv_model.onnx", providers=["CPUExecutionProvider"]
    )
    produced = session.run(None, {"encoder_states": encoder_states})

    # The projection emits [batch, frames, heads * headDim]; the decoder wants the
    # cache layout [batch, heads, frames, headDim], so the comparison reshapes.
    rows = []
    for layer in range(6):
        for index, role in ((0, "key"), (1, "value")):
            expected = ordered[layer * 2 + index]
            batch, frames, _hidden = produced[layer * 2 + index].shape
            actual = produced[layer * 2 + index].reshape(
                batch, expected.shape[1], frames, expected.shape[3]
            )
            rows.append(
                {
                    "layer": layer,
                    "role": role,
                    "shape": list(actual.shape),
                    "max_abs_error": float(np.abs(expected - actual).max()),
                    "mean_abs_error": float(np.abs(expected - actual).mean()),
                }
            )

    worst = max(row["max_abs_error"] for row in rows)
    receipt = {
        "status": "PASS" if worst <= 1e-4 else "FAIL",
        "gate": "max_abs_error <= 1e-4 against PyTorch cross-attention",
        "encoder_frames": int(encoder_states.shape[1]),
        "worst_max_abs_error": worst,
        "per_tensor": rows,
    }
    write_new_text(
        Path("tools/data/results/moonshine/cross-kv-parity-v2-2026-10-04.json"),
        json.dumps(receipt, indent=2) + "\n",
    )
    print(json.dumps({k: v for k, v in receipt.items() if k != "per_tensor"}, indent=2))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
