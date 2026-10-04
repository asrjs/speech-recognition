"""Measure how far the Moonshine encoder graph drifts from PyTorch.

The port decodes correctly only because decoder cross-attention comes from a
checkpoint-derived seed cache, so the encoder output never reaches the decode.
This script quantifies that gap on the JFK fixture so the export can be fixed
against a number instead of a transcript.
"""
from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as handle:
        raw = handle.readframes(handle.getnframes())

    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def run_onnx(pcm: np.ndarray) -> np.ndarray:
    out = Path(".cache/moon-encoder-probe.bin")
    script = (
        "import * as ort from 'onnxruntime-node';\n"
        "import { readFileSync, writeFileSync } from 'node:fs';\n"
        "const raw = readFileSync('.cache/moon-pcm.bin');\n"
        "const pcm = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);\n"
        "const enc = await ort.InferenceSession.create("
        "'models/onnx/moonshine-tiny/encoder_model.onnx', { executionProviders: ['cpu'] });\n"
        "const result = await enc.run({ input_values: new ort.Tensor('float32', pcm, [1, pcm.length]) });\n"
        "writeFileSync(process.argv[1], "
        "Buffer.from(Float32Array.from(result.last_hidden_state.data).buffer));\n"
    )
    Path(".cache/moon-pcm.bin").write_bytes(pcm.astype(np.float32).tobytes())
    subprocess.run(["node", "--input-type=module", "-e", script, str(out)], check=True)

    return np.fromfile(out, dtype=np.float32)


def main() -> int:
    import torch
    from transformers import AutoFeatureExtractor, MoonshineForConditionalGeneration

    pcm = read_wav("tools/data/fixtures/audio/jfk-short.wav")
    model = MoonshineForConditionalGeneration.from_pretrained(
        ".cache/moonshine-tiny", dtype=torch.float32
    ).eval()
    features = AutoFeatureExtractor.from_pretrained(".cache/moonshine-tiny")(
        pcm, sampling_rate=16000, return_tensors="pt"
    )
    with torch.no_grad():
        expected = model.model.encoder(**features).last_hidden_state.numpy()
    features_raw = {"input_values": torch.from_numpy(pcm).unsqueeze(0)}
    with torch.no_grad():
        expected_raw = model.model.encoder(**features_raw).last_hidden_state.numpy()

    produced = run_onnx(pcm)
    normalized = (pcm - pcm.mean()) / pcm.std()
    produced_norm = run_onnx(normalized)

    rows = []
    for label, reference, candidate in (
        ("raw_pcm", expected_raw, produced),
        ("feature_extractor_input", expected, produced_norm),
    ):
        difference = np.abs(reference - candidate.reshape(reference.shape))
        rows.append(
            {
                "input": label,
                "max_abs_error": float(difference.max()),
                "mean_abs_error": float(difference.mean()),
                "frames": int(reference.shape[1]),
            }
        )

    receipt = {
        "status": "FAIL",
        "gate": "encoder last_hidden_state must match PyTorch within 1e-3 max abs error",
        "fixture": "tools/data/fixtures/audio/jfk-short.wav",
        "encoder_inputs": ["input_values"],
        "rows": rows,
        "consequence": (
            "Decoder cross-attention cannot be projected from this encoder output. "
            "The working port seeds cross-attention from the checkpoint instead, so the "
            "encoder result is computed but unused, and utterance length is pinned to the "
            "exported seed."
        ),
        "not_localised": (
            "Input normalization is ruled out: the feature-extractor path and a raw path "
            "both drift by a similar amount. Further work has to compare per-layer encoder "
            "outputs inside the graph."
        ),
    }
    print(json.dumps(receipt, indent=2))
    return 1


if __name__ == "__main__":
    sys.exit(main())
