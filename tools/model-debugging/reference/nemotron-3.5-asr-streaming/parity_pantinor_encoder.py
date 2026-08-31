#!/usr/bin/env python
"""Test pantinor Nemotron 3.5 ONNX encoder parity vs NeMo reference.

The pantinor export is structured like sherpa-onnx/parakeet-rs:
- encoder.onnx: streaming with cache_last_channel/cache_last_time/prompt_index
                outputs encoded [B, 1024, T] (pre-projection, matches NeMo)
- decoder_joint.onnx: batched combined decoder+joint, takes all enc frames
                      and all targets at once, returns [B, T, U, 13088]
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", default="N:/models/onnx/nemo/nemotron-3.5-asr-streaming-pantinor")
    p.add_argument("--nemo", default="N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo")
    p.add_argument("--fixture", default="tools/data/fixtures/audio/jfk-short.wav")
    p.add_argument("--output", default="tools/data/results/nemotron/nemotron-3.5-pantinor-encoder-parity-2026-08-31.json")
    p.add_argument("--prompt-id", type=int, default=0)
    return p.parse_args()


def load_wav(path, target_sr=16000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.squeeze(0).numpy(), target_sr


def main() -> None:
    args = parse_args()
    onnx_dir = Path(args.onnx_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading pantinor encoder ...", flush=True)
    enc = ort.InferenceSession(str(onnx_dir / "encoder.onnx"), providers=["CPUExecutionProvider"])

    print("Loading NeMo model ...", flush=True)
    nemo = EncDecRNNTBPEModelWithPrompt.restore_from(restore_path=args.nemo, map_location="cpu")
    nemo.eval()
    nemo.set_inference_prompt("en")

    # Get mel via NeMo preprocessor
    wav, sr = load_wav(args.fixture)
    sig = torch.from_numpy(wav).float().unsqueeze(0)
    length = torch.tensor([len(wav)], dtype=torch.long)
    with torch.no_grad():
        mel_t = nemo.preprocessor(input_signal=sig, length=length)[0]
    mel = mel_t.transpose(1, 2).squeeze(0).numpy()  # [T, 128]
    print(f"Mel shape: {mel.shape}, max={float(mel.max()):.2f}", flush=True)

    # Run pantinor encoder (offline full pass, no streaming chunks)
    feeds = {
        "processed_signal": mel.T[np.newaxis, ...].astype(np.float32),  # [1, 128, T]
        "processed_signal_length": np.array([mel.shape[0]], dtype=np.int64),
        "cache_last_channel": np.zeros([24, 1, 56, 1024], dtype=np.float32),
        "cache_last_time": np.zeros([24, 1, 1024, 8], dtype=np.float32),
        "cache_last_channel_len": np.zeros([1], dtype=np.int64),
        "prompt_index": np.array([args.prompt_id], dtype=np.int64),
    }
    print("Running pantinor encoder ...", flush=True)
    outputs = enc.run(None, feeds)
    result = dict(zip([o.name for o in enc.get_outputs()], outputs))
    pantinor_enc = result["encoded"]  # [1, 1024, T]
    pantinor_len = int(result["encoded_len"][0])
    print(f"Pantinor encoded shape: {pantinor_enc.shape}, len={pantinor_len}", flush=True)
    print(f"Pantinor encoded maxAbs: {float(np.abs(pantinor_enc).max()):.3f}, meanAbs: {float(np.abs(pantinor_enc).mean()):.4f}", flush=True)

    # Run NeMo encoder for comparison
    mel_torch = torch.from_numpy(mel).float().unsqueeze(0).transpose(1, 2)  # [1, 128, T]
    mel_len_torch = torch.tensor([mel.shape[0]], dtype=torch.long)
    with torch.no_grad():
        nemo_enc, nemo_len = nemo.encoder(audio_signal=mel_torch, length=mel_len_torch)
    nemo_enc_np = nemo_enc.numpy()  # [1, 1024, T]
    print(f"NeMo encoded shape: {nemo_enc_np.shape}, len={int(nemo_len[0])}", flush=True)
    print(f"NeMo encoded maxAbs: {float(np.abs(nemo_enc_np).max()):.3f}, meanAbs: {float(np.abs(nemo_enc_np).mean()):.4f}", flush=True)

    # Compare first valid enc frames
    T = min(pantinor_enc.shape[2], nemo_enc_np.shape[2], int(nemo_len[0]), pantinor_len)
    print(f"Comparing first {T} enc frames ...", flush=True)
    p_first = pantinor_enc[0, :, :T]
    n_first = nemo_enc_np[0, :, :T]
    diff = np.abs(p_first - n_first)
    cos_sim = float(np.dot(p_first.flatten(), n_first.flatten()) / (
        np.linalg.norm(p_first.flatten()) * np.linalg.norm(n_first.flatten()) + 1e-9
    ))
    print(f"  maxAbsErr: {float(diff.max()):.4f}")
    print(f"  meanAbsErr: {float(diff.mean()):.4f}")
    print(f"  cosSim: {cos_sim:.4f}")

    record = {
        "schemaVersion": 1,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "Pantinor Nemotron 3.5 ONNX encoder parity vs NeMo reference",
        "onnxSource": "pantinor/nemotron-3.5-asr-streaming-0.6b-onnx (sherpa-onnx/parakeet-rs architecture)",
        "fixture": args.fixture,
        "promptId": args.prompt_id,
        "nemoEncoderOutShape": list(nemo_enc_np.shape),
        "nemoEncoderOutMaxAbs": float(np.abs(nemo_enc_np).max()),
        "nemoEncoderOutMeanAbs": float(np.abs(nemo_enc_np).mean()),
        "pantinorEncoderOutShape": list(pantinor_enc.shape),
        "pantinorEncoderOutMaxAbs": float(np.abs(pantinor_enc).max()),
        "pantinorEncoderOutMeanAbs": float(np.abs(pantinor_enc).mean()),
        "framesCompared": T,
        "maxAbsErr": float(diff.max()),
        "meanAbsErr": float(diff.mean()),
        "cosineSim": cos_sim,
        "verdict": "PASS" if (cos_sim > 0.99 and float(diff.max()) < 0.5) else "FAIL",
    }
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output_path}", flush=True)
    print(f"Verdict: {record['verdict']}", flush=True)


if __name__ == "__main__":
    main()