#!/usr/bin/env python
"""Parity test for the onnx-community INT4 Nemotron 3.5 export vs NeMo reference.

INT4 export architecture (different from pantinor):
- encoder.onnx: chunk-based streaming, fixed 65-frame input chunk,
  outputs encoder features [B, 7, 1024] per chunk
- decoder.onnx: standard RNN-T decoder (target + h_in/c_in → decoder_output)
- joint.onnx: takes encoder_output + decoder_output, returns logits

Architecture mirrors the codavidgarcia ONNX layout but INT4 quantized.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch, torchaudio
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", default="N:/models/onnx/nemo/nemotron-3.5-asr-streaming-int4")
    p.add_argument("--nemo", default="N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo")
    p.add_argument("--fixture", default="tools/data/fixtures/audio/jfk-short.wav")
    p.add_argument("--output", default="tools/data/results/nemotron/nemotron-3.5-int4-encoder-parity-2026-08-31.json")
    p.add_argument("--lang-id", type=int, default=0)
    return p.parse_args()


def load_wav(p, target_sr=16000):
    wav, sr = torchaudio.load(p)
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr: wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.squeeze(0).numpy(), target_sr


def main():
    args = parse_args()
    onnx_dir = Path(args.onnx_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading INT4 encoder ...", flush=True)
    enc = ort.InferenceSession(str(onnx_dir / "encoder.onnx"), providers=["CPUExecutionProvider"])

    print("Loading NeMo ...", flush=True)
    nemo = EncDecRNNTBPEModelWithPrompt.restore_from(restore_path=args.nemo, map_location="cpu")
    nemo.eval()
    nemo.set_inference_prompt("en")

    wav, _ = load_wav(args.fixture)
    sig = torch.from_numpy(wav).float().unsqueeze(0)
    length = torch.tensor([len(wav)], dtype=torch.long)
    with torch.no_grad():
        mel_t = nemo.preprocessor(input_signal=sig, length=length)[0]
    mel = mel_t.transpose(1, 2).squeeze(0).numpy()
    print(f"Mel: {mel.shape}, max={float(mel.max()):.2f}", flush=True)

    # Stream the INT4 encoder: 65-frame chunks
    chunk_size = 65
    cache_ch = np.zeros([1, 24, 56, 1024], dtype=np.float32)
    cache_t = np.zeros([1, 24, 1024, 8], dtype=np.float32)
    cache_ch_len = np.zeros([1], dtype=np.int64)
    all_enc = []
    mel_idx = 0
    iters = 0
    while mel_idx < mel.shape[0] and iters < 30:
        chunk = mel[mel_idx:mel_idx+chunk_size]
        if chunk.shape[0] < chunk_size:
            chunk = np.vstack([chunk, np.zeros([chunk_size-chunk.shape[0], 128], dtype=np.float32)])
        # INT4 expects audio_signal [1, 65, 128] (time, mel)
        feeds = {
            "audio_signal": chunk[np.newaxis, ...].astype(np.float32),  # [1, 65, 128]
            "length": np.array([chunk.shape[0]], dtype=np.int64),
            "cache_last_channel": cache_ch,
            "cache_last_time": cache_t,
            "cache_last_channel_len": cache_ch_len,
            "lang_id": np.array([args.lang_id], dtype=np.int64),
        }
        outputs = enc.run(None, feeds)
        result = dict(zip([o.name for o in enc.get_outputs()], outputs))
        enc_chunk = result["outputs"]  # [1, 7, 1024]
        enc_len = int(result["encoded_lengths"][0])
        cache_ch = result["cache_last_channel_next"]
        cache_t = result["cache_last_time_next"]
        cache_ch_len = result["cache_last_channel_len_next"]
        all_enc.append(enc_chunk[0])  # [7, 1024]
        print(f"iter {iters}: mel_idx={mel_idx}, enc_chunk_shape={enc_chunk.shape}, enc_len={enc_len}", flush=True)
        mel_idx += chunk_size
        iters += 1
    enc_full = np.concatenate(all_enc, axis=0)  # [N*7, 1024]
    print(f"Total INT4 enc: {enc_full.shape}", flush=True)

    # NeMo reference
    mel_torch = torch.from_numpy(mel).float().unsqueeze(0).transpose(1, 2)
    mel_len_torch = torch.tensor([mel.shape[0]], dtype=torch.long)
    with torch.no_grad():
        nemo_enc, nemo_len = nemo.encoder(audio_signal=mel_torch, length=mel_len_torch)
    nemo_enc_np = nemo_enc.numpy()
    print(f"NeMo enc: {nemo_enc_np.shape}", flush=True)

    T = min(enc_full.shape[0], nemo_enc_np.shape[2])
    p = enc_full[:T, :]
    n = nemo_enc_np[0, :, :T].T  # [T, 1024]
    diff = np.abs(p - n)
    cos = float(np.dot(p.flatten(), n.flatten()) / (np.linalg.norm(p.flatten()) * np.linalg.norm(n.flatten()) + 1e-9))
    print(f"\nFirst {T} frames: maxAbsErr={float(diff.max()):.4f}, meanAbsErr={float(diff.mean()):.4f}, cosSim={cos:.4f}")

    record = {
        "schemaVersion": 1,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "INT4 Nemotron 3.5 ONNX encoder parity vs NeMo reference",
        "onnxSource": "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4",
        "fixture": args.fixture,
        "langId": args.lang_id,
        "nemoEncShape": list(nemo_enc_np.shape),
        "int4EncShape": list(enc_full.shape),
        "framesCompared": T,
        "maxAbsErr": float(diff.max()),
        "meanAbsErr": float(diff.mean()),
        "cosineSim": cos,
        "verdict": "PASS" if cos > 0.99 and float(diff.max()) < 1.0 else "FAIL",
    }
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output_path}", flush=True)
    print(f"Verdict: {record['verdict']}", flush=True)


if __name__ == "__main__":
    main()