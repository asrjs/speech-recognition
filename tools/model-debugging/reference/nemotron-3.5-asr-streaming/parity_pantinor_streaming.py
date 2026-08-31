#!/usr/bin/env python
"""Stream the pantinor encoder and compare frame-by-frame vs NeMo reference."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch, torchaudio
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

OUT = Path("tools/data/results/nemotron/nemotron-3.5-pantinor-streaming-parity-2026-08-31.json")

def load_wav(p, target_sr=16000):
    wav, sr = torchaudio.load(p)
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr: wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.squeeze(0).numpy(), target_sr

print("Loading ...", flush=True)
enc = ort.InferenceSession("N:/models/onnx/nemo/nemotron-3.5-asr-streaming-pantinor/encoder.onnx", providers=["CPUExecutionProvider"])
nemo = EncDecRNNTBPEModelWithPrompt.restore_from(
    "N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo",
    map_location="cpu")
nemo.eval(); nemo.set_inference_prompt("en")

wav, _ = load_wav("tools/data/fixtures/audio/jfk-short.wav")
sig = torch.from_numpy(wav).float().unsqueeze(0); length = torch.tensor([len(wav)], dtype=torch.long)
with torch.no_grad():
    mel_t = nemo.preprocessor(input_signal=sig, length=length)[0]
mel = mel_t.transpose(1, 2).squeeze(0).numpy()
print(f"Mel shape: {mel.shape}", flush=True)

# NeMo offline
mel_torch = torch.from_numpy(mel).float().unsqueeze(0).transpose(1, 2)
mel_len_torch = torch.tensor([mel.shape[0]], dtype=torch.long)
with torch.no_grad():
    nemo_enc, nemo_len = nemo.encoder(audio_signal=mel_torch, length=mel_len_torch)
nemo_enc_np = nemo_enc.numpy()
print(f"NeMo enc: {nemo_enc_np.shape}", flush=True)

# Pantinor streaming: chunk=32, accumulate all enc frames
chunk_size = 32
cache_ch = np.zeros([24, 1, 56, 1024], dtype=np.float32)
cache_t = np.zeros([24, 1, 1024, 8], dtype=np.float32)
cache_ch_len = np.zeros([1], dtype=np.int64)
all_enc = []
mel_idx = 0
i = 0
while mel_idx < mel.shape[0]:
    chunk = mel[mel_idx:mel_idx+chunk_size]
    if chunk.shape[0] < chunk_size:
        chunk = np.vstack([chunk, np.zeros([chunk_size-chunk.shape[0], 128], dtype=np.float32)])
    feeds = {
        "processed_signal": chunk.T[np.newaxis, ...],
        "processed_signal_length": np.array([chunk.shape[0]], dtype=np.int64),
        "cache_last_channel": cache_ch,
        "cache_last_time": cache_t,
        "cache_last_channel_len": cache_ch_len,
        "prompt_index": np.array([0], dtype=np.int64),
    }
    outputs = enc.run(None, feeds)
    result = dict(zip([o.name for o in enc.get_outputs()], outputs))
    all_enc.append(result["encoded"][0])  # [1024, T_chunk]
    cache_ch = result["cache_last_channel_next"]
    cache_t = result["cache_last_time_next"]
    cache_ch_len = result["cache_last_channel_len_next"]
    mel_idx += chunk_size
    i += 1
pantinor_full = np.concatenate(all_enc, axis=1)  # [1024, total_frames]
print(f"Pantinor streaming enc: {pantinor_full.shape}", flush=True)

# Compare frame-by-frame (assume alignment)
T = min(pantinor_full.shape[1], nemo_enc_np.shape[2])
p = pantinor_full[:, :T]
n = nemo_enc_np[0, :, :T]
diff = np.abs(p - n)
cos = float(np.dot(p.flatten(), n.flatten()) / (np.linalg.norm(p.flatten()) * np.linalg.norm(n.flatten()) + 1e-9))
print(f"\nFrame-by-frame (first {T} frames):")
print(f"  maxAbsErr={float(diff.max()):.4f}, meanAbsErr={float(diff.mean()):.4f}, cosSim={cos:.4f}")

# Now try with shifted alignment (pantinor might lag by 1 due to cache boundary)
for shift in [0, 1, 2, 3]:
    p_s = pantinor_full[:, shift:shift+T]
    n_s = nemo_enc_np[0, :, :T]
    if p_s.shape[1] < T:
        continue
    d = np.abs(p_s - n_s)
    cs = float(np.dot(p_s.flatten(), n_s.flatten()) / (np.linalg.norm(p_s.flatten()) * np.linalg.norm(n_s.flatten()) + 1e-9))
    print(f"  shift={shift}: maxAbsErr={float(d.max()):.4f}, meanAbsErr={float(d.mean()):.4f}, cosSim={cs:.4f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
record = {
    "schemaVersion": 1,
    "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "purpose": "Pantinor streaming encoder parity vs NeMo (with cache management)",
    "frames": T,
    "nemoEncShape": list(nemo_enc_np.shape),
    "pantinorEncShape": list(pantinor_full.shape),
    "alignmentShift0_maxAbsErr": float(np.abs(pantinor_full[:, :T] - nemo_enc_np[0, :, :T]).max()),
    "alignmentShift0_cosSim": cos,
    "verdict": "PASS" if cos > 0.99 else "FAIL",
}
OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(f"\nWrote {OUT}", flush=True)
print(f"Verdict: {record['verdict']}", flush=True)