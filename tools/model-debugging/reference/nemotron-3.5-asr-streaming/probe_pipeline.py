#!/usr/bin/env python
"""Probe encoder+decoder+joiner frame-by-frame to diagnose token parity issue."""
import sys
import numpy as np
import onnxruntime as ort
from pathlib import Path

mel = np.load("tools/data/results/nemotron/_tmp_mel.npy")
print(f"Mel shape: {mel.shape}, max={float(mel.max()):.2f}")

enc_first = ort.InferenceSession("N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-onnx/encoder_320ms_first_fp16.onnx", providers=["CPUExecutionProvider"])
enc_cont = ort.InferenceSession("N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-onnx/encoder_320ms_fp16.onnx", providers=["CPUExecutionProvider"])
decoder = ort.InferenceSession("N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-onnx/decoder.onnx", providers=["CPUExecutionProvider"])
joiner = ort.InferenceSession("N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-onnx/joiner.onnx", providers=["CPUExecutionProvider"])

vocab_path = Path("N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-onnx/tokens.txt")
id_to_token = {}
for line in vocab_path.read_text(encoding="utf-8").splitlines():
    parts = line.split("\t", 1)
    if len(parts) == 2:
        id_to_token[int(parts[0])] = parts[1]


def init_caches(sess):
    caches = {}
    for inp in sess.get_inputs():
        if inp.name in ("input_features", "prompt_ids", "cache_mask"):
            continue
        shape = [max(1, d) for d in inp.shape]
        dtype = np.int64 if "int64" in inp.type else np.float32
        caches[inp.name] = np.zeros(shape, dtype=dtype)
    return caches


def run_enc(sess, chunk, prompt_id, caches, cache_mask_value):
    feeds = {
        "input_features": chunk.astype(np.float32),
        "prompt_ids": np.array([prompt_id], dtype=np.int64),
        "cache_mask": np.full([1, 1, 1, 60], cache_mask_value, dtype=np.float32),
    }
    feeds.update(caches)
    outs = sess.run(None, feeds)
    out_names = [o.name for o in sess.get_outputs()]
    result = dict(zip(out_names, outs))
    new_caches = {}
    for k, v in result.items():
        if k == "encoder_out":
            continue
        if "_out_" in k:
            base, _, tail = k.partition("_out_")
            new_caches[f"{base}_{tail}"] = v
        else:
            new_caches[k] = v
    return result["encoder_out"], new_caches


def run_dec(token_id, h, c):
    feeds = {
        "token": np.array([[token_id]], dtype=np.int64),
        "h_in": h.astype(np.float32),
        "c_in": c.astype(np.float32),
    }
    outs = decoder.run(None, feeds)
    out_names = [o.name for o in decoder.get_outputs()]
    result = dict(zip(out_names, outs))
    return result["decoder_out"], result["h_out"], result["c_out"]


def run_join(enc_frame, dec_out):
    feeds = {
        "encoder_frame": enc_frame.astype(np.float32),
        "decoder_out": dec_out.astype(np.float32),
    }
    return joiner.run(None, feeds)[0]


# Test: encoder first chunk only
first_caches = init_caches(enc_first)
chunk = mel[:25][np.newaxis, ...].astype(np.float32)
enc1, _ = run_enc(enc_first, chunk, 0, first_caches, 0.0)
print(f"enc1 shape: {enc1.shape}")

# Run decoder with blank target, then joiner
h = np.zeros([2, 1, 640], dtype=np.float32)
c = np.zeros([2, 1, 640], dtype=np.float32)

print("=== First enc chunk, decoder state=zero, target=blank ===")
for t in range(enc1.shape[1]):
    enc_frame = enc1[:, t, :]
    dec_out, h, c = run_dec(13087, h, c)
    logits = run_join(enc_frame, dec_out)
    y = int(np.argmax(logits))
    lp = float(logits[0, y])
    tok = id_to_token.get(y, "?")
    print(f"  t={t}: argmax={y} ({tok!r}), logProb={lp:.2f}")

# Test: now run second chunk using cont encoder
print("\n=== Second enc chunk ===")
cont_caches = init_caches(enc_cont)
chunk2 = mel[32:64][np.newaxis, ...].astype(np.float32)
enc2, cont_caches = run_enc(enc_cont, chunk2, 0, cont_caches, 1.0)
print(f"enc2 shape: {enc2.shape}, maxAbs={float(np.abs(enc2).max()):.3f}")

h = np.zeros([2, 1, 640], dtype=np.float32)
c = np.zeros([2, 1, 640], dtype=np.float32)
for t in range(enc2.shape[1]):
    enc_frame = enc2[:, t, :]
    dec_out, h, c = run_dec(13087, h, c)
    logits = run_join(enc_frame, dec_out)
    y = int(np.argmax(logits))
    lp = float(logits[0, y])
    tok = id_to_token.get(y, "?")
    print(f"  t={t}: argmax={y} ({tok!r}), logProb={lp:.2f}")