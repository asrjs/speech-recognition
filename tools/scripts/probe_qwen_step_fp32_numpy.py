"""Diagnose the Qwen INT4 step probe: feed sensitivity check."""

import numpy as np
import onnxruntime as ort

PATH = "N:/models/onnx/qwen3-asr-0.6b-official/decoder-step.onnx"
LAYERS, HEADS, HEAD_DIM = 28, 8, 128

session = ort.InferenceSession(PATH, providers=["CPUExecutionProvider"])


def run(past_len, token_id, scale, seed):
    rng = np.random.default_rng(seed)
    feed = {
        "input_ids": np.asarray([[token_id]], dtype=np.int64),
        "position_ids": np.asarray([[past_len]], dtype=np.int64),
        "past_keys": (rng.standard_normal((LAYERS, 1, HEADS, past_len, HEAD_DIM)) * scale).astype(np.float32),
        "past_values": (rng.standard_normal((LAYERS, 1, HEADS, past_len, HEAD_DIM)) * scale).astype(np.float32),
    }
    logits = session.run(None, feed)[0]
    return logits


for scale in (0.0, 0.02, 0.5):
    a = run(2, 100, scale, seed=1)
    b = run(2, 100, scale, seed=2)
    print("scale", scale, "logit range", float(a.min()), float(a.max()), "cross-seed maxdiff", float(np.abs(a - b).max()))

zero = run(2, 100, 0.0, seed=1)
print("zero-KV argmax", int(np.argmax(zero)), "top logit", float(zero.max()))

int4 = ort.InferenceSession("N:/models/onnx/qwen3-asr-0.6b-official/decoder-step.int4.onnx", providers=["CPUExecutionProvider"])


def run_on(session, past_len, token_id, scale, seed):
    rng = np.random.default_rng(seed)
    feed = {
        "input_ids": np.asarray([[token_id]], dtype=np.int64),
        "position_ids": np.asarray([[past_len]], dtype=np.int64),
        "past_keys": (rng.standard_normal((LAYERS, 1, HEADS, past_len, HEAD_DIM)) * scale).astype(np.float32),
        "past_values": (rng.standard_normal((LAYERS, 1, HEADS, past_len, HEAD_DIM)) * scale).astype(np.float32),
    }
    return session.run(None, feed)[0]


for label, scale, seed in (("zerokv", 0.0, 1), ("small", 0.02, 1)):
    ref = run_on(session, 2, 9707, scale, seed)
    cand = run_on(int4, 2, 9707, scale, seed)
    diff = np.abs(ref - cand)
    print(label, "maxdiff", float(diff.max()), "meandiff", float(diff.mean()), "argmax", int(np.argmax(ref)), int(np.argmax(cand)))
