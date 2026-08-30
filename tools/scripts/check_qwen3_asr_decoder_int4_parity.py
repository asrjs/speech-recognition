#!/usr/bin/env python
"""One-step CPU parity probe: Qwen3-ASR decoder-step fp32 vs INT4.

Builds a synthetic past_len=2 step feed matching the official graph
contract, runs both graphs on ORT CPU, and reports logits agreement.
Synthetic inputs cannot prove transcription quality; the browser
exact-token gate remains the promotion authority.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import onnxruntime as ort


BASE = pathlib.Path("N:/models/onnx/qwen3-asr-0.6b-official")
LAYERS, HEADS, HEAD_DIM = 28, 8, 128


def make_feed(past_len: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    # Small magnitudes matter: logits are residuals around the embedding
    # lookup, so realistic attention/KV values are near zero, not unit
    # normal. The probe uses a fixed token id and tiny KV noise.
    return {
        "input_ids": np.asarray([[9707]], dtype=np.int64),
        "position_ids": np.asarray([[past_len]], dtype=np.int64),
        "past_keys": np.zeros((LAYERS, 1, HEADS, past_len, HEAD_DIM), dtype=np.float32),
        "past_values": np.zeros((LAYERS, 1, HEADS, past_len, HEAD_DIM), dtype=np.float32),
    }


def run(path: pathlib.Path, feed: dict) -> tuple[np.ndarray, float]:
    session = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    started = time.perf_counter()
    outputs = session.run(None, feed)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return outputs[0], elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=pathlib.Path, default=BASE / "decoder-step.onnx")
    parser.add_argument("--candidate", type=pathlib.Path, default=BASE / "decoder-step.int4.onnx")
    parser.add_argument("--past-len", type=int, default=2)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    feed = make_feed(args.past_len, seed=20260830)
    reference, reference_ms = run(args.reference, feed)
    candidate, candidate_ms = run(args.candidate, feed)
    diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    dot = float(np.sum(reference.astype(np.float64) * candidate.astype(np.float64)))
    norm = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
    top5_ref = np.argsort(reference[0])[::-1][:5].tolist()
    top5_cand = np.argsort(candidate[0])[::-1][:5].tolist()
    result = {
        "schema": "asrjs.qwen.decoder-int4-parity.v1",
        "reference": args.reference.as_posix(),
        "candidate": args.candidate.as_posix(),
        "pastLen": args.past_len,
        "logitsShape": list(reference.shape),
        "maxAbsDiff": float(np.max(diff)),
        "meanAbsDiff": float(np.mean(diff)),
        "cosine": dot / norm if norm else 0.0,
        "top5Reference": top5_ref,
        "top5Candidate": top5_cand,
        "argmaxEqual": bool(np.argmax(reference) == np.argmax(candidate)),
        "referenceMs": reference_ms,
        "candidateMs": candidate_ms,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
