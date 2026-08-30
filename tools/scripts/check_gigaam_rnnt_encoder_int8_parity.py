#!/usr/bin/env python
"""Compare a GigaAM RNN-T encoder variant with the fp32 reference.

The input is a captured [1, 64, frames] feature tensor (the official
GigaAM frontend layout).  This check intentionally runs on ORT CPU so the
numerical comparison is independent of browser execution-provider support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
import onnxruntime as ort


DEFAULT_REFERENCE = pathlib.Path(
    "N:/models/onnx/gigaam/v3-e2e-rnnt/v3_e2e_rnnt_encoder.onnx"
)
DEFAULT_CANDIDATE = pathlib.Path(
    "N:/models/onnx/gigaam/v3-e2e-rnnt/v3_e2e_rnnt_encoder.int8.onnx"
)
DEFAULT_FEATURES = pathlib.Path(
    "N:/models/gigaam/v3-e2e-rnnt/captures/example.features.npy"
)


def sha256(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes()).hexdigest()


def run(path: pathlib.Path, features: np.ndarray, length: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    session = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    started = time.perf_counter()
    outputs = session.run(None, {"audio_signal": features, "length": length})
    elapsed_ms = (time.perf_counter() - started) * 1000
    return outputs[0], outputs[1], elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=pathlib.Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--candidate", type=pathlib.Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--features", type=pathlib.Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    features = np.asarray(np.load(args.features), dtype=np.float32)
    if features.ndim == 2:
        features = features[None, ...]
    if features.shape[0] != 1 or features.shape[1] != 64:
        raise ValueError(f"expected [1,64,frames] features, got {features.shape}")
    length = np.asarray([features.shape[2]], dtype=np.int64)
    reference, reference_len, reference_ms = run(args.reference, features, length)
    candidate, candidate_len, candidate_ms = run(args.candidate, features, length)
    diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    dot = float(np.sum(reference.astype(np.float64) * candidate.astype(np.float64)))
    norm = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
    result = {
        "schema": "asrjs.gigaam-rnnt.encoder-int8-parity.v1",
        "reference": args.reference.as_posix(),
        "candidate": args.candidate.as_posix(),
        "features": args.features.as_posix(),
        "featureShape": list(features.shape),
        "featureSha256": sha256(features),
        "referenceShape": list(reference.shape),
        "candidateShape": list(candidate.shape),
        "referenceLength": reference_len.tolist(),
        "candidateLength": candidate_len.tolist(),
        "maxAbsDiff": float(np.max(diff)),
        "meanAbsDiff": float(np.mean(diff)),
        "cosine": dot / norm if norm else 0.0,
        "referenceMs": reference_ms,
        "candidateMs": candidate_ms,
        "exactLength": bool(np.array_equal(reference_len, candidate_len)),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
