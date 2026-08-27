"""Run official GigaAM CTC ONNX with JS frontend features.

This is the JS-features → native ORT text gate. It is not a WASM/WebGPU claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as rt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--js-features", type=Path, required=True)
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path(r"N:\models\onnx\gigaam\multilingual-ctc"),
    )
    parser.add_argument("--model-name", default="multilingual_ctc")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def greedy_ctc(log_probs: np.ndarray, length: int, blank_id: int, vocab: list[str]) -> tuple[str, list[int]]:
    labels = log_probs.argmax(axis=-1)[:length]
    token_ids: list[int] = []
    previous = None
    for label in labels.tolist():
        if label == blank_id or label == previous:
            previous = label
            continue
        token_ids.append(int(label))
        previous = label
    text = "".join(vocab[index] for index in token_ids if 0 <= index < len(vocab))
    return text, token_ids


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.reshape(-1).astype(np.float64)
    b = right.reshape(-1).astype(np.float64)
    count = min(a.size, b.size)
    if count == 0:
        return 0.0
    a = a[:count]
    b = b[:count]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    count = min(reference.size, candidate.size)
    ref = reference.reshape(-1)[:count]
    cand = candidate.reshape(-1)[:count]
    diff = np.abs(ref - cand)
    return {
        "shape": {"reference": list(reference.shape), "candidate": list(candidate.shape)},
        "max_abs": float(diff.max()) if count else 0.0,
        "mean_abs": float(diff.mean()) if count else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if count else 0.0,
        "cosine": cosine(reference, candidate),
        "count": count,
    }


def main() -> int:
    args = parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    onnx_path = args.onnx_dir / f"{args.model_name}.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if not args.js_features.is_file():
        raise FileNotFoundError(args.js_features)

    session = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    vocab = list(reference["tokenizer"]["vocab"])
    blank_id = int(reference["tokenizer"]["blank_id"])
    sample = reference["samples"][0]
    js_features = np.load(args.js_features)
    if js_features.ndim == 2:
        js_features = js_features[np.newaxis, ...]
    lengths = np.asarray([js_features.shape[-1]], dtype=np.int64)
    outputs = session.run(
        ["log_probs", "encoded_lengths"],
        {
            "features": js_features.astype(np.float32),
            "feature_lengths": lengths,
        },
    )
    onnx_logits = np.asarray(outputs[0])
    onnx_lengths = np.asarray(outputs[1]).reshape(-1)
    official_logits = np.load(sample["stages"]["log_probs"]["npy"])
    text, token_ids = greedy_ctc(onnx_logits[0], int(onnx_lengths[0]), blank_id, vocab)
    pytorch_text = sample["text"]
    payload = {
        "schema_version": 1,
        "engine": "js-frontend-official-gigaam-onnxruntime-cpu",
        "onnx_path": str(onnx_path),
        "js_features_path": str(args.js_features),
        "js_features_shape": list(js_features.shape),
        "pytorch_text": pytorch_text,
        "js_onnx_text": text,
        "text_match": text == pytorch_text,
        "token_ids": {
            "reference": sample["token_ids"],
            "js_onnx": token_ids,
            "match": token_ids == sample["token_ids"],
        },
        "log_probs_vs_pytorch": compare_arrays(official_logits, onnx_logits),
        "status": "experimental-js-frontend-native-ort",
    }
    text_out = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text_out, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        sys.stdout.write(text_out)
    print(f"JS features → native ORT text match: {payload['text_match']}")
    return 0 if payload["text_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
