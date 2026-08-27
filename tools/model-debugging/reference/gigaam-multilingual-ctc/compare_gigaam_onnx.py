"""Compare official GigaAM PyTorch logits/text against native ONNX Runtime.

Uses the official exported graph and the same preprocessor that produced the
PyTorch reference. This is native ORT parity, not a WASM/WebGPU claim.
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
    parser = argparse.ArgumentParser(
        description="Compare official GigaAM CTC PyTorch vs native ORT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path(r"N:\models\onnx\gigaam\multilingual-ctc"),
    )
    parser.add_argument("--model-name", default="multilingual_ctc")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--abs-tolerance", type=float, default=1e-4)
    parser.add_argument("--rel-tolerance", type=float, default=1e-4)
    return parser.parse_args()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1).astype(np.float64)
    right_flat = right.reshape(-1).astype(np.float64)
    count = min(left_flat.size, right_flat.size)
    if count == 0:
        return 0.0
    a = left_flat[:count]
    b = right_flat[:count]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def compare_arrays(reference: np.ndarray, candidate: np.ndarray, abs_tol: float, rel_tol: float) -> dict[str, Any]:
    count = min(reference.size, candidate.size)
    ref = reference.reshape(-1)[:count]
    cand = candidate.reshape(-1)[:count]
    diff = np.abs(ref - cand)
    allowed = abs_tol + rel_tol * np.maximum(np.abs(ref), np.abs(cand))
    mismatches = int(np.count_nonzero(diff > allowed))
    return {
        "shape": {"reference": list(reference.shape), "candidate": list(candidate.shape)},
        "max_abs": float(diff.max()) if count else 0.0,
        "mean_abs": float(diff.mean()) if count else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if count else 0.0,
        "cosine": cosine(reference, candidate),
        "mismatches": mismatches,
        "count": count,
        "pass": mismatches == 0 and list(reference.shape) == list(candidate.shape),
    }


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


def main() -> int:
    args = parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    onnx_path = args.onnx_dir / f"{args.model_name}.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    session = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    vocab = list(reference["tokenizer"]["vocab"])
    blank_id = int(reference["tokenizer"]["blank_id"])
    rows = []
    all_pass = True
    for sample in reference["samples"]:
        features_path = sample["stages"]["features"]["npy"]
        log_probs_path = sample["stages"]["log_probs"]["npy"]
        features = np.load(features_path)
        reference_logits = np.load(log_probs_path)
        lengths = np.asarray(sample["stages"]["features"]["lengths"], dtype=np.int64)
        outputs = session.run(
            ["log_probs", "encoded_lengths"],
            {
                "features": features.astype(np.float32),
                "feature_lengths": lengths,
            },
        )
        onnx_logits = np.asarray(outputs[0])
        onnx_lengths = np.asarray(outputs[1]).reshape(-1)
        logits_cmp = compare_arrays(
            reference_logits,
            onnx_logits,
            args.abs_tolerance,
            args.rel_tolerance,
        )
        onnx_text, onnx_ids = greedy_ctc(onnx_logits[0], int(onnx_lengths[0]), blank_id, vocab)
        pytorch_text = sample["text"]
        text_match = onnx_text == pytorch_text
        row = {
            "sample_id": sample["sample_id"],
            "audio_sha256": sample["audio"]["sha256"],
            "pytorch_text": pytorch_text,
            "onnx_text": onnx_text,
            "text_match": text_match,
            "token_ids": {
                "reference": sample["token_ids"],
                "onnx": onnx_ids,
                "match": onnx_ids == sample["token_ids"],
            },
            "encoded_lengths": {
                "reference": sample["stages"]["encoded"]["lengths"],
                "onnx": [int(value) for value in onnx_lengths.tolist()],
            },
            "log_probs": logits_cmp,
        }
        rows.append(row)
        if not text_match or not row["token_ids"]["match"]:
            all_pass = False

    payload = {
        "schema_version": 1,
        "engine": "official-gigaam-onnxruntime-cpu",
        "onnx_path": str(onnx_path),
        "providers": session.get_providers(),
        "abs_tolerance": args.abs_tolerance,
        "rel_tolerance": args.rel_tolerance,
        "samples": rows,
        "pass": all_pass and all(row["text_match"] for row in rows),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        sys.stdout.write(text)
    print(f"native ORT text match: {payload['pass']}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
