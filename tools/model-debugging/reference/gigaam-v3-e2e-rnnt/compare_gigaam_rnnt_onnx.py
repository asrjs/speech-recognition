"""Compare official GigaAM v3 E2E RNN-T PyTorch vs native ORT greedy text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as rt

GIGAAM_SRC = Path(r"N:\github\salute-developers\GigaAM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare official GigaAM RNN-T PyTorch vs native ORT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, default=Path(r"N:\models\onnx\gigaam\v3-e2e-rnnt"))
    parser.add_argument("--model-name", default="v3_e2e_rnnt")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--abs-tolerance", type=float, default=1e-4)
    return parser.parse_args()


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


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
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
    sys.path.insert(0, str(GIGAAM_SRC))
    from omegaconf import OmegaConf
    from gigaam.onnx_utils import infer_onnx, load_onnx

    sessions, model_cfg = load_onnx(str(args.onnx_dir), args.model_name, provider="CPUExecutionProvider")
    enc_sess = sessions[0]
    rows = []
    all_pass = True
    for sample in reference["samples"]:
        features = np.load(sample["stages"]["features"]["npy"])
        encoded_ref = np.load(sample["stages"]["encoded"]["npy"])
        lengths = np.asarray(sample["stages"]["features"]["lengths"], dtype=np.int64)
        enc_out = enc_sess.run(
            [node.name for node in enc_sess.get_outputs()],
            {
                "audio_signal": features.astype(np.float32),
                "length": lengths,
            },
        )
        encoded = np.asarray(enc_out[0])
        encoded_len = np.asarray(enc_out[1]).reshape(-1)
        enc_cmp = compare_arrays(encoded_ref, encoded)
        texts = infer_onnx(
            [sample["audio"]["path"]],
            model_cfg,
            sessions,
            batch_size=1,
            progress=False,
        )
        onnx_text = texts[0] if texts else ""
        pytorch_text = sample["text"]
        text_match = onnx_text == pytorch_text
        row = {
            "sample_id": sample["sample_id"],
            "pytorch_text": pytorch_text,
            "onnx_text": onnx_text,
            "text_match": text_match,
            "encoded": enc_cmp,
            "encoded_lengths": {
                "reference": sample["stages"]["encoded"]["lengths"],
                "onnx": [int(value) for value in encoded_len.tolist()],
            },
        }
        rows.append(row)
        if not text_match:
            all_pass = False

    payload = {
        "schema_version": 1,
        "engine": "official-gigaam-onnxruntime-cpu",
        "onnx_dir": str(args.onnx_dir),
        "providers": enc_sess.get_providers(),
        "samples": rows,
        "pass": all_pass,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
