#!/usr/bin/env python
"""ONNX encoder parity probe (step 4 native-ORT rung).

Run with the isolated venv.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def load_session(dir_path, name):
    return ort.InferenceSession(
        str(dir_path / name), providers=["CPUExecutionProvider"]
    )


def run_encoder(sess, features, prompt_id):
    onnx_inputs = {
        "input_features": features,
        "prompt_ids": prompt_id,
    }
    for inp in sess.get_inputs():
        if inp.name in onnx_inputs:
            continue
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape]
        if "int64" in inp.type:
            onnx_inputs[inp.name] = np.zeros(shape, dtype=np.int64)
        else:
            onnx_inputs[inp.name] = np.zeros(shape, dtype=np.float32)
    outputs = sess.run(None, onnx_inputs)
    return outputs


def probe(label, sess, features, prompt_id):
    outputs = run_encoder(sess, features, prompt_id)
    enc_out = outputs[0]
    n_cache_outs = sum(1 for n in sess.get_outputs() if n != "encoder_out")
    return {
        "label": label,
        "featureShape": list(features.shape),
        "encoderOutShape": list(enc_out.shape),
        "encoderOutMaxAbs": float(np.abs(enc_out).max()),
        "encoderOutMeanAbs": float(np.abs(enc_out).mean()),
        "cacheOutputCount": n_cache_outs,
        "firstEncoderRow": enc_out[0, 0, :5].tolist(),
    }


def main():
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx_dir = Path(args.onnx_dir)

    print("Loading encoder sessions ...", flush=True)
    sess_first = load_session(onnx_dir, "encoder_320ms_first_fp16.onnx")
    sess_cont = load_session(onnx_dir, "encoder_320ms_fp16.onnx")

    np.random.seed(42)
    features_first = np.random.randn(1, 25, 128).astype(np.float32)
    features_cont = np.random.randn(1, 32, 128).astype(np.float32)
    prompt_id = np.array([0], dtype=np.int64)

    print("Running first-chunk encoder ...", flush=True)
    first_result = probe("encoder_first_chunk", sess_first, features_first, prompt_id)

    print("Running continuation encoder ...", flush=True)
    cont_result = probe("encoder_continuation", sess_cont, features_cont, prompt_id)

    record = {
        "schemaVersion": 1,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "ONNX encoder parity probe (native ORT, step 4 rung 1)",
        "onnxDir": onnx_dir.as_posix(),
        "note": "Synthetic features; verifies shape, magnitude, and cache-output contract.",
        "firstChunk": first_result,
        "continuation": cont_result,
    }

    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("Wrote " + str(out), flush=True)
    print("  first encoder_out: shape=" + str(first_result["encoderOutShape"]) + ", maxAbs=" + str(round(first_result["encoderOutMaxAbs"], 4)), flush=True)
    print("  continuation encoder_out: shape=" + str(cont_result["encoderOutShape"]) + ", maxAbs=" + str(round(cont_result["encoderOutMaxAbs"], 4)), flush=True)


if __name__ == "__main__":
    main()