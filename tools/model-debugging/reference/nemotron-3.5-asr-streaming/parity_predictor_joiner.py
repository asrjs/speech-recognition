#!/usr/bin/env python
"""Numerical parity: NeMo (PyTorch) vs ONNX (native ORT) for predictor and joiner.

API contract verified from source (nemo/collections/asr/modules/rnnt.py):
- decoder.predict(y=(B,U) int64, state=(h,c) each [L,B,H], add_sos=False)
    -> g (B,U,640), hid=(h_out, c_out) each [L,B,H]
- joint.joint(f=[B,T,640], g=[B,U,640]) -> logits (B,T,U,V+1)

Run with the isolated venv:
  .venv/Scripts/python.exe parity_predictor_joiner.py \
    --nemo N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
    --onnx-dir N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-onnx \
    --output tools/data/results/nemotron/nemotron-3.5-predictor-joiner-parity-2026-08-31.json
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
    EncDecRNNTBPEModelWithPrompt,
)

ATOL = 1e-4
RTOL = 1e-4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nemo", required=True)
    p.add_argument("--onnx-dir", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def metrics(name, ref, target):
    ref = np.asarray(ref, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    diff = np.abs(ref - target)
    denom = np.linalg.norm(ref) * np.linalg.norm(target)
    cos = float(np.dot(ref.flatten(), target.flatten()) / denom) if denom > 1e-12 else 0.0
    return {
        "component": name,
        "refShape": list(ref.shape),
        "targetShape": list(target.shape),
        "maxAbsErr": float(diff.max()),
        "meanAbsErr": float(diff.mean()),
        "cosineSim": cos,
        "allclose": bool(np.allclose(ref, target, atol=ATOL, rtol=RTOL)),
        "atol": ATOL,
        "rtol": RTOL,
        "refSample": ref.flatten()[:4].tolist(),
        "targetSample": target.flatten()[:4].tolist(),
    }


def main():
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx_dir = Path(args.onnx_dir)

    print("Restoring NeMo model ...", flush=True)
    model = EncDecRNNTBPEModelWithPrompt.restore_from(
        restore_path=args.nemo, map_location="cpu"
    )
    model.eval()
    if hasattr(model, "set_inference_prompt"):
        model.set_inference_prompt("en")

    # Identical inputs for both implementations (shared RNG seed)
    rng = np.random.default_rng(1234)
    token_np = np.array([[7]], dtype=np.int64)
    h_np = rng.standard_normal((2, 1, 640)).astype(np.float32)
    c_np = rng.standard_normal((2, 1, 640)).astype(np.float32)
    enc_np = rng.standard_normal((1, 1, 640)).astype(np.float32)

    # ---- Predictor: NeMo ----
    with torch.no_grad():
        y = torch.from_numpy(token_np)
        h_t = torch.from_numpy(h_np)
        c_t = torch.from_numpy(c_np)
        g, hid = model.decoder.predict(y=y, state=(h_t, c_t), add_sos=False)
        g = g.cpu().numpy()
        h_out, c_out = hid[0].cpu().numpy(), hid[1].cpu().numpy()

    # ---- Predictor: ONNX ----
    sess_dec = ort.InferenceSession(
        str(onnx_dir / "decoder.onnx"), providers=["CPUExecutionProvider"]
    )
    dec_outs = sess_dec.run(
        None,
        {"token": token_np, "h_in": h_np, "c_in": c_np},
    )
    onnx_g, onnx_h, onnx_c = dec_outs[0], dec_outs[1], dec_outs[2]

    results = [
        metrics("predictor.decoder_out", g.squeeze(), onnx_g.squeeze()),
        metrics("predictor.h_out", h_out, onnx_h),
        metrics("predictor.c_out", c_out, onnx_c),
    ]

    # ---- Joiner: NeMo (raw logits) ----
    joint = model.joint
    if hasattr(joint, "log_softmax"):
        joint.log_softmax = False
    dec_frame = torch.from_numpy(onnx_g).reshape(1, 1, 640)
    enc_frame = torch.from_numpy(enc_np)
    with torch.no_grad():
        logits = joint.joint_after_projection(f=enc_frame, g=dec_frame)
    logits = logits.cpu().numpy().reshape(-1)

    # ---- Joiner: ONNX ----
    sess_join = ort.InferenceSession(
        str(onnx_dir / "joiner.onnx"), providers=["CPUExecutionProvider"]
    )
    onnx_logits = sess_join.run(
        None,
        {
            "encoder_frame": enc_np.reshape(1, 640),
            "decoder_out": onnx_g.reshape(1, 640),
        },
    )[0].reshape(-1)

    jm = metrics("joiner.logits", logits, onnx_logits)
    jm["refArgmax"] = int(np.argmax(logits))
    jm["targetArgmax"] = int(np.argmax(onnx_logits))
    jm["argmaxAgree"] = jm["refArgmax"] == jm["targetArgmax"]
    results.append(jm)

    record = {
        "schemaVersion": 1,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "numerical parity: NeMo vs ONNX for predictor and joiner (native ORT)",
        "onnxDir": onnx_dir.as_posix(),
        "nemoCheckpoint": Path(args.nemo).as_posix(),
        "inputs": {
            "tokenId": int(token_np[0][0]),
            "note": "h/c/encoder frames from numpy default_rng(1234), shared by both sides",
        },
        "results": results,
        "overall": {
            "allComponentsAllclose": all(r["allclose"] for r in results),
            "argmaxAgree": jm["argmaxAgree"],
        },
    }
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("Wrote " + str(out), flush=True)
    for r in results:
        print(
            "  " + r["component"]
            + ": maxAbs=" + format(r["maxAbsErr"], ".3e")
            + ", cosSim=" + format(r["cosineSim"], ".6f")
            + ", allclose=" + str(r["allclose"]),
            flush=True,
        )
    print("  argmaxAgree: " + str(jm["argmaxAgree"]), flush=True)


if __name__ == "__main__":
    main()
