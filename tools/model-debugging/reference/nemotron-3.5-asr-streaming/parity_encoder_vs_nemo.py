#!/usr/bin/env python
"""Numerical encoder parity: NeMo conformer vs ONNX encoder.

NeMo conformer outputs [B, 1024, T_enc] where T_enc ≈ T_mel / 8.
The ONNX encoder graph folds the 1024->640 projection, producing [B, T_enc, 640].
The projection is in joint.enc (Linear(1024, 640)).

Parity: apply joint.enc to NeMo's [B, 1024, T_enc] to get [B, T_enc, 640],
compare element-wise with ONNX encoder [B, T_enc, 640] output.

Run with the isolated venv:
  .venv/Scripts/python.exe parity_encoder_vs_nemo.py \
    --nemo N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
    --onnx-dir N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-onnx \
    --fixture tools/data/fixtures/audio/jfk-short.wav \
    --output tools/data/results/nemotron/nemotron-3.5-encoder-parity-vs-nemo-2026-08-31.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
    EncDecRNNTBPEModelWithPrompt,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nemo", required=True)
    p.add_argument("--onnx-dir", required=True)
    p.add_argument("--fixture", action="append", required=True, dest="fixtures")
    p.add_argument("--output", required=True)
    p.add_argument("--prompt", default="en")
    return p.parse_args()


def load_wav(path: str, target_sr: int = 16000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.squeeze(0).numpy(), target_sr


def metrics(name, ref, target, atol=1e-4, rtol=1e-4):
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
        "allclose": bool(np.allclose(ref, target, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
    }


def main() -> None:
    args = parse_args()
    nemo_path = Path(args.nemo)
    onnx_dir = Path(args.onnx_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading NeMo model ...", flush=True)
    model = EncDecRNNTBPEModelWithPrompt.restore_from(
        restore_path=str(nemo_path), map_location="cpu"
    )
    model.eval()
    if hasattr(model, "set_inference_prompt"):
        model.set_inference_prompt(args.prompt)

    encoder = model.encoder
    preprocessor = model.preprocessor
    joint = model.joint

    print(f"Encoder: {type(encoder).__name__}", flush=True)
    print(f"Joint: {type(joint).__name__}", flush=True)

    # Find encoder projection in joint (1024->640)
    proj_layer = None
    proj_name = None
    if hasattr(joint, 'enc'):
        proj_layer = joint.enc
        proj_name = 'joint.enc'
    elif hasattr(joint, 'encoder_proj'):
        proj_layer = joint.encoder_proj
        proj_name = 'joint.encoder_proj'
    elif hasattr(joint, '_enc_projection'):
        proj_layer = joint._enc_projection
        proj_name = 'joint._enc_projection'
    else:
        for name, module in joint.named_modules():
            if isinstance(module, torch.nn.Linear):
                if module.in_features == 1024 and module.out_features == 640:
                    proj_layer = module
                    proj_name = f'joint.{name}'
                    break

    if proj_layer is not None:
        print(f"  Projection: {proj_name} {proj_layer.in_features}->{proj_layer.out_features}", flush=True)
    else:
        print("  WARNING: no 1024->640 projection found in joint!", flush=True)

    print("\nLoading ONNX sessions ...", flush=True)
    sess_first = ort.InferenceSession(
        str(onnx_dir / "encoder_320ms_first_fp16.onnx"),
        providers=["CPUExecutionProvider"]
    )
    sess_cont = ort.InferenceSession(
        str(onnx_dir / "encoder_320ms_fp16.onnx"),
        providers=["CPUExecutionProvider"]
    )
    print(f"  ONNX encoder first output: {sess_first.get_outputs()[0].name} "
          f"shape={sess_first.get_outputs()[0].shape}", flush=True)

    # Inspect ONNX encoder input shapes from metadata
    onnx_input_shapes = {}
    for inp in sess_first.get_inputs():
        onnx_input_shapes[inp.name] = inp.shape
    print(f"  ONNX input shapes: {onnx_input_shapes}", flush=True)

    # cache_mask expected shape
    cache_mask_shape = onnx_input_shapes.get("cache_mask", [1, 1, 1, 60])

    results = []
    ATOL = 1e-4
    RTOL = 1e-4

    for fixture_path in args.fixtures:
        fixture = Path(fixture_path)
        print(f"\nProcessing: {fixture.name}", flush=True)

        wav, sr = load_wav(str(fixture))
        audio_signal = torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]
        audio_length = torch.tensor([wav.shape[0]], dtype=torch.long)

        # Get mel features from NeMo preprocessor
        with torch.no_grad():
            processed = preprocessor(input_signal=audio_signal, length=audio_length)
        mel_features = processed[0]  # [B, C, T_mel]
        # ONNX encoder expects [B, T_mel, C]
        mel_BTC = mel_features.transpose(1, 2).numpy()  # [B, T_mel, C]
        print(f"  Mel: {mel_features.shape} -> ONNX: {mel_BTC.shape}", flush=True)

        # ---- NeMo: encoder + projection ----
        print("  Running NeMo encoder ...", flush=True)
        with torch.no_grad():
            enc_raw = encoder(audio_signal=mel_features, length=audio_length)
        if isinstance(enc_raw, tuple):
            enc_features = enc_raw[0]  # [B, 1024, T_enc]
        else:
            enc_features = enc_raw
        print(f"  NeMo enc raw: {enc_features.shape}", flush=True)

        enc_projected = None
        if proj_layer is not None:
            # enc_features: [B, 1024, T_enc]; joint.enc expects [*, 1024]
            # Transpose to [B, T_enc, 1024], project to [B, T_enc, 640]
            enc_proj_t = enc_features.transpose(1, 2)  # [B, T_enc, 1024]
            with torch.no_grad():
                enc_proj = proj_layer(enc_proj_t)  # [B, T_enc, 640]
            enc_proj = enc_proj.cpu().numpy()  # [B, T_enc, 640]
            enc_projected = enc_proj
            print(f"  NeMo projected: {enc_proj.shape}", flush=True)

        # ---- ONNX: first chunk ----
        print("  Running ONNX encoder (first chunk, 25 mel frames) ...", flush=True)
        chunk_first = mel_BTC[:, :25, :]  # [B, 25, C]
        onnx_inp_first = {
            "input_features": chunk_first.astype(np.float32),
            "prompt_ids": np.array([0], dtype=np.int64),  # [1] not [1,1]
            "cache_mask": np.ones(cache_mask_shape, dtype=np.float32),
        }
        for inp in sess_first.get_inputs():
            if inp.name in onnx_inp_first:
                continue
            shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape]
            dtype = np.int64 if "int64" in inp.type else np.float32
            onnx_inp_first[inp.name] = np.zeros(shape, dtype=dtype)
        onnx_out_first = sess_first.run(None, onnx_inp_first)
        onnx_enc_first = onnx_out_first[0]  # [B, T_enc, 640]
        print(f"  ONNX first: {onnx_enc_first.shape}", flush=True)

        record = {
            "fixture": fixture.as_posix(),
            "fixtureName": fixture.name,
            "nemoRawShape": list(enc_features.shape),
            "nemoProjectedShape": list(enc_projected.shape) if enc_projected is not None else None,
            "onnxOutputShape": list(onnx_enc_first.shape),
            "projectionLayer": proj_name,
            "metrics": [],
        }

        if enc_projected is not None:
            # Compare first 4 time steps (ONNX output for 25 mel frames)
            T_onnx = onnx_enc_first.shape[1]
            nemo_slice = enc_projected[:, :T_onnx, :]
            m1 = metrics("encoder.projected.first_chunk", nemo_slice, onnx_enc_first, atol=ATOL, rtol=RTOL)
            record["metrics"].append(m1)
            print(f"  First chunk: maxAbsErr={m1['maxAbsErr']:.3e}, cosSim={m1['cosineSim']:.8f}, allclose={m1['allclose']}", flush=True)

            # Continuation chunk (32 frames, starting at frame 25)
            print("  Running ONNX encoder (continuation, frames 25-57) ...", flush=True)
            chunk_cont = mel_BTC[:, 25:57, :]  # [B, 32, C]
            onnx_inp_cont = {
                "input_features": chunk_cont.astype(np.float32),
                "prompt_ids": np.array([0], dtype=np.int64),  # [1] not [1,1]
                "cache_mask": np.ones(cache_mask_shape, dtype=np.float32),
            }
            for inp in sess_cont.get_inputs():
                if inp.name in onnx_inp_cont:
                    continue
                shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape]
                dtype = np.int64 if "int64" in inp.type else np.float32
                onnx_inp_cont[inp.name] = np.zeros(shape, dtype=dtype)
            onnx_out_cont = sess_cont.run(None, onnx_inp_cont)
            onnx_enc_cont = onnx_out_cont[0]  # [B, T_enc, 640]
            print(f"  ONNX cont: {onnx_enc_cont.shape}", flush=True)

            # NeMo continuation: frames 3:3+T_onnx_cont (skip first 3 frames from first 25)
            T_onnx_cont = onnx_enc_cont.shape[1]
            nemo_slice_cont = enc_projected[:, 3:3+T_onnx_cont, :]
            m2 = metrics("encoder.projected.continuation", nemo_slice_cont, onnx_enc_cont, atol=ATOL, rtol=RTOL)
            record["metrics"].append(m2)
            print(f"  Continuation: maxAbsErr={m2['maxAbsErr']:.3e}, cosSim={m2['cosineSim']:.8f}, allclose={m2['allclose']}", flush=True)
        else:
            record["note"] = "No 1024->640 projection found; cannot compute numerical parity"

        results.append(record)

    overall = {
        "allMetricsAllclose": all(m["allclose"] for r in results for m in r["metrics"]),
        "anyMetrics": len(results) > 0 and len(results[0]["metrics"]) > 0,
    }

    output_record = {
        "schemaVersion": 1,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "NeMo conformer vs ONNX encoder numerical parity (projected outputs)",
        "atol": ATOL,
        "rtol": RTOL,
        "results": results,
        "overall": overall,
    }

    output_path.write_text(
        json.dumps(output_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {output_path}", flush=True)
    print(f"OVERALL allclose: {overall['allMetricsAllclose']}", flush=True)


if __name__ == "__main__":
    main()
