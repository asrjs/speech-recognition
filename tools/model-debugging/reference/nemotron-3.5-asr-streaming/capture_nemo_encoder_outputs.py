#!/usr/bin/env python
"""Capture NeMo conformer encoder outputs for numerical parity with the ONNX export.

NeMo encoder signature (confirmed from inspection):
  forward(audio_signal, length, cache_last_channel=None, cache_last_time=None,
          cache_last_channel_len=None, bypass_pre_encode=False)

It takes raw waveform [B, samples] and produces conformer outputs [B, T, H].
The mel transform runs inside the encoder. The ONNX encoder accepts mel features
directly because mel filtering is folded into the exported graph.

For numerical parity: feed raw audio to NeMo encoder and capture the conformer
output tensor. This is the ground truth to compare against ONNX encoder output.

Run with the isolated venv:
  .venv/Scripts/python.exe capture_nemo_encoder_outputs.py \
    --nemo N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
    --fixture tools/data/fixtures/audio/jfk-short.wav \
    --fixture tools/data/fixtures/audio/librivox-blankgaps-synthetic.wav \
    --output tools/data/results/nemotron/nemotron-3.5-nemo-encoder-outputs-2026-08-31.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
    EncDecRNNTBPEModelWithPrompt,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nemo", required=True, help="Path to .nemo checkpoint")
    p.add_argument("--fixture", action="append", required=True, dest="fixtures", help="WAV fixture(s)")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--device", default="cpu")
    p.add_argument("--prompt", default="en")
    return p.parse_args()


def load_wav(path: str, target_sr: int = 16000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    # Return [1, T] for the model
    return wav, target_sr


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Restoring {args.nemo} on {args.device} ...", flush=True)
    sys.stdout.flush()
    model = EncDecRNNTBPEModelWithPrompt.restore_from(
        restore_path=args.nemo, map_location=torch.device(args.device)
    )
    model.eval()
    print(f"Model restored: {type(model).__name__}", flush=True)
    print(f"Encoder type: {type(model.encoder).__name__}", flush=True)

    has_prompt_api = hasattr(model, "set_inference_prompt")
    if has_prompt_api:
        model.set_inference_prompt(args.prompt)
        print(f"Inference prompt set: {args.prompt}", flush=True)

    records = []
    for fixture_path in args.fixtures:
        fixture = Path(fixture_path)
        print(f"\nProcessing: {fixture.name}", flush=True)

        wav, sr = load_wav(str(fixture))
        # wav: [1, T]
        B, T = wav.shape
        audio_signal = wav  # [B, T]
        audio_length = torch.tensor([T], dtype=torch.long)
        print(f"  Audio: shape={wav.shape}, sr={sr} Hz ({T/sr:.2f}s)", flush=True)

        # Also get mel features for reference (what ONNX encoder consumes)
        preprocessor = model.preprocessor
        with torch.no_grad():
            processed = preprocessor(
                input_signal=audio_signal,
                length=audio_length,
            )
        mel_features = processed[0]  # [B, C, T]
        # ONNX encoder expects [B, T, C] — transpose
        mel_T_C = mel_features.transpose(1, 2).squeeze(0).numpy()  # [T, C]
        print(f"  Mel features (preprocessor): shape={mel_T_C.shape}", flush=True)
        # First chunk [1, 25, 128]
        first_chunk = mel_T_C[:25, :]  # [25, 128]
        print(f"  First chunk [1,25,128]: shape={first_chunk.shape}, maxAbs={float(np.abs(first_chunk).max()):.4f}", flush=True)

        # Run encoder forward with raw audio
        captured = {}

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    captured[name] = output.detach().cpu().numpy()
                elif isinstance(output, tuple):
                    captured[name] = tuple(
                        o.detach().cpu().numpy() if isinstance(o, torch.Tensor) else o
                        for o in output
                    )
            return hook

        # Hook the final encoder output
        # The ConformerEncoder returns encoded features + cache in a tuple
        # Let's register on the root encoder to capture final output
        h = model.encoder.register_forward_hook(
            lambda m, i, o: captured.__setitem__("encoder_forward_output",
                o.detach().cpu().numpy() if isinstance(o, torch.Tensor) else
                tuple(x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in o)
            )
        )

        try:
            with torch.no_grad():
                # The encoder expects [B, C, T] mel features (preprocessed),
                # NOT raw audio. audio_signal=[1,T] was wrong.
                enc_out = model.encoder(
                    audio_signal=mel_features,  # [B, C, T] from preprocessor
                    length=audio_length,
                )
            print(f"  Encoder output: type={type(enc_out)}", flush=True)
            if isinstance(enc_out, torch.Tensor):
                print(f"  Shape: {enc_out.shape}", flush=True)
                enc_out_np = enc_out.detach().cpu().numpy()
            elif isinstance(enc_out, tuple):
                enc_out_np = tuple(
                    x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                    for x in enc_out
                )
                shapes = [x.shape if isinstance(x, np.ndarray) else type(x).__name__
                          for x in enc_out_np]
                print(f"  Tuple shapes: {shapes}", flush=True)
            else:
                enc_out_np = enc_out
        finally:
            h.remove()

        # Build record
        record = {
            "fixture": fixture.as_posix(),
            "fixtureName": fixture.name,
            "audioSamples": int(T),
            "sampleRate": sr,
            "melShape": list(mel_T_C.shape),
            "melFirstChunk": first_chunk.tolist(),
            "melFirstChunkMaxAbs": float(np.abs(first_chunk).max()),
            "melFirstChunkMeanAbs": float(np.abs(first_chunk).mean()),
            "encoderOutputType": type(enc_out).__name__,
            "encoderOutput": None,
            "encoderOutputShape": None,
            "encoderOutputFirstRows": None,
            "encoderForwardHookOutput": None,
        }

        if isinstance(enc_out_np, np.ndarray):
            record["encoderOutputShape"] = list(enc_out_np.shape)
            record["encoderOutput"] = "single_tensor"
            if enc_out_np.ndim == 3:
                record["encoderOutputFirstRows"] = enc_out_np[0, :4, :5].tolist()
            elif enc_out_np.ndim == 2:
                record["encoderOutputFirstRows"] = enc_out_np[:4, :5].tolist()
        elif isinstance(enc_out_np, tuple):
            record["encoderOutput"] = "tuple"
            shapes = []
            first_tensor = None
            for i, x in enumerate(enc_out_np):
                if isinstance(x, np.ndarray):
                    shapes.append(list(x.shape))
                    if first_tensor is None:
                        first_tensor = x
                else:
                    shapes.append(str(type(x).__name__))
            record["encoderOutputShape"] = shapes
            if first_tensor is not None and first_tensor.ndim == 3:
                record["encoderOutputFirstRows"] = first_tensor[0, :4, :5].tolist()

        # Hook output
        if "encoder_forward_output" in captured:
            hout = captured["encoder_forward_output"]
            if isinstance(hout, np.ndarray):
                record["encoderForwardHookOutput"] = {
                    "type": "ndarray",
                    "shape": list(hout.shape),
                    "maxAbs": float(np.abs(hout).max()),
                    "meanAbs": float(np.abs(hout).mean()),
                    "firstRows": hout[0, :4, :5].tolist() if hout.ndim == 3 else None,
                }
            elif isinstance(hout, tuple):
                record["encoderForwardHookOutput"] = {
                    "type": "tuple",
                    "shapes": [list(x.shape) if isinstance(x, np.ndarray) else str(type(x).__name__)
                               for x in hout],
                }

        print(f"  encoder output: {record['encoderOutput']} {record.get('encoderOutputShape')}", flush=True)
        if record.get("encoderForwardHookOutput"):
            print(f"  hook output: {record['encoderForwardHookOutput']}", flush=True)

        records.append(record)

    output_record = {
        "schemaVersion": 1,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "NeMo encoder outputs for ONNX encoder numerical parity",
        "modelType": type(model).__name__,
        "encoderType": type(model.encoder).__name__,
        "records": records,
    }

    output_path.write_text(json.dumps(output_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
