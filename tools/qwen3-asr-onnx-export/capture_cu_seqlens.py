"""Capture the Qwen3-ASR window schedule from the encoder itself.

The schedule must not be reconstructed: the encoder derives chunk lengths by
reshaping the feature mask into chunks, so a reconstruction coincides with the
truth at one length and drifts by a few frames at others. This runs a real
forward with the encoder's own function intercepted, and records what it returns.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.qwen3_asr import modeling_qwen3_asr as modeling


def capture_schedule(model, features, mask=None):
    """Return the window schedule the encoder computes for these features.

    The schedule has to be taken from the encoder's own function, because
    reconstructing it disagrees by a few frames.
    """
    captured = []
    original = modeling.get_audio_cu_seqlens

    def spy(*call_args, **call_kwargs):
        result = original(*call_args, **call_kwargs)
        if not call_kwargs.get("cu_seqlens") is not None:
            captured.append(result.detach().clone())
        return result

    modeling.get_audio_cu_seqlens = spy
    try:
        with torch.no_grad():
            if mask is None:
                mask = torch.ones(features.shape[0], features.shape[-1], dtype=features.dtype)
            model.model.multi_modal_projector(
                model.model.audio_tower(features, mask).last_hidden_state
            )
    finally:
        modeling.get_audio_cu_seqlens = original
    if not captured:
        raise RuntimeError("the encoder did not compute a schedule")
    return captured[0].to(torch.int32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/qwen3-asr-0.6b"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/onnx/qwen3-asr-0.6b"))
    parser.add_argument(
        "--fixture",
        action="append",
        default=[],
        help="name=path pair; repeat for several fixtures",
    )
    args = parser.parse_args(argv)

    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        args.model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    captured: list[torch.Tensor] = []
    original = modeling.get_audio_cu_seqlens

    def spy(*call_args, **call_kwargs):
        result = original(*call_args, **call_kwargs)
        if not call_kwargs.get("cu_seqlens") is not None:
            captured.append(result.detach().clone())
        return result

    modeling.get_audio_cu_seqlens = spy

    rows = []
    for spec in args.fixture or ["jfk=tools/data/fixtures/audio/jfk-short.wav"]:
        name, _, path = spec.partition("=")
        captured.clear()
        with wave.open(path, "rb") as handle:
            raw = handle.readframes(handle.getnframes())
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        processor = transformers.AutoProcessor.from_pretrained(args.model_dir)
        inputs = processor.apply_transcription_request(audio=pcm, sampling_rate=16000)
        features = inputs["input_features"].to(torch.float32)
        mask = inputs["input_features_mask"]
        if mask is None:
            mask = inputs.get("attention_mask")

        with torch.no_grad():
            reference = model.model.multi_modal_projector(
                model.model.audio_tower(features, mask).last_hidden_state
            ).numpy()

        if not captured:
            raise SystemExit(f"{name}: the encoder did not compute a schedule")
        schedule = captured[0].to(torch.int32)
        destination = args.output_dir / f"{name}_cu_seqlens.bin"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        schedule.numpy().tofile(destination)

        rows.append(
            {
                "name": name,
                "audio": path,
                "frames": int(features.shape[-1]),
                "capturedCalls": len(captured),
                "cuSeqlens": schedule.tolist(),
                "windows": int(schedule.shape[0]) - 1,
                "audioEmbeds": list(reference.shape),
                "file": destination.name,
            }
        )

    modeling.get_audio_cu_seqlens = original
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
