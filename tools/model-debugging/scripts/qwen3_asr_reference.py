"""Capture a Qwen3-ASR reference transcript on a local WAV.

Qwen3-ASR is a conditional model: the processor builds a prompt alongside the
audio, and the raw output carries a language tag and an <asr_text> marker, so
`transcription_only` is the form a library gate should compare against.

Run it with the transformers 5.x environment:
    tools/model-debugging/reference/transformers5/Scripts/python.exe \\
        tools/model-debugging/scripts/qwen3_asr_reference.py --audio <wav>
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


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as handle:
        raw = handle.readframes(handle.getnframes())

    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=".cache/qwen3-asr-0.6b")
    parser.add_argument("--audio", default="tools/data/fixtures/audio/jfk-short.wav")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)

    processor = transformers.AutoProcessor.from_pretrained(args.model_dir)
    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        args.model_dir, dtype=torch.float32
    ).eval()

    pcm = read_wav(args.audio)
    inputs = processor.apply_transcription_request(audio=pcm, sampling_rate=16000)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]

    payload = {
        "audio": args.audio,
        "samples": int(pcm.shape[0]),
        "raw": processor.decode(generated)[0],
        "parsed": processor.decode(generated, return_format="parsed")[0],
        "transcription": processor.decode(generated, return_format="transcription_only")[0],
        "token_ids": generated[0].tolist(),
        "encoderFrames": int(inputs["input_features"].shape[-1]),
        "note": "reference capture from the official implementation; not human ground truth",
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
