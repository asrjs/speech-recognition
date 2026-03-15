from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic mono 16 kHz WAV for cross-runtime ASR parity checks.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--subtype", default="PCM_16")
    return parser.parse_args()


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)

    ratio = target_rate / source_rate
    out_len = max(1, int(round(audio.shape[0] * ratio)))
    out = np.empty(out_len, dtype=np.float32)
    scale = source_rate / target_rate
    for i in range(out_len):
        pos = i * scale
        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, audio.shape[0] - 1)
        frac = pos - i0
        out[i] = audio[i0] * (1 - frac) + audio[i1] * frac
    return out


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    audio, sample_rate = sf.read(str(input_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    resampled = resample_linear(audio, sample_rate, args.sample_rate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), resampled, args.sample_rate, subtype=args.subtype)
    print(output_path)


if __name__ == "__main__":
    main()
