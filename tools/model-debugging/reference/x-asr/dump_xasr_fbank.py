"""Dump sherpa-onnx / knf fbank for X-ASR-zh-en.

Matches OnlineRecognizer.from_transducer defaults: 80-bin Kaldi fbank,
dither 0, snip_edges False, high_freq -400.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-npy", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wave, rate = sf.read(str(args.audio), dtype="float32", always_2d=True)
    wave = wave.mean(axis=1).astype(np.float32)
    if rate != 16000:
        raise ValueError(f"expected 16 kHz, got {rate}")
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.remove_dc_offset = True
    opts.frame_opts.preemph_coeff = 0.97
    opts.frame_opts.round_to_power_of_two = True
    opts.mel_opts.num_bins = 80
    opts.mel_opts.low_freq = 20
    opts.mel_opts.high_freq = -400
    opts.mel_opts.is_librosa = False
    opts.use_log_fbank = True
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, wave)
    fbank.input_finished()
    feats = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)]).astype(
        np.float32
    )
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, feats)
    payload = {
        "audio": str(args.audio.resolve()),
        "samples": int(wave.size),
        "frames": int(feats.shape[0]),
        "feature_dim": int(feats.shape[1]),
        "mean": float(feats.mean()),
        "std": float(feats.std()),
        "min": float(feats.min()),
        "max": float(feats.max()),
        "frame0": feats[0, :8].tolist(),
        "frame_last": feats[-1, :8].tolist(),
        "contract": {
            "snip_edges": False,
            "dither": 0.0,
            "low_freq": 20,
            "high_freq": -400,
            "num_bins": 80,
            "window": "povey",
        },
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": payload["frames"], "npy": str(args.output_npy)}, indent=2))


if __name__ == "__main__":
    main()
