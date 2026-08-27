"""Dump official torchaudio STFT / filterbank / pre-log / log-mel stages.

Used to localize JS PREPROCESSING_MISMATCH against the GigaAM oracle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"N:\models\gigaam\multilingual-ctc\captures\frontend-stages"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture = json.loads(args.reference.read_text(encoding="utf-8"))
    cfg = capture["preprocessor"]
    sample = capture["samples"][0]
    waveform = np.load(sample["audio"]["waveform_npy"])
    waveform = np.ascontiguousarray(waveform.reshape(-1), dtype=np.float32)

    sample_rate = int(cfg["sample_rate"])
    n_fft = int(cfg["n_fft"])
    win_length = int(cfg["win_length"])
    hop_length = int(cfg["hop_length"])
    n_mels = int(cfg["n_mels"])
    center = bool(cfg["center"])
    n_freqs = n_fft // 2 + 1

    audio = torch.from_numpy(waveform).unsqueeze(0)
    window = torch.hann_window(win_length, periodic=True)
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=2.0,
        center=center,
        pad=0,
        normalized=False,
        onesided=True,
        window_fn=torch.hann_window,
    )
    stft_power = spectrogram(audio)[0]
    fbanks = torchaudio.functional.melscale_fbanks(
        n_freqs=n_freqs,
        f_min=0.0,
        f_max=float(sample_rate // 2),
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm=None,
        mel_scale="htk",
    )
    pre_log = torch.matmul(fbanks.transpose(0, 1), stft_power)
    log_mel = torch.log(pre_log.clamp(1e-9, 1e9))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "stft_power.npy", np.ascontiguousarray(stft_power.numpy()))
    np.save(args.output_dir / "filterbank.npy", np.ascontiguousarray(fbanks.numpy()))
    np.save(args.output_dir / "pre_log_mel.npy", np.ascontiguousarray(pre_log.numpy()))
    np.save(args.output_dir / "log_mel.npy", np.ascontiguousarray(log_mel.numpy()))
    np.save(args.output_dir / "window.npy", np.ascontiguousarray(window.numpy()))

    payload = {
        "stft_power_shape": list(stft_power.shape),
        "filterbank_shape": list(fbanks.shape),
        "pre_log_mel_shape": list(pre_log.shape),
        "log_mel_shape": list(log_mel.shape),
        "window_length": int(window.numel()),
        "frame0_stft_max": float(stft_power[:, 0].max()),
        "frame0_log_mel_min": float(log_mel[:, 0].min()),
        "frame0_log_mel_max": float(log_mel[:, 0].max()),
        "pre_log_zeros": int((pre_log <= 0).sum()),
        "pre_log_below_guard": int((pre_log < 1e-9).sum()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
