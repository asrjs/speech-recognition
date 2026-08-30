#!/usr/bin/env python
"""Extract official NeMo mel features from a WAV fixture.

Run with the isolated venv:
  .venv/Scripts/python.exe extract_nemo_features.py \
    --nemo N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
    --fixture tools/data/fixtures/audio/jfk-short.wav \
    --output tools/data/results/nemotron/nemotron-3.5-nemo-features-jfk.json
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
    EncDecRNNTBPEModelWithPrompt,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nemo", required=True)
    p.add_argument("--fixture", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_wav(path: str, target_sr: int = 16000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.squeeze(0).numpy(), target_sr


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Restoring {args.nemo} ...", flush=True)
    model = EncDecRNNTBPEModelWithPrompt.restore_from(
        restore_path=args.nemo, map_location=torch.device(args.device)
    )
    model.eval()

    preprocessor = model.preprocessor
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(model.cfg.preprocessor, resolve=True)
    except Exception:
        cfg = {}

    print(f"Loading {args.fixture} ...", flush=True)
    audio_np, sr = load_wav(args.fixture, cfg.get("sample_rate", 16000))
    audio_signal = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
    audio_len = torch.tensor([len(audio_np)], dtype=torch.long)

    with torch.no_grad():
        features, features_len = preprocessor(
            input_signal=audio_signal, length=audio_len
        )

    feat_np = features.squeeze(0).cpu().float().numpy()
    feat_len = int(features_len[0].item())

    record = {
        "schemaVersion": 1,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "fixture": Path(args.fixture).as_posix(),
        "sampleRate": cfg.get("sample_rate", 16000),
        "nFft": cfg.get("n_fft", 512),
        "winLength": cfg.get("win_length", 400),
        "hopLength": cfg.get("hop_length", 160),
        "nMels": cfg.get("n_mels", 128),
        "normalize": cfg.get("normalize", "NA"),
        "preemphasis": cfg.get("preemphasis", 0.97),
        "dither": cfg.get("dither", 1e-5),
        "logZeroGuard": cfg.get("log_zero_guard", 2**-24),
        "featureShape": [feat_len, feat_np.shape[1]],
        "feature": feat_np[:feat_len].tolist(),
    }

    out.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Features: {feat_len} frames x {feat_np.shape[1]} bins", flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
