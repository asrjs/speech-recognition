#!/usr/bin/env python
"""Dump NeMo mel features to a numpy file for fast iteration."""
import sys
import numpy as np
import torch, torchaudio
from pathlib import Path
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

fixture = sys.argv[1] if len(sys.argv) > 1 else "tools/data/fixtures/audio/jfk-short.wav"
out_path = sys.argv[2] if len(sys.argv) > 2 else "tools/data/results/nemotron/_tmp_mel.npy"

nemo = EncDecRNNTBPEModelWithPrompt.restore_from(
    "N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo",
    map_location="cpu"
)
nemo.eval()
wav, sr = torchaudio.load(fixture)
if sr != 16000:
    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
sig = torch.from_numpy(wav.squeeze().numpy()).float().unsqueeze(0)
length = torch.tensor([wav.shape[1]], dtype=torch.long)
mel_t = nemo.preprocessor(input_signal=sig, length=length)[0]
mel = mel_t.transpose(1, 2).squeeze(0).numpy()
np.save(out_path, mel)
print(f"Saved {out_path}: shape={mel.shape}, max={float(mel.max()):.2f}, min={float(mel.min()):.2f}")