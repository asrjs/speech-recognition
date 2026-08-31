#!/usr/bin/env python
"""Run full pantinor Nemotron 3.5 ONNX pipeline and compare tokens to NeMo oracle.

Pantinor export: streaming encoder + batched decoder_joint that takes all
encoder frames and all targets at once, returns logits [B, T_enc, U, 13088].
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch, torchaudio
import sentencepiece as spm
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

OUT = Path("tools/data/results/nemotron/nemotron-3.5-pantinor-pipeline-2026-08-31.json")
ONNX = Path("N:/models/onnx/nemo/nemotron-3.5-asr-streaming-pantinor")
NEMO = Path("N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo")
FIXTURE = "tools/data/fixtures/audio/jfk-short.wav"


def load_wav(p, target_sr=16000):
    wav, sr = torchaudio.load(p)
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr: wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.squeeze(0).numpy(), target_sr


def main():
    print("Loading ...", flush=True)
    enc = ort.InferenceSession(str(ONNX / "encoder.onnx"), providers=["CPUExecutionProvider"])
    dj = ort.InferenceSession(str(ONNX / "decoder_joint.onnx"), providers=["CPUExecutionProvider"])

    nemo = EncDecRNNTBPEModelWithPrompt.restore_from(restore_path=str(NEMO), map_location="cpu")
    nemo.eval(); nemo.set_inference_prompt("en")

    sp = spm.SentencePieceProcessor(); sp.load(str(ONNX / "tokenizer.model"))
    print(f"Vocab: {sp.get_piece_size()} tokens", flush=True)

    wav, _ = load_wav(FIXTURE)
    sig = torch.from_numpy(wav).float().unsqueeze(0)
    length = torch.tensor([len(wav)], dtype=torch.long)
    with torch.no_grad():
        mel_t = nemo.preprocessor(input_signal=sig, length=length)[0]
    mel = mel_t.transpose(1, 2).squeeze(0).numpy()
    print(f"Mel: {mel.shape}", flush=True)

    # Streaming encoder
    chunk_size = 32
    cache_ch = np.zeros([24, 1, 56, 1024], dtype=np.float32)
    cache_t = np.zeros([24, 1, 1024, 8], dtype=np.float32)
    cache_ch_len = np.zeros([1], dtype=np.int64)
    all_enc = []
    mel_idx = 0
    while mel_idx < mel.shape[0]:
        chunk = mel[mel_idx:mel_idx+chunk_size]
        if chunk.shape[0] < chunk_size:
            chunk = np.vstack([chunk, np.zeros([chunk_size-chunk.shape[0], 128], dtype=np.float32)])
        feeds = {
            "processed_signal": chunk.T[np.newaxis, ...],
            "processed_signal_length": np.array([chunk.shape[0]], dtype=np.int64),
            "cache_last_channel": cache_ch,
            "cache_last_time": cache_t,
            "cache_last_channel_len": cache_ch_len,
            "prompt_index": np.array([0], dtype=np.int64),
        }
        outputs = enc.run(None, feeds)
        result = dict(zip([o.name for o in enc.get_outputs()], outputs))
        all_enc.append(result["encoded"][0])
        cache_ch = result["cache_last_channel_next"]
        cache_t = result["cache_last_time_next"]
        cache_ch_len = result["cache_last_channel_len_next"]
        mel_idx += chunk_size
    enc_full = np.concatenate(all_enc, axis=1)  # [1024, T_enc]
    T_enc = enc_full.shape[1]
    print(f"Encoded: {enc_full.shape}", flush=True)

    # blank_id is the last token in the vocab (RNN-T convention: V+1 classes)
    blank_id = sp.get_piece_size()  # 13087 for Nemotron (vocab=13087, blank=13087)
    print(f"Vocab size: {sp.get_piece_size()}, blank_id={blank_id}", flush=True)
    T_enc = enc_full.shape[1]

    # decoder_joint expects [B, 1024, T_enc]
    enc_b = enc_full[np.newaxis, ...].astype(np.float32)
    # Greedy RNN-T decode via iterative batched calls:
    # At each step, run decoder_joint with current targets, argmax over [B, T, U, V]
    # at the last target position; emit non-blank, advance.
    targets = [blank_id]
    token_ids = []
    h1 = np.zeros([2, 1, 640], dtype=np.float32)
    h2 = np.zeros([2, 1, 640], dtype=np.float32)

    # RNN-T loop: at each frame, decode with current targets, decide blank or emit
    # Since decoder_joint is batched over T_enc, we process all frames at once
    # for each "step" of the greedy algorithm.
    for step in range(100):
        targets_arr = np.array([targets], dtype=np.int32)
        target_length_arr = np.array([len(targets)], dtype=np.int32)
        feeds = {
            "encoder_outputs": enc_b,
            "targets": targets_arr,
            "target_length": target_length_arr,
            "input_states_1": h1,
            "input_states_2": h2,
        }
        outs = dj.run(None, feeds)
        result = dict(zip([o.name for o in dj.get_outputs()], outs))
        logits = result["outputs"]  # [1, T_enc, U, 13088]
        # Take logits at last target position [1, T_enc, U_last, 13088]
        last_logits = logits[0, :, -1, :]  # [T_enc, 13088]
        # Greedy: find first frame (smallest t) where argmax != blank
        non_blank_frame = -1
        for t in range(T_enc):
            y = int(np.argmax(last_logits[t]))
            if y != blank_id:
                non_blank_frame = t
                y_emitted = y
                break
        if non_blank_frame < 0:
            print(f"Step {step}: all blank, stopping", flush=True)
            break
        token_ids.append(y_emitted)
        targets.append(y_emitted)
        print(f"Step {step}: emitted {y_emitted} ('{sp.id_to_piece(y_emitted)}') at frame {non_blank_frame}", flush=True)
        if len(token_ids) >= 100:
            break

    # Decode
    pieces = [sp.id_to_piece(t) for t in token_ids if t < sp.get_piece_size()]
    text = "".join(pieces).replace("\u2581", " ").strip()
    print(f"\nFinal text: {text}", flush=True)

    OUT.write_text(json.dumps({
        "schemaVersion": 1,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "Pantinor ONNX pipeline token test",
        "encodedShape": list(enc_full.shape),
        "tokenIds": token_ids[:200],
        "tokenCount": len(token_ids),
        "text": text,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()