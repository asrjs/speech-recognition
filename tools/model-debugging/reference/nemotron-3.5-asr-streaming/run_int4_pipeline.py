#!/usr/bin/env python
"""End-to-end INT4 Nemotron 3.5 ONNX pipeline (encoder + decoder + joint)."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch, torchaudio
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

OUT = Path("tools/data/results/nemotron/nemotron-3.5-int4-pipeline-2026-08-31.json")
ONNX = Path("N:/models/onnx/nemo/nemotron-3.5-asr-streaming-int4")
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
    dec = ort.InferenceSession(str(ONNX / "decoder.onnx"), providers=["CPUExecutionProvider"])
    jnt = ort.InferenceSession(str(ONNX / "joint.onnx"), providers=["CPUExecutionProvider"])

    nemo = EncDecRNNTBPEModelWithPrompt.restore_from(restore_path=str(NEMO), map_location="cpu")
    nemo.eval(); nemo.set_inference_prompt("en")

    id_to_token = {idx: tok for idx, tok in enumerate((ONNX / "vocab.txt").read_text(encoding="utf-8").splitlines())}
    blank_id = next((tid for tid, tok in id_to_token.items() if tok == "<blank>"), 13087)
    print(f"Vocab: {len(id_to_token)} tokens, blank_id={blank_id}", flush=True)

    wav, _ = load_wav(FIXTURE)
    sig = torch.from_numpy(wav).float().unsqueeze(0)
    length = torch.tensor([len(wav)], dtype=torch.long)
    with torch.no_grad():
        mel_t = nemo.preprocessor(input_signal=sig, length=length)[0]
    mel = mel_t.transpose(1, 2).squeeze(0).numpy()
    print(f"Mel: {mel.shape}", flush=True)

    chunk_size = 65
    cache_ch = np.zeros([1, 24, 56, 1024], dtype=np.float32)
    cache_t = np.zeros([1, 24, 1024, 8], dtype=np.float32)
    cache_ch_len = np.zeros([1], dtype=np.int64)
    all_enc = []
    mel_idx = 0
    while mel_idx < mel.shape[0]:
        chunk = mel[mel_idx:mel_idx+chunk_size]
        if chunk.shape[0] < chunk_size:
            chunk = np.vstack([chunk, np.zeros([chunk_size-chunk.shape[0], 128], dtype=np.float32)])
        outputs = enc.run(None, {
            "audio_signal": chunk[np.newaxis, ...].astype(np.float32),
            "length": np.array([chunk.shape[0]], dtype=np.int64),
            "cache_last_channel": cache_ch,
            "cache_last_time": cache_t,
            "cache_last_channel_len": cache_ch_len,
            "lang_id": np.array([0], dtype=np.int64),
        })
        result = dict(zip([o.name for o in enc.get_outputs()], outputs))
        all_enc.append(result["outputs"][0])
        cache_ch = result["cache_last_channel_next"]
        cache_t = result["cache_last_time_next"]
        cache_ch_len = result["cache_last_channel_len_next"]
        mel_idx += chunk_size
    enc_full = np.concatenate(all_enc, axis=0)
    print(f"Encoded: {enc_full.shape}", flush=True)

    enc_b = enc_full[np.newaxis, ...].astype(np.float32)
    targets = [blank_id]
    token_ids = []
    h = np.zeros([2, 1, 640], dtype=np.float32)
    c = np.zeros([2, 1, 640], dtype=np.float32)
    last_t = 0
    for step in range(200):
        targets_arr = np.array([targets], dtype=np.int64)
        dec_outs = dec.run(None, {"targets": targets_arr, "h_in": h, "c_in": c})
        dec_out = dec_outs[0]  # [1, 640, target_len]
        h = dec_outs[1]; c = dec_outs[2]
        enc_remaining = enc_b[:, last_t:, :]
        if enc_remaining.shape[1] == 0:
            print(f"Step {step}: exhausted enc frames", flush=True)
            break
        dec_out_t = dec_out.transpose(0, 2, 1)  # [1, target_len, 640]
        jnt_outs = jnt.run(None, {"encoder_output": enc_remaining, "decoder_output": dec_out_t})
        logits = jnt_outs[0]
        last_logits = logits[0, :, -1, :]
        y_emitted = -1; non_blank_frame = -1
        for t_local in range(last_logits.shape[0]):
            y = int(np.argmax(last_logits[t_local]))
            if y != blank_id:
                non_blank_frame = t_local; y_emitted = y; break
        if non_blank_frame < 0:
            last_t += last_logits.shape[0]
            continue
        token_ids.append(y_emitted)
        targets.append(y_emitted)
        last_t += non_blank_frame
        if step % 10 == 0:
            print(f"Step {step}: emitted {y_emitted} ({id_to_token.get(y_emitted, '?')}) at frame {last_t}", flush=True)
        if len(token_ids) >= 100: break

    pieces = [id_to_token.get(t, "?") for t in token_ids]
    text = "".join(pieces).replace("\u2581", " ").strip()
    print(f"\nFinal text: {text}", flush=True)
    print(f"Total tokens: {len(token_ids)}", flush=True)

    OUT.write_text(json.dumps({
        "schemaVersion": 1,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "INT4 Nemotron 3.5 ONNX pipeline end-to-end test",
        "encodedShape": list(enc_full.shape),
        "tokenIds": token_ids,
        "tokenCount": len(token_ids),
        "text": text,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()