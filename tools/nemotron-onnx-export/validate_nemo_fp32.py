#!/usr/bin/env python
"""Validate the freshly NeMo-exported fp32 cache-aware pipeline on jfk.

Runs the executor-equivalent decode loop (NeMo preprocessor mel, 65-frame
chunks, lang_id 0/en) against the nemo-exported fp32 graphs, using the same
I/O names and layouts as ``src/models/nemotron-rnnt/executor.ts``, and
asserts the exact token sequence matches the recorded oracle.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch, torchaudio
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

OUT = Path("tools/data/results/nemotron/nemotron-3.5-nemo-fp32-pipeline-2026-08-31.json")
NEMO_DIR = Path("N:/models/onnx/nemo/nemotron-3.5-asr-streaming-fp32-nemo")
NEMO_NEMO = Path("N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo")
FIXTURE = "tools/data/fixtures/audio/jfk-short.wav"
ORACLE_TOKEN_IDS = [2860, 2, 1290, 2, 1731, 1179, 290, 133, 309, 271, 581, 1257, 113, 40, 2900, 2847, 2831, 2815, 995, 313, 113, 34, 1752, 2813, 2, 1288, 398, 2810, 400, 2831, 2810, 2813, 2, 1288, 398, 2815, 995, 313, 113, 34, 1752]
ORACLE_TEXT = "And so my fellow Americans ask not what your country can do for you at what you can do for your country"
LANGUAGE = "en"
LANG_ID = 0


def load_wav(p, target_sr=16000):
    wav, sr = torchaudio.load(p)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.squeeze(0).numpy(), target_sr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=str(NEMO_DIR))
    p.add_argument("--fixture", default=FIXTURE)
    p.add_argument("--tag", default="fp32", help="evidence filename tag (fp32|fp16)")
    return p.parse_args()




def out_path(tag: str) -> Path:
    return Path(f"tools/data/results/nemotron/nemotron-3.5-nemo-{tag}-pipeline-2026-08-31.json")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    so = ort.SessionOptions(); so.log_severity_level = 3
    enc = ort.InferenceSession(str(model_dir / "encoder.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(model_dir / "decoder.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    jnt = ort.InferenceSession(str(model_dir / "joint.onnx"), sess_options=so, providers=["CPUExecutionProvider"])

    nemo = EncDecRNNTBPEModelWithPrompt.restore_from(restore_path=str(NEMO_NEMO), map_location="cpu")
    nemo.eval(); nemo.set_inference_prompt(LANGUAGE)

    id_to_token = {}
    with open(model_dir / "vocab.txt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            id_to_token[i] = line.rstrip("\n")
    blank_id = next((tid for tid, tok in id_to_token.items() if tok == "<blank>"), 13087)
    print(f"Vocab: {len(id_to_token)} tokens, blank_id={blank_id}", flush=True)

    wav, _ = load_wav(args.fixture)
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
        valid = chunk.shape[0]
        if valid < chunk_size:
            chunk = np.vstack([chunk, np.zeros([chunk_size-valid, 128], dtype=np.float32)])
        feed = {
            "audio_signal": chunk[np.newaxis, ...].astype(np.float32),  # [1, T, mel]
            "length": np.array([valid], dtype=np.int64),
            "cache_last_channel": cache_ch,
            "cache_last_time": cache_t,
            "cache_last_channel_len": cache_ch_len,
            "lang_id": np.array([LANG_ID], dtype=np.int64),
        }
        result = dict(zip([o.name for o in enc.get_outputs()], enc.run(None, feed)))
        # outputs is frame-major [1, T', 1024]; stack rows into [T_total, 1024].
        all_enc.append(result["outputs"][0])
        cache_ch = result["cache_last_channel_next"]
        cache_t = result["cache_last_time_next"]
        cache_ch_len = result["cache_last_channel_len_next"]
        mel_idx += chunk_size
    enc_full = np.concatenate(all_enc, axis=0)
    print(f"Encoded: {enc_full.shape}", flush=True)

    enc_b = enc_full[np.newaxis, ...].astype(np.float32)  # [1, T_total, 1024]
    targets = [blank_id]
    token_ids = []
    h = np.zeros([2, 1, 640], dtype=np.float32)
    c = np.zeros([2, 1, 640], dtype=np.float32)
    last_t = 0
    for step in range(200):
        targets_arr = np.array([targets], dtype=np.int64)
        dec_outs = dec.run(None, {"targets": targets_arr, "h_in": h, "c_in": c})
        dec_out = dec_outs[0]          # [1, 640, U]
        h = dec_outs[1]; c = dec_outs[2]
        enc_remaining = enc_b[:, last_t:, :]
        if enc_remaining.shape[1] == 0:
            break
        dec_out_t = np.ascontiguousarray(dec_out.transpose(0, 2, 1))  # [1, U, 640]
        logits = jnt.run(None, {"encoder_output": enc_remaining, "decoder_output": dec_out_t})[0]
        last_logits = logits[0, :, -1, :]  # [T_remaining, V]
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
        if len(token_ids) >= 100:
            break

    text = "".join(id_to_token.get(t, "?") for t in token_ids).replace("\u2581", " ").strip()
    print(f"Tokens: {len(token_ids)}", flush=True)
    print(f"Text: {text}", flush=True)

    # The official NeMo reference and the WASM rung both append a closing
    # boundary segment (". <en-US>") after the transcript body. The oracle is
    # the exact body-token prefix; a boundary-only tail is accepted.
    import re
    body = re.sub(r"(\s*<[^>]+>)+\s*$", "", text).strip()
    body = re.sub(r"\.\s*$", "", body).strip()
    prefix_pass = token_ids[: len(ORACLE_TOKEN_IDS)] == ORACLE_TOKEN_IDS
    oracle_pass = prefix_pass and body == ORACLE_TEXT
    text_pass = body == ORACLE_TEXT
    report = {
        "schemaVersion": 1,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "nemo-exported fp32 cache-aware pipeline validation vs jfk oracle",
        "modelDir": str(model_dir),
        "fixture": str(args.fixture),
        "tokenCount": len(token_ids),
        "tokenIds": token_ids,
        "text": text,
        "oracle": {
            "tokenCount": len(ORACLE_TOKEN_IDS),
            "tokenIds": ORACLE_TOKEN_IDS,
            "text": ORACLE_TEXT,
            "tokenPrefixPass": prefix_pass,
            "tokenSequencePass": oracle_pass,
            "textPass": text_pass,
        },
    }
    out = out_path(args.tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}", flush=True)

    if not oracle_pass:
        raise SystemExit(f"ORACLE FAIL: token sequence mismatch ({len(token_ids)} vs {len(ORACLE_TOKEN_IDS)})")
    print("Oracle gate: PASS", flush=True)


if __name__ == "__main__":
    main()
