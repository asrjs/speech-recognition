#!/usr/bin/env python
"""Run ONNX encoder+decoder+joiner for Nemotron 3.5 and compare tokens with NeMo oracle.

Run with the isolated venv:
  .venv/Scripts/python.exe run_onnx_full_pipeline.py \
    --onnx-dir N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-onnx \
    --nemo N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
    --fixture tools/data/fixtures/audio/jfk-short.wav \
    --output tools/data/results/nemotron/nemotron-3.5-onnx-pipeline-2026-08-31.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
    EncDecRNNTBPEModelWithPrompt,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", required=True)
    p.add_argument("--nemo", required=True)
    p.add_argument("--fixture", action="append", required=True, dest="fixtures")
    p.add_argument("--output", required=True)
    p.add_argument("--prompt", default="en")
    return p.parse_args()


def load_wav(path: str, target_sr: int = 16000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.squeeze(0).numpy(), target_sr


# ---------- ONNX session helpers ----------

def create_onnx_session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def init_encoder_caches(sess):
    """Build zero-initialized cache tensors from session input metadata."""
    caches = {}
    for inp in sess.get_inputs():
        if inp.name in ("input_features", "prompt_ids", "cache_mask"):
            continue
        shape = [max(1, dim) for dim in inp.shape]
        dtype = np.int64 if "int64" in inp.type else np.float32
        caches[inp.name] = np.zeros(shape, dtype=dtype)
    return caches


def run_encoder(sess, features: np.ndarray, prompt_id: int, caches: dict,
                cache_mask_value: float = 1.0):
    """Run encoder session. features: [1, T, 128] float32. Returns (enc_out, updated_caches).

    Updated caches are keyed by the input names (e.g. k_cache_0), stripping the
    "_out_" suffix that the encoder's output layer applies.
    cache_mask_value: 0.0 for first chunk (empty cache, ignore positions),
                      1.0 for continuation (use populated cache).
    """
    onnx_inp = {
        "input_features": features.astype(np.float32),
        "prompt_ids": np.array([prompt_id], dtype=np.int64),  # [1] not [1,1]
        "cache_mask": np.full([1, 1, 1, 60], cache_mask_value, dtype=np.float32),
    }
    for name, cache in caches.items():
        if name not in onnx_inp:
            onnx_inp[name] = cache
    outputs = sess.run(None, onnx_inp)
    out_names = [o.name for o in sess.get_outputs()]
    result = dict(zip(out_names, outputs))
    enc_out = result["encoder_out"]  # [1, T_enc, 640]
    # Strip _out_ suffix from output cache names so they match the input naming
    new_caches = {}
    for k, v in result.items():
        if k == "encoder_out":
            continue
        input_name = k.replace("_out_", "_") if k.endswith(f"_out_{k.split('_out_')[-1]}") else k
        # Simpler: trim trailing "_out_<digits>"
        if "_out_" in k:
            base, _, tail = k.partition("_out_")
            new_caches[f"{base}_{tail}"] = v
        else:
            new_caches[k] = v
    return enc_out, new_caches


def run_decoder(sess, target_id: int, h_in: np.ndarray, c_in: np.ndarray):
    """Run decoder session. target_id: int. h_in/c_in: [2,1,640]. Returns (dec_out, h_out, c_out)."""
    feeds = {
        "token": np.array([[target_id]], dtype=np.int64),
        "h_in": h_in.astype(np.float32),
        "c_in": c_in.astype(np.float32),
    }
    outs = sess.run(None, feeds)
    out_names = [o.name for o in sess.get_outputs()]
    result = dict(zip(out_names, outs))
    return result["decoder_out"], result["h_out"], result["c_out"]


def run_joiner(sess, enc_frame: np.ndarray, dec_out: np.ndarray):
    """Run joiner session. enc_frame: [1,640], dec_out: [1,640]. Returns: [1,13088]."""
    feeds = {
        "encoder_frame": enc_frame.astype(np.float32),
        "decoder_out": dec_out.astype(np.float32),
    }
    return sess.run(None, feeds)[0]


# ---------- Greedy RNNT decode ----------

def greedy_decode(encoder_first_sess, encoder_cont_sess, decoder_sess, joiner_sess,
                 mel: np.ndarray, prompt_id: int = 0, blank_id: int = 13087):
    """Greedy RNNT decode on mel features [T_mel, 128].

    Each encoder session carries its own cache dict (first vs continuation differ).
    """
    T = mel.shape[0]
    token_ids = []
    log_probs = []

    first_caches = init_encoder_caches(encoder_first_sess)
    cont_caches = init_encoder_caches(encoder_cont_sess)
    h = np.zeros([2, 1, 640], dtype=np.float32)
    c = np.zeros([2, 1, 640], dtype=np.float32)
    target = blank_id
    mel_idx = 0
    is_first = True
    chunk_count = 0
    while mel_idx < T:
        chunk_size = 25 if is_first else 32
        chunk = mel[mel_idx:mel_idx + chunk_size]
        if chunk.shape[0] < chunk_size:
            pad = np.zeros([chunk_size - chunk.shape[0], 128], dtype=np.float32)
            chunk = np.vstack([chunk, pad])
        chunk_btc = chunk[np.newaxis, ...]  # [1, chunk_size, 128]

        if is_first:
            # First chunk: cache is empty so cache_mask=0 (ignore cache positions)
            enc_out, first_caches = run_encoder(encoder_first_sess, chunk_btc, prompt_id, first_caches, cache_mask_value=0.0)
        else:
            # Continuation: cache is populated so cache_mask=1 (use cache positions)
            enc_out, cont_caches = run_encoder(encoder_cont_sess, chunk_btc, prompt_id, cont_caches, cache_mask_value=1.0)
        is_first = False
        chunk_count += 1
        if chunk_count <= 3 or chunk_count % 10 == 0:
            print(f"  chunk {chunk_count}: mel_idx={mel_idx}, enc_out shape={enc_out.shape}, maxAbs={float(np.abs(enc_out).max()):.3f}", flush=True)

        # Process ALL encoder frames in this chunk (don't break on blank).
        # The streaming encoder advances by the full chunk; partial advance would
        # desync the cache. RNN-T step at each enc frame: blank=advance, token=emit&stay.
        for t in range(enc_out.shape[1]):
            enc_frame = enc_out[:, t, :]  # [1, 640]
            dec_out, h, c = run_decoder(decoder_sess, target, h, c)
            logits = run_joiner(joiner_sess, enc_frame, dec_out)
            y = int(np.argmax(logits))
            lp = float(logits[0, y])

            if y == blank_id:
                mel_idx += 8  # advance one encoder frame worth of mel
                target = blank_id  # reset to blank for next frame
            else:
                token_ids.append(y)
                log_probs.append(lp)
                target = y

    return token_ids, log_probs


# ---------- Main ----------

def main() -> None:
    args = parse_args()
    onnx_dir = Path(args.onnx_dir)
    nemo_path = Path(args.nemo)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading ONNX sessions ...", flush=True)
    enc_first = create_onnx_session(onnx_dir / "encoder_320ms_first_fp16.onnx")
    enc_cont = create_onnx_session(onnx_dir / "encoder_320ms_fp16.onnx")
    decoder = create_onnx_session(onnx_dir / "decoder.onnx")
    joiner = create_onnx_session(onnx_dir / "joiner.onnx")

    print("Loading NeMo model for mel features ...", flush=True)
    nemo_model = EncDecRNNTBPEModelWithPrompt.restore_from(
        restore_path=str(nemo_path), map_location="cpu"
    )
    nemo_model.eval()
    if hasattr(nemo_model, "set_inference_prompt"):
        nemo_model.set_inference_prompt(args.prompt)

    # Tokenizer: tokens.txt is flat vocab (id<TAB>token); build reverse mapping for decode
    tokenizer = None
    try:
        vocab_path = onnx_dir / "tokens.txt"
        if not vocab_path.exists():
            vocab_path = onnx_dir.parent / "tokens.txt"
        id_to_token = {}
        for line in vocab_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                idx = int(parts[0])
                id_to_token[idx] = parts[1]
        tokenizer = id_to_token
        print(f"  Tokenizer: {len(id_to_token)} tokens (flat vocab)", flush=True)
    except Exception as e:
        print(f"  Tokenizer load failed: {e}", flush=True)
    prompt_map = {"en": 0, "en-US": 0, "auto": 101, "tr": 18}
    prompt_id = prompt_map.get(args.prompt, 0)
    # NeMo reference transcripts
    ref_path = Path("tools/data/results/nemotron/nemotron-3.5-official-reference-2026-08-30.json")
    ref_data = {}
    if ref_path.exists():
        with open(ref_path) as f:
            ref_full = json.load(f)
        for r in ref_full.get("results", []):
            ref_data[r["fixtureName"]] = r

    records = []
    for fixture_path in args.fixtures:
        fixture = Path(fixture_path)
        print(f"\nProcessing: {fixture.name}", flush=True)

        wav, sr = load_wav(str(fixture))
        print(f"  Audio: {len(wav)} samples @ {sr} Hz ({len(wav)/sr:.2f}s)", flush=True)

        # Get mel from NeMo preprocessor
        sig_t = torch.from_numpy(wav).float().unsqueeze(0)
        len_t = torch.tensor([len(wav)], dtype=torch.long)
        with torch.no_grad():
            mel_t = nemo_model.preprocessor(input_signal=sig_t, length=len_t)[0]
        mel = mel_t.transpose(1, 2).squeeze(0).numpy()  # [T_mel, 128]
        print(f"  Mel: {mel.shape}, max={float(mel.max()):.2f}", flush=True)

        # Run greedy decode
        token_ids, log_probs = greedy_decode(
            enc_first, enc_cont, decoder, joiner, mel,
            prompt_id=prompt_id, blank_id=13087
        )
        print(f"  ONNX tokens: {len(token_ids)}", flush=True)
        print(f"  First 10 tokens: {token_ids[:10]}", flush=True)

        # Decode to text
        text = None
        if tokenizer:
            try:
                pieces = [tokenizer[t] for t in token_ids if t in tokenizer]
                # SentencePiece vocab uses ▁ for word boundary
                text = "".join(pieces).replace("▁", " ").strip()
                print(f"  ONNX text: {text}", flush=True)
            except Exception as e:
                text = f"<err:{e}>"
                print(f"  Decode error: {e}", flush=True)
        else:
            text = None

        # Compare with NeMo reference
        ref = ref_data.get(fixture.name)
        token_match = None
        text_match = None
        if ref:
            ref_tokens = ref.get("tokenIds")
            ref_text = ref.get("text", "")
            if ref_tokens is not None:
                token_match = token_ids == ref_tokens
                print(f"  NeMo tokens: {len(ref_tokens)}", flush=True)
                print(f"  Token match: {token_match}", flush=True)
                if not token_match:
                    # Find first diff
                    for i, (a, b) in enumerate(zip(token_ids, ref_tokens)):
                        if a != b:
                            print(f"  First diff at {i}: ONNX={a} NeMo={b}", flush=True)
                            break
            text_match = (text or "").strip() == ref_text.strip() if text else None
            print(f"  NeMo text: {ref_text}", flush=True)
            print(f"  Text match: {text_match}", flush=True)

        record = {
            "fixture": fixture.as_posix(),
            "fixtureName": fixture.name,
            "tokenIds": token_ids,
            "tokenCount": len(token_ids),
            "logProbs": log_probs,
            "text": text,
            "melShape": list(mel.shape),
            "referenceTokens": ref.get("tokenIds") if ref else None,
            "referenceText": ref.get("text") if ref else None,
            "tokenMatch": token_match,
            "textMatch": text_match,
        }
        records.append(record)

    output_record = {
        "schemaVersion": 1,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "ONNX pipeline token and text output vs NeMo oracle",
        "onnxDir": onnx_dir.as_posix(),
        "promptId": prompt_id,
        "records": records,
    }
    output_path.write_text(json.dumps(output_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
