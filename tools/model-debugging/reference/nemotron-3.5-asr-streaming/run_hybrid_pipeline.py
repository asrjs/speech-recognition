#!/usr/bin/env python
"""Hybrid pipeline: NeMo encoder (correct output) + ONNX predictor/joiner.

The community ONNX encoder is broken (cosSim 0.05 vs NeMo reference). This script
uses NeMo's streaming conformer encoder for the encoder step, then ONNX predictor
and joiner for the RNN-T decode loop. This isolates whether the predictor+joiner
produce correct tokens given correct encoder features.

Usage:
  .venv/Scripts/python.exe run_hybrid_pipeline.py \
    --onnx-dir N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-onnx \
    --nemo N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
    --fixture tools/data/fixtures/audio/jfk-short.wav \
    --output tools/data/results/nemotron/nemotron-3.5-hybrid-pipeline-2026-08-31.json
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", required=True)
    p.add_argument("--nemo", required=True)
    p.add_argument("--fixture", action="append", required=True, dest="fixtures")
    p.add_argument("--output", required=True)
    p.add_argument("--prompt", default="en")
    p.add_argument("--chunk-size", type=int, default=32)
    return p.parse_args()


def load_wav(path, target_sr=16000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.squeeze(0).numpy(), target_sr


def run_decoder(sess, token_id, h, c):
    feeds = {
        "token": np.array([[token_id]], dtype=np.int64),
        "h_in": h.astype(np.float32),
        "c_in": c.astype(np.float32),
    }
    outs = sess.run(None, feeds)
    out_names = [o.name for o in sess.get_outputs()]
    result = dict(zip(out_names, outs))
    return result["decoder_out"], result["h_out"], result["c_out"]


def run_joiner(sess, enc_frame, dec_out):
    feeds = {
        "encoder_frame": enc_frame.astype(np.float32),
        "decoder_out": dec_out.astype(np.float32),
    }
    return sess.run(None, feeds)[0]


def main() -> None:
    args = parse_args()
    onnx_dir = Path(args.onnx_dir)
    nemo_path = Path(args.nemo)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading ONNX decoder + joiner ...", flush=True)
    decoder = ort.InferenceSession(str(onnx_dir / "decoder.onnx"), providers=["CPUExecutionProvider"])
    joiner = ort.InferenceSession(str(onnx_dir / "joiner.onnx"), providers=["CPUExecutionProvider"])

    print("Loading NeMo model ...", flush=True)
    nemo_model = EncDecRNNTBPEModelWithPrompt.restore_from(
        restore_path=str(nemo_path), map_location="cpu"
    )
    nemo_model.eval()
    if hasattr(nemo_model, "set_inference_prompt"):
        nemo_model.set_inference_prompt(args.prompt)
    encoder = nemo_model.encoder

    # Tokenizer
    vocab_path = onnx_dir / "tokens.txt"
    if not vocab_path.exists():
        vocab_path = onnx_dir.parent / "tokens.txt"
    id_to_token = {}
    for line in vocab_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            id_to_token[int(parts[0])] = parts[1]
    print(f"  Tokenizer: {len(id_to_token)} tokens", flush=True)

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

        # Run NeMo encoder on full mel (offline mode for simplicity)
        mel_torch = torch.from_numpy(mel).float().unsqueeze(0).transpose(1, 2)  # [1, 128, T]
        mel_len_torch = torch.tensor([mel.shape[0]], dtype=torch.long)
        with torch.no_grad():
            nemo_enc_out, _ = encoder(audio_signal=mel_torch, length=mel_len_torch)
            enc_projected = nemo_model.joint.enc(nemo_enc_out.transpose(1, 2))
        # nemo_enc_out: [1, 1024, T_enc]; apply joint.enc projection to get [1, T_enc, 640]
        enc_features = enc_projected.squeeze(0).detach().numpy()  # [T_enc, 640]
        print(f"  NeMo encoder out: {nemo_enc_out.shape}, projected: {enc_features.shape}", flush=True)

        # RNN-T greedy decode using ONNX decoder+joiner with NeMo encoder features
        T_enc = enc_features.shape[0]
        h = np.zeros([2, 1, 640], dtype=np.float32)
        c = np.zeros([2, 1, 640], dtype=np.float32)
        target = 13087  # blank
        token_ids = []
        log_probs = []
        for t in range(T_enc):
            enc_frame = enc_features[t:t+1, :]  # [1, 640]
            dec_out, h, c = run_decoder(decoder, target, h, c)
            logits = run_joiner(joiner, enc_frame, dec_out)
            y = int(np.argmax(logits))
            lp = float(logits[0, y])
            if y == 13087:
                target = 13087  # reset to blank
            else:
                token_ids.append(y)
                log_probs.append(lp)
                target = y
        print(f"  Hybrid tokens: {len(token_ids)}", flush=True)
        print(f"  First 10 tokens: {token_ids[:10]}", flush=True)

        # Decode to text
        try:
            pieces = [id_to_token[t] for t in token_ids if t in id_to_token]
            text = "".join(pieces).replace("▁", " ").strip()
            print(f"  Hybrid text: {text}", flush=True)
        except Exception as e:
            text = f"<err:{e}>"
            print(f"  Decode error: {e}", flush=True)

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
                    for i, (a, b) in enumerate(zip(token_ids, ref_tokens)):
                        if a != b:
                            print(f"  First diff at {i}: Hybrid={a} NeMo={b}", flush=True)
                            break
            text_match = text.strip() == ref_text.strip() if text else None
            print(f"  Text match: {text_match}", flush=True)

        record = {
            "fixture": fixture.as_posix(),
            "fixtureName": fixture.name,
            "tokenIds": token_ids,
            "tokenCount": len(token_ids),
            "logProbs": log_probs,
            "text": text,
            "encoderShape": list(enc_features.shape),
            "referenceTokens": ref.get("tokenIds") if ref else None,
            "referenceText": ref.get("text") if ref else None,
            "tokenMatch": token_match,
            "textMatch": text_match,
        }
        records.append(record)

    output_record = {
        "schemaVersion": 1,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "Hybrid pipeline: NeMo encoder + ONNX decoder/joiner token parity",
        "onnxDir": onnx_dir.as_posix(),
        "neMoCheckpoint": nemo_path.as_posix(),
        "promptId": prompt_id,
        "encoderSource": "NeMo (community ONNX encoder produces garbage, cosSim 0.05)",
        "predictorJoinerSource": "ONNX (community export)",
        "records": records,
    }
    output_path.write_text(json.dumps(output_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {output_path}", flush=True)


if __name__ == "__main__":
    main()