#!/usr/bin/env python
"""Validate the freshly NeMo-exported fp32 cache-aware pipeline per streaming window.

Runs the executor-equivalent decode loop (NeMo preprocessor mel, per-window
chunk sizes, lang_id 0/en) against the nemo-exported fp32 graphs, using the
same I/O names and layouts as ``src/models/nemotron-rnnt/executor.ts``, and
checks each window against the NeMo model itself run with the matching
attention context (ground truth):

  * numeric encoder parity per chunk (max abs error over output frames)
  * exact token-sequence equality through the RNN-T decode loop
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

NEMO_DIR = Path("N:/models/onnx/nemo/nemotron-3.5-asr-streaming-fp32-nemo")
NEMO_NEMO = Path("N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo")
FIXTURE = "tools/data/fixtures/audio/jfk-short.wav"
LANGUAGE = "en"
LANG_ID = 0

# Streaming operating points: window duration -> chunk frames / att context.
WINDOW_GEOMETRY = {
    80: {"chunk_frames": 17, "att_context_right": 0},
    160: {"chunk_frames": 25, "att_context_right": 0},
    320: {"chunk_frames": 41, "att_context_right": 3},
    560: {"chunk_frames": 65, "att_context_right": 6},
    1120: {"chunk_frames": 121, "att_context_right": 13},
}


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
    p.add_argument("--windows", default="80,160,320,560,1120")
    p.add_argument("--tag", default="fp32", help="evidence filename tag (fp32|fp16)")
    p.add_argument("--enc-tolerance", type=float, default=1e-3,
                   help="max abs error gate vs the NeMo encoder (fp32: 1e-3, fp16: 0.05)")
    return p.parse_args()


def out_path(tag: str, window: int) -> Path:
    return Path(f"tools/data/results/nemotron/nemotron-3.5-nemo-{tag}-window{window}-pipeline-2026-09-01.json")


def stream_encoder(enc, mel, chunk_size, cache_ch, cache_t, cache_ch_len):
    """Stream the ONNX encoder over the mel chunk-by-chunk; return frame stack."""
    all_enc = []
    mel_idx = 0
    while mel_idx < mel.shape[0]:
        chunk = mel[mel_idx : mel_idx + chunk_size]
        valid = chunk.shape[0]
        if valid < chunk_size:
            chunk = np.vstack([chunk, np.zeros([chunk_size - valid, 128], dtype=np.float32)])
        feed = {
            "audio_signal": chunk[np.newaxis, ...].astype(np.float32),  # [1, T, mel]
            "length": np.array([valid], dtype=np.int64),
            "cache_last_channel": cache_ch,
            "cache_last_time": cache_t,
            "cache_last_channel_len": cache_ch_len,
            "lang_id": np.array([LANG_ID], dtype=np.int64),
        }
        result = dict(zip([o.name for o in enc.get_outputs()], enc.run(None, feed)))
        all_enc.append(result["outputs"][0])
        cache_ch = result["cache_last_channel_next"]
        cache_t = result["cache_last_time_next"]
        cache_ch_len = result["cache_last_channel_len_next"]
        mel_idx += chunk_size
    return np.concatenate(all_enc, axis=0), cache_ch, cache_t, cache_ch_len


def greedy_decode(dec, jnt, enc_full, blank_id, max_steps=200, max_tokens=100):
    enc_b = enc_full[np.newaxis, ...].astype(np.float32)  # [1, T_total, 1024]
    targets = [blank_id]
    token_ids = []
    h = np.zeros([2, 1, 640], dtype=np.float32)
    c = np.zeros([2, 1, 640], dtype=np.float32)
    last_t = 0
    for _ in range(max_steps):
        targets_arr = np.array([targets], dtype=np.int64)
        dec_outs = dec.run(None, {"targets": targets_arr, "h_in": h, "c_in": c})
        dec_out = dec_outs[0]  # [1, 640, U]
        h = dec_outs[1]
        c = dec_outs[2]
        enc_remaining = enc_b[:, last_t:, :]
        if enc_remaining.shape[1] == 0:
            break
        dec_out_t = np.ascontiguousarray(dec_out.transpose(0, 2, 1))  # [1, U, 640]
        logits = jnt.run(None, {"encoder_output": enc_remaining, "decoder_output": dec_out_t})[0]
        last_logits = logits[0, :, -1, :]  # [T_remaining, V]
        y_emitted = -1
        non_blank_frame = -1
        for t_local in range(last_logits.shape[0]):
            y = int(np.argmax(last_logits[t_local]))
            if y != blank_id:
                non_blank_frame = t_local
                y_emitted = y
                break
        if non_blank_frame < 0:
            last_t += last_logits.shape[0]
            continue
        token_ids.append(y_emitted)
        targets.append(y_emitted)
        last_t += non_blank_frame
        if len(token_ids) >= max_tokens:
            break
    return token_ids


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    dec = ort.InferenceSession(str(model_dir / "decoder.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    jnt = ort.InferenceSession(str(model_dir / "joint.onnx"), sess_options=so, providers=["CPUExecutionProvider"])

    windows = [int(w) for w in args.windows.split(",") if w.strip()]
    for w in windows:
        if w not in WINDOW_GEOMETRY:
            raise SystemExit(f"Unsupported window {w} ms; choose from {sorted(WINDOW_GEOMETRY)}.")

    nemo = EncDecRNNTBPEModelWithPrompt.restore_from(restore_path=str(NEMO_NEMO), map_location="cpu")
    nemo.eval()
    nemo.set_inference_prompt(LANGUAGE)

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

    results = {}
    all_pass = True
    for window in windows:
        geom = WINDOW_GEOMETRY[window]
        chunk_size = geom["chunk_frames"]
        right = geom["att_context_right"]
        variant_dir = model_dir / "windows" / str(window)
        enc_path = variant_dir / "encoder.onnx"
        if not enc_path.exists():
            raise SystemExit(f"Missing encoder variant: {enc_path}")
        enc = ort.InferenceSession(str(enc_path), sess_options=so, providers=["CPUExecutionProvider"])

        # NeMo ground truth with the same attention context.
        nemo.encoder.set_default_att_context_size([56, right])
        nemo.encoder.export_cache_support = True
        cache_ch_t, cache_t_t, cache_ch_len_t = nemo.encoder.get_initial_cache_state(
            batch_size=1, dtype=torch.float32, device="cpu", max_dim=0,
        )
        mel_torch = torch.from_numpy(mel).float().unsqueeze(0)  # [1, T, 128]
        nemo_frames = []
        mel_idx = 0
        with torch.no_grad():
            while mel_idx < mel.shape[0]:
                take = min(chunk_size, mel.shape[0] - mel_idx)
                chunk = mel_torch[:, mel_idx : mel_idx + chunk_size, :]
                if take < chunk_size:
                    pad = torch.zeros([1, chunk_size - take, 128])
                    chunk = torch.cat([chunk, pad], dim=1)
                length_c = torch.tensor([take], dtype=torch.int64)
                encoded, _enc_len, cache_ch_next, cache_t_next, cache_ch_len_next = (
                    nemo.encoder.forward_internal(
                        audio_signal=chunk.transpose(1, 2),
                        length=length_c,
                        cache_last_channel=cache_ch_t,
                        cache_last_time=cache_t_t,
                        cache_last_channel_len=cache_ch_len_t,
                    )
                )
                # [1, D, T'] -> [1, T', D]; the exported graph additionally
                # applies the language-prompt MLP, so mirror it here.
                encoded_t = encoded.transpose(1, 2)
                B, T, D = encoded_t.shape
                prompt = torch.zeros((B, T, int(nemo.num_prompts)), dtype=encoded_t.dtype)
                prompt[:, :, LANG_ID] = 1.0
                encoded_t = nemo.prompt_kernel(torch.cat([encoded_t, prompt], dim=-1))
                nemo_frames.append(encoded_t.squeeze(0).numpy())
                cache_ch_t = cache_ch_next
                cache_t_t = cache_t_next
                cache_ch_len_t = cache_ch_len_next
                mel_idx += chunk_size
        nemo_enc = np.concatenate(nemo_frames, axis=0)  # [T_total, 1024]

        # ONNX encoder stream (fresh cache per window).
        cache_ch = np.zeros([1, 24, 56, 1024], dtype=np.float32)
        cache_t = np.zeros([1, 24, 1024, 8], dtype=np.float32)
        cache_ch_len = np.zeros([1], dtype=np.int64)
        onnx_enc, _, _, _ = stream_encoder(enc, mel, chunk_size, cache_ch, cache_t, cache_ch_len)

        if nemo_enc.shape != onnx_enc.shape:
            raise SystemExit(f"window {window}: shape mismatch {nemo_enc.shape} vs {onnx_enc.shape}")
        diff = np.abs(nemo_enc - onnx_enc)
        max_abs_err = float(diff.max())
        mean_abs_err = float(diff.mean())

        # Token streams: ONNX graphs through the RNN-T decode loop.
        token_ids = greedy_decode(dec, jnt, onnx_enc, blank_id)
        text = "".join(id_to_token.get(t, "?") for t in token_ids).replace("\u2581", " ").strip()
        enc_pass = max_abs_err < args.enc_tolerance
        token_pass = token_ids == greedy_decode(dec, jnt, nemo_enc, blank_id)
        window_pass = token_pass and enc_pass
        all_pass = all_pass and window_pass
        print(
            f"window {window} ms: enc_max_abs={max_abs_err:.3e} enc_mean_abs={mean_abs_err:.3e} "
            f"tokens={len(token_ids)} token_match={token_pass} -> {'PASS' if window_pass else 'FAIL'}",
            flush=True,
        )
        print(f"  text: {text[:120]}", flush=True)

        results[str(window)] = {
            "chunkFrames": chunk_size,
            "attContextRight": right,
            "encoder": {"nemoShape": list(nemo_enc.shape), "onnxShape": list(onnx_enc.shape),
                        "maxAbsErr": max_abs_err, "meanAbsErr": mean_abs_err,
                        "pass": enc_pass},
            "decode": {"tokenCount": len(token_ids), "tokenIds": token_ids, "text": text,
                       "tokenMatch": token_pass, "pass": window_pass},
        }

    report = {
        "schemaVersion": 1,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "encTolerance": args.enc_tolerance,
        "modelDir": str(model_dir),
        "fixture": str(args.fixture),
        "windows": results,
        "allPass": all_pass,
    }
    out = out_path(args.tag, 0)
    out = out.with_name(f"nemotron-3.5-nemo-{args.tag}-windows-pipeline-2026-09-01.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}", flush=True)

    if not all_pass:
        raise SystemExit("ORACLE FAIL: one or more windows diverged from NeMo ground truth")
    print("All windows: PASS", flush=True)


if __name__ == "__main__":
    main()
