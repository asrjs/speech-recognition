#!/usr/bin/env python3
"""
Generate reference JSON for splitgraph reproducibility comparison.

Runs both PyTorch Whisper generate() and splitgraph ONNX Runtime inference
on a given audio file, producing a comparison artifact for TypeScript tests.

Usage:
  python generate_hf_reference.py \\
    --model-dir /path/to/exported/whisper-tiny \\
    --audio /path/to/audio.wav \\
    --output reference.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import soundfile as sf
from transformers import (
    AutoTokenizer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
)


def load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio, resample to target_sr if needed, return (samples, sr)."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo → mono
    # Resample if needed (simple linear — for fixtures this should be exact 16k)
    if sr != target_sr:
        import scipy.signal
        num = int(len(data) * target_sr / sr)
        data = scipy.signal.resample(data, num)
        sr = target_sr
    return data.astype(np.float32), sr


def pytorch_generate(
    model,
    tokenizer,
    input_features,
    max_new_tokens: int = 128,
    language: str = "en",
    task: str = "transcribe",
    return_timestamps: bool = False,
) -> dict:
    """Run PyTorch generate and return tokens, text, optional timestamps."""
    with torch.no_grad():
        generated = model.generate(
            input_features=input_features,
            max_new_tokens=max_new_tokens,
            language=language,
            task=task,
            return_timestamps=return_timestamps,
            output_scores=False,
            return_dict_in_generate=True,
        )

    if return_timestamps:
        # When return_timestamps=True, output is a dict with 'sequences' and 'segments'
        tokens = generated["sequences"][0].tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        timestamp_tokens = [t for t in tokens if t >= 50364]
        return {
            "tokens": tokens,
            "text": text,
            "timestamp_tokens": timestamp_tokens,
            "num_segments": len(generated.get("segments", [])),
        }
    else:
        tokens = generated.sequences[0].tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        return {"tokens": tokens, "text": text}


def suppress_logits(logits: np.ndarray, token_ids: list[int] | None):
    if token_ids is None:
        return
    for tid in token_ids:
        if 0 <= tid < logits.shape[-1]:
            logits[..., tid] = -np.inf


def onnx_infer(
    ort_sessions: dict,
    manifest: dict,
    mel_features: torch.Tensor,
    prompt_ids: list[int],
    max_new_tokens: int = 128,
) -> dict:
    """Run splitgraph ONNX inference and return tokens, text."""
    enc_sess = ort_sessions["encoder"]
    init_sess = ort_sessions["decoder_init"]
    step_sess = ort_sessions["decoder_step"]
    align_sess = ort_sessions.get("decoder_align")

    eos = manifest["special_tokens"]["eos_token_id"]
    suppress = manifest["special_tokens"].get("suppress_tokens")
    begin_suppress = manifest["special_tokens"].get("begin_suppress_tokens")

    mel_arr = mel_features.numpy().astype(np.float32)

    # Encoder
    enc_out = enc_sess.run(["last_hidden_state"], {"input_features": mel_arr})[0]

    # Decoder init
    prompt_arr = np.array([prompt_ids], dtype=np.int64)
    init_out = init_sess.run(None, {"input_ids": prompt_arr, "encoder_hidden_states": enc_out})
    init_names = [o.name for o in init_sess.get_outputs()]

    logits = init_out[0]  # [1, prompt_len, vocab]
    last_logits = logits[0, -1, :].copy()
    suppress_logits(last_logits, suppress)
    suppress_logits(last_logits, begin_suppress)
    next_token = int(np.argmax(last_logits))

    past_kv = {}
    for name, val in zip(init_names[1:], init_out[1:]):
        past_kv[name.replace("present.", "past_key_values.")] = val

    generated = [next_token]

    # Step loop
    for _ in range(max_new_tokens - 1):
        step_input = np.array([[next_token]], dtype=np.int64)
        step_out = step_sess.run(None, {"input_ids": step_input, **past_kv})
        step_names = [o.name for o in step_sess.get_outputs()]

        logits_s = step_out[0]
        last_logits = logits_s[0, -1, :].copy()
        suppress_logits(last_logits, suppress)
        next_token = int(np.argmax(last_logits))

        if next_token == eos:
            break

        generated.append(next_token)
        for name, val in zip(step_names[1:], step_out[1:]):
            past_kv[name.replace("present.", "past_key_values.")] = val

    all_tokens = prompt_ids + generated

    # Alignment (optional)
    alignment_info = None
    if align_sess and generated:
        align_input = np.array([all_tokens], dtype=np.int64)
        align_out = align_sess.run(None, {"input_ids": align_input, "encoder_hidden_states": enc_out})
        alignment = align_out[0]  # [B, T, S]
        b, t, s = alignment.shape
        # Skip prompt rows: alignment[:, prompt_len:, :]
        text_align = alignment[0, len(prompt_ids):, :]
        text_t, text_s = text_align.shape
        row_sums = text_align.sum(axis=1)
        alignment_info = {
            "shape": [b, t, s],
            "text_shape": [text_t, text_s],
            "row_sum_min": float(row_sums.min()),
            "row_sum_max": float(row_sums.max()),
            "row_sum_mean": float(row_sums.mean()),
        }

    return {
        "tokens": all_tokens,
        "generated": generated,
        "alignment": alignment_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate HF Whisper reference JSON")
    parser.add_argument("--model-dir", required=True, help="Exported splitgraph model directory")
    parser.add_argument("--audio", required=True, help="Audio file (WAV, 16kHz mono preferred)")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--model-id", default="openai/whisper-tiny", help="HF model ID")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--export-mel", action="store_true",
                        help="Also export mel features as .npy for feature-input mode")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    audio_path = args.audio
    output_path = Path(args.output)

    # Load audio
    print(f"Loading audio: {audio_path}")
    audio_np, sr = load_audio(audio_path)
    print(f"  Sample rate: {sr}, samples: {len(audio_np)}, duration: {len(audio_np)/sr:.2f}s")

    # Load PyTorch model
    print(f"Loading PyTorch model: {args.model_id}")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_id)

    # Compute mel features
    inputs = feature_extractor(audio_np, sampling_rate=sr, return_tensors="pt")
    mel = inputs.input_features
    if mel.shape[-1] < 3000:
        pad = torch.zeros(1, mel.shape[1], 3000 - mel.shape[-1])
        mel = torch.cat([mel, pad], dim=-1)
    mel = mel[:, :, :3000]
    print(f"Mel shape: {mel.shape}")

    # Build prompt
    sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    lang = tokenizer.convert_tokens_to_ids(f"<|{args.language}|>")
    task_tok = tokenizer.convert_tokens_to_ids("<|transcribe|>")
    prompt_ids = [sot, lang, task_tok]

    # PyTorch: no_timestamps
    print("Running PyTorch generate (no_timestamps)...")
    pt_no_ts = pytorch_generate(
        model, tokenizer, mel, max_new_tokens=args.max_new_tokens,
        language=args.language, return_timestamps=False,
    )
    print(f"  Tokens: {pt_no_ts['tokens'][:20]}...")
    print(f"  Text:   \"{pt_no_ts['text'][:100]}\"")

    # PyTorch: with timestamps
    print("Running PyTorch generate (with timestamps)...")
    pt_with_ts = pytorch_generate(
        model, tokenizer, mel, max_new_tokens=args.max_new_tokens,
        language=args.language, return_timestamps=True,
    )
    print(f"  Tokens: {pt_with_ts['tokens'][:20]}...")
    print(f"  Timestamp tokens: {pt_with_ts.get('timestamp_tokens', [])[:10]}")
    print(f"  Text:   \"{pt_with_ts['text'][:100]}\"")

    # ONNX splitgraph inference
    print("Running ONNX splitgraph inference...")
    manifest_path = model_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_sessions = {
        "encoder": ort.InferenceSession(
            str(model_dir / "encoder_model.onnx"), sess_opts, providers=["CPUExecutionProvider"]
        ),
        "decoder_init": ort.InferenceSession(
            str(model_dir / "decoder_init.onnx"), sess_opts, providers=["CPUExecutionProvider"]
        ),
        "decoder_step": ort.InferenceSession(
            str(model_dir / "decoder_step.onnx"), sess_opts, providers=["CPUExecutionProvider"]
        ),
    }
    if (model_dir / "decoder_align.onnx").exists():
        ort_sessions["decoder_align"] = ort.InferenceSession(
            str(model_dir / "decoder_align.onnx"), sess_opts, providers=["CPUExecutionProvider"]
        )

    onnx_no_ts = onnx_infer(ort_sessions, manifest, mel, prompt_ids, args.max_new_tokens)
    onnx_no_ts_text = tokenizer.decode(onnx_no_ts["tokens"], skip_special_tokens=True)
    print(f"  ONNX tokens: {onnx_no_ts['tokens'][:20]}...")
    print(f"  ONNX text:   \"{onnx_no_ts_text[:100]}\"")
    if onnx_no_ts["alignment"]:
        al = onnx_no_ts["alignment"]
        print(f"  Alignment: shape={al['shape']}, text_shape={al['text_shape']}, "
              f"row_sum=[{al['row_sum_min']:.4f}, {al['row_sum_max']:.4f}]")

    # Build reference
    reference = {
        "audio": {
            "path": str(Path(audio_path).resolve()),
            "sample_rate": sr,
            "duration_seconds": round(len(audio_np) / sr, 3),
            "num_samples": len(audio_np),
        },
        "model": {
            "id": args.model_id,
            "export_dir": str(model_dir.resolve()),
            "format": manifest.get("format", "unknown"),
            "d_model": manifest.get("d_model"),
            "decoder_layers": manifest.get("decoder_layers"),
            "decoder_attention_heads": manifest.get("decoder_attention_heads"),
        },
        "prompt_ids": prompt_ids,
        "pytorch": {
            "no_timestamps": pt_no_ts,
            "with_timestamps": pt_with_ts,
        },
        "onnx_python": onnx_no_ts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"\nReference written to {output_path}")

    # Export mel features for feature-input mode
    if args.export_mel:
        mel_path = output_path.with_suffix(".mel.npy")
        np.save(mel_path, mel.numpy().astype(np.float32))
        print(f"Mel features exported to {mel_path}")
        reference["mel_features_path"] = str(mel_path.resolve())


if __name__ == "__main__":
    main()
