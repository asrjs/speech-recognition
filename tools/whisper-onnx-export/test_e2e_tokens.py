#!/usr/bin/env python3
"""End-to-end ONNX vs PyTorch token output comparison test.

Exports whisper-tiny to 4-graph ONNX, runs ONNX Runtime inference,
and compares generated token sequences with PyTorch generate() output.

Usage:
  python test_e2e_tokens.py
"""

import json
import sys
from pathlib import Path
import tempfile

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration

# Add export tool to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_whisper import export_all


def pytorch_generate(model, input_features, prompt_ids, max_new_tokens=20):
    """Run PyTorch generate() and return token IDs."""
    with torch.no_grad():
        generated = model.generate(
            input_features=input_features,
            max_new_tokens=max_new_tokens,
            language="en",
            task="transcribe",
            return_timestamps=False,
            output_scores=False,
            return_dict_in_generate=True,
        )
    return generated.sequences[0].tolist()


def suppress_logits(logits: np.ndarray, token_ids: list[int] | None):
    """Set suppressed token logits to -inf."""
    if token_ids is None:
        return
    for tid in token_ids:
        if 0 <= tid < logits.shape[-1]:
            logits[..., tid] = -np.inf


def onnx_infer(ort_sessions, manifest, mel_features, prompt_ids, max_new_tokens=20):
    """Run ONNX Runtime inference: encoder -> init -> step loop -> token list."""
    enc_sess = ort_sessions["encoder"]
    init_sess = ort_sessions["decoder_init"]
    step_sess = ort_sessions["decoder_step"]

    eos_token_id = manifest["special_tokens"]["eos_token_id"]
    suppress_tokens = manifest["special_tokens"].get("suppress_tokens")
    begin_suppress_tokens = manifest["special_tokens"].get("begin_suppress_tokens")

    # ---- Encoder ----
    mel_arr = mel_features.numpy().astype(np.float32)
    enc_out = enc_sess.run(["last_hidden_state"], {"input_features": mel_arr})[0]

    # ---- Decoder init ----
    prompt_arr = np.array([prompt_ids], dtype=np.int64)
    init_feeds = {
        "input_ids": prompt_arr,
        "encoder_hidden_states": enc_out,
    }
    init_outputs = init_sess.run(None, init_feeds)
    init_names = [o.name for o in init_sess.get_outputs()]

    # Last-position logits
    logits = init_outputs[0]  # [1, prompt_len, vocab]
    last_logits = logits[0, -1, :]

    # Apply suppress + begin_suppress (match HF's logits processors)
    suppress_logits(last_logits, suppress_tokens)
    suppress_logits(last_logits, begin_suppress_tokens)

    next_token = int(np.argmax(last_logits))

    # Collect present KV from init
    past_kv = {}
    init_kv_names = init_names[1:]  # skip logits
    for name, val in zip(init_kv_names, init_outputs[1:]):
        past_kv[name.replace("present.", "past_key_values.")] = val

    generated_tokens = [next_token]

    # ---- Decoder step loop ----
    for _step in range(max_new_tokens - 1):
        step_input = np.array([[next_token]], dtype=np.int64)
        feeds = {"input_ids": step_input, **past_kv}

        step_out = step_sess.run(None, feeds)
        step_names = [o.name for o in step_sess.get_outputs()]

        logits = step_out[0]  # [1, 1, vocab]
        last_logits = logits[0, -1, :]
        suppress_logits(last_logits, suppress_tokens)
        next_token = int(np.argmax(last_logits))

        if next_token == eos_token_id:
            break

        generated_tokens.append(next_token)

        # Update ONLY self-attention KV from step outputs.
        # Encoder (cross-attention) KV is static — keep from init.
        for name, val in zip(step_names[1:], step_out[1:]):
            past_name = name.replace("present.", "past_key_values.")
            past_kv[past_name] = val

    return prompt_ids + generated_tokens


def test_e2e_tokens():
    """Main test: export model, run ONNX, compare with PyTorch."""
    model_id = "openai/whisper-tiny"

    # ---- 1. Load PyTorch model ----
    print(f"Loading PyTorch model: {model_id}...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)

    # ---- 2. Create synthetic audio (1s 440Hz sine, mono 16kHz) ----
    sample_rate = 16000
    duration = 1.0
    t = torch.arange(0, int(sample_rate * duration), dtype=torch.float32) / sample_rate
    audio = torch.sin(2 * torch.pi * 440.0 * t) * 0.3  # [samples]
    audio_np = audio.numpy()

    # ---- 3. Compute mel features via HF feature extractor ----
    inputs = feature_extractor(
        audio_np,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    mel_features = inputs.input_features  # [1, n_mels, 3000]
    print(f"Mel features shape: {mel_features.shape}")

    # Ensure 3000 frames
    if mel_features.shape[-1] < 3000:
        pad = torch.zeros(1, mel_features.shape[1], 3000 - mel_features.shape[-1])
        mel_features = torch.cat([mel_features, pad], dim=-1)
    mel_features = mel_features[:, :, :3000]
    print(f"Mel features (padded): {mel_features.shape}")

    # ---- 4. PyTorch inference ----
    sot_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    lang_id = tokenizer.convert_tokens_to_ids("<|en|>")
    task_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")
    no_ts_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    prompt_ids = [sot_id, lang_id, task_id, no_ts_id]
    print(f"Prompt IDs: {prompt_ids}")

    pt_tokens = pytorch_generate(model, mel_features, prompt_ids, max_new_tokens=10)
    pt_text = tokenizer.decode(pt_tokens, skip_special_tokens=True)
    print(f"PyTorch generated tokens: {len(pt_tokens)} tokens")
    print(f"PyTorch decoded text: '{pt_text}'")

    # ---- 5. Export to ONNX ----
    print(f"\nExporting 4-graph ONNX...")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "whisper-tiny-e2e"
        out_dir.mkdir()

        export_all(
            model_id=model_id,
            output_dir=out_dir,
            opset=17,
            prompt_len=4,
            past_len=4,
        )

        # ---- 6. Load ONNX sessions ----
        print(f"Loading ONNX Runtime sessions...")
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        enc_sess = ort.InferenceSession(
            str(out_dir / "encoder_model.onnx"),
            sess_opts,
            providers=["CPUExecutionProvider"],
        )
        init_sess = ort.InferenceSession(
            str(out_dir / "decoder_init.onnx"),
            sess_opts,
            providers=["CPUExecutionProvider"],
        )
        step_sess = ort.InferenceSession(
            str(out_dir / "decoder_step.onnx"),
            sess_opts,
            providers=["CPUExecutionProvider"],
        )

        # ---- 7. Load manifest ----
        with open(out_dir / "manifest.json") as f:
            manifest = json.load(f)

        ort_sessions = {
            "encoder": enc_sess,
            "decoder_init": init_sess,
            "decoder_step": step_sess,
        }

        # ---- 8. ONNX inference ----
        onnx_tokens = onnx_infer(
            ort_sessions, manifest, mel_features, prompt_ids, max_new_tokens=10
        )
        onnx_text = tokenizer.decode(onnx_tokens, skip_special_tokens=True)
        print(f"\nONNX generated tokens: {len(onnx_tokens)} tokens")
        print(f"ONNX decoded text: '{onnx_text}'")

        # ---- 9. Compare ----
        print(f"\n--- Comparison ---")
        print(f"PyTorch tokens ({len(pt_tokens)}): {pt_tokens}")
        print(f"ONNX    tokens ({len(onnx_tokens)}): {onnx_tokens}")

        # For a 1s 440Hz sine, both should produce similar output
        # (likely silence/meaningless text but tokens should match)
        if pt_tokens == onnx_tokens:
            print(f"\n^ EXACT MATCH -- PyTorch and ONNX produce identical token sequences")
        else:
            min_len = min(len(pt_tokens), len(onnx_tokens))
            matches = sum(1 for i in range(min_len) if pt_tokens[i] == onnx_tokens[i])
            pct = 100 * matches / max(min_len, 1)
            print(f"\n  Matching prefix: {matches}/{min_len} tokens ({pct:.0f}%)")

            if pct >= 80:
                print(f"  ^ CLOSE MATCH -- >80% token agreement")
            else:
                print(f"  x MISMATCH -- only {matches}/{min_len} tokens agree")
                print(f"  PyTorch text: '{pt_text}'")
                print(f"  ONNX text:    '{onnx_text}'")
                raise AssertionError(
                    f"Token mismatch: {matches}/{min_len} tokens agree ({pct:.0f}%). "
                    f"PyTorch='{pt_text}', ONNX='{onnx_text}'"
                )

    print(f"\n^ E2E token comparison test passed\n")


if __name__ == "__main__":
    test_e2e_tokens()
