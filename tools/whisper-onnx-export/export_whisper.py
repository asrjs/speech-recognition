#!/usr/bin/env python3
"""Export Whisper ONNX artifacts for ASR.js - fully self-contained.

Generates:
  encoder_model.onnx           - mel to encoder hidden states
  decoder_model_merged.onnx    - autoregressive decode (init + KV-cache step)
  decoder_align_model.onnx     - forced alignment with selected cross-attention heads
  manifest.json                - model metadata
  Plus copies tokenizer.json, generation_config.json, config.json

Options:
  --fp16    Also generate fp16 variants
  --int8    Also generate int8 variants (CPU-friendly)

Usage:
  python export_whisper.py openai/whisper-tiny ./output/whisper-tiny
  python export_whisper.py openai/whisper-base ./output/whisper-base --fp16 --int8
"""

import argparse, json, os, shutil
from pathlib import Path
import torch
import onnx
from torch.export import Dim
from transformers import WhisperForConditionalGeneration
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import quantize_dynamic, QuantType


def load_model(model_id):
    print(f"Loading {model_id}...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    return model


def get_config(model, model_id):
    cfg = model.config
    gen = getattr(model, "generation_config", None) or {}
    ah = getattr(gen, "alignment_heads", None) or []
    return {
        "model_id": model_id,
        "n_mels": cfg.num_mel_bins,
        "d_model": cfg.d_model,
        "max_source_positions": cfg.max_source_positions,
        "max_target_positions": cfg.max_target_positions,
        "vocab_size": cfg.vocab_size,
        "decoder_layers": cfg.decoder_layers,
        "decoder_attention_heads": cfg.decoder_attention_heads,
        "encoder_layers": cfg.encoder_layers,
        "alignment_heads": ah,
        "median_filter_width": getattr(cfg, "median_filter_width", 7),
    }


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder
    def forward(self, mel):
        return self.encoder(mel).last_hidden_state


class DecoderMergedWrapper(torch.nn.Module):
    """Merged decoder handling both init (first-step) and cache-step.

    ONNX inputs:
      input_ids              : int64 [batch, seq]     - full prompt on first step, single token on cache step
      encoder_hidden_states  : float  [batch, enc_seq, d_model]
      use_cache_branch       : bool   [1]              - True on first step, False on cache step
      past_key_values.N.decoder.key   : float  [batch, heads, past_len, head_dim]
      past_key_values.N.decoder.value : float  [batch, heads, past_len, head_dim]
      past_key_values.N.encoder.key   : float  [batch, heads, enc_seq, head_dim]
      past_key_values.N.encoder.value : float  [batch, heads, enc_seq, head_dim]

    ONNX outputs:
      logits                 : float  [batch, seq, vocab]
      present.N.decoder.key  : float  [batch, heads, past_len + new_tokens, head_dim]
      present.N.decoder.value: float
      present.N.encoder.key  : float
      present.N.encoder.value: float
    """
    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
        self.n_layers = model.config.decoder_layers
        self.n_heads = model.config.decoder_attention_heads
        self.head_dim = model.config.d_model // model.config.decoder_attention_heads

    def forward(self, input_ids, encoder_hidden_states, use_cache_branch,
                *past_key_values):
        from transformers.cache_utils import DynamicCache

        # Build cache from past_key_values if not first step
        cache = DynamicCache()
        if not torch.all(use_cache_branch.bool()):
            for i in range(self.n_layers):
                k0 = past_key_values[i * 4]
                k1 = past_key_values[i * 4 + 1]
                k2 = past_key_values[i * 4 + 2]
                k3 = past_key_values[i * 4 + 3]
                cache.update(k0, k1, i, cache_key="self_attn")
                cache.update(k2, k3, i, cache_key="encoder_attn")

        outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=cache if not torch.all(use_cache_branch.bool()) else None,
            use_cache=True,
            return_dict=True,
        )

        logits = self.proj_out(outputs.last_hidden_state)

        # Flatten present cache
        present = []
        if outputs.past_key_values is not None:
            # EncoderDecoderCache supports iteration (but not indexing)
            for layer_pkv in outputs.past_key_values:
                present.extend(layer_pkv)
        else:
            # First step with no cache: return empty past cache
            # Actual decoder will use zero-length tensors for decoder cache
            batch = encoder_hidden_states.shape[0]
            enc_seq = encoder_hidden_states.shape[1]
            for i in range(self.n_layers):
                present.extend([
                    torch.zeros(batch, self.n_heads, 0, self.head_dim),
                    torch.zeros(batch, self.n_heads, 0, self.head_dim),
                    torch.zeros(batch, self.n_heads, enc_seq, self.head_dim),
                    torch.zeros(batch, self.n_heads, enc_seq, self.head_dim),
                ])

        return logits, *present


class DecoderAlignWrapper(torch.nn.Module):
    def __init__(self, model, alignment_heads):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
        self.alignment_heads = alignment_heads

    def forward(self, input_ids, encoder_hidden_states):
        outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        logits = self.proj_out(outputs.last_hidden_state)

        selected = []
        if self.alignment_heads and outputs.cross_attentions:
            for layer_idx, head_idx in self.alignment_heads:
                if layer_idx < len(outputs.cross_attentions):
                    attn = outputs.cross_attentions[layer_idx]
                    selected.append(attn[:, head_idx:head_idx + 1, :, :])

        selected_cross = (torch.cat(selected, dim=1) if selected
                          else torch.zeros(1, 1, 1, 1))
        return logits, selected_cross


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_encoder(model, output_dir, config, suffix=""):
    wrapper = EncoderWrapper(model)
    n_mels = config["n_mels"]
    mel = torch.randn(1, n_mels, 3000)

    fname = f"encoder_model{suffix}.onnx"
    path = output_dir / fname
    torch.onnx.export(
        wrapper, (mel,), str(path),
        input_names=["mel"],
        output_names=["encoder_hidden_states"],
        dynamic_shapes={"mel": {0: Dim("batch"), 2: Dim("num_frames", min=1, max=3000)}},
        opset_version=18,
    )
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  {fname}  ({size_mb:.1f} MB)")
    return fname


def export_decoder_merged(model, output_dir, config, suffix=""):
    wrapper = DecoderMergedWrapper(model)
    n_layers = config["decoder_layers"]
    n_heads = config["decoder_attention_heads"]
    head_dim = config["d_model"] // n_heads
    d_model = config["d_model"]
    enc_frames = config["max_source_positions"] // 2

    # First-step trace: full prompt
    input_ids = torch.zeros(1, 4, dtype=torch.int64)  # SOT + lang + task + notimestamps
    enc_out = torch.randn(1, enc_frames, d_model)
    use_cache_branch = torch.tensor([1], dtype=torch.bool)

    # Empty past_key_values for first step
    pkv = []
    pkv_names = []
    for i in range(n_layers):
        for kv_type in ["decoder.key", "decoder.value", "encoder.key", "encoder.value"]:
            if "decoder" in kv_type:
                pkv.append(torch.zeros(1, n_heads, 0, head_dim))
            else:
                pkv.append(torch.zeros(1, n_heads, enc_frames, head_dim))
            pkv_names.append(f"past_key_values.{i}.{kv_type}")

    args = [input_ids, enc_out, use_cache_branch] + pkv
    input_names = ["input_ids", "encoder_hidden_states", "use_cache_branch"] + pkv_names

    output_names = ["logits"]
    for i in range(n_layers):
        for kv_type in ["decoder.key", "decoder.value", "encoder.key", "encoder.value"]:
            output_names.append(f"present.{i}.{kv_type}")

    fname = f"decoder_model_merged{suffix}.onnx"
    path = output_dir / fname

    torch.onnx.export(
        wrapper, tuple(args), str(path),
        input_names=input_names,
        output_names=output_names,
        dynamo=False,
        opset_version=20,
    )
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  {fname}  ({size_mb:.1f} MB)")
    return fname


def export_decoder_align(model, output_dir, config, suffix=""):
    ah = config.get("alignment_heads", [])
    model.config._attn_implementation = "eager"
    wrapper = DecoderAlignWrapper(model, ah)
    wrapper.eval()
    model.eval()  # Ensure eval mode for export
    enc_frames = config["max_source_positions"] // 2
    d_model = config["d_model"]
    input_ids = torch.zeros(1, 10, dtype=torch.int64)
    enc_out = torch.randn(1, enc_frames, d_model)

    fname = f"decoder_align_model{suffix}.onnx"
    path = output_dir / fname

    torch.onnx.export(
        wrapper, (input_ids, enc_out), str(path),
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["logits", "selected_cross_attentions"],
        opset_version=18,
    )
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  {fname}  ({size_mb:.1f} MB)")
    return fname


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def convert_fp16(model_dir, names):
    """Convert fp32 models to fp16."""
    imported = False
    for name in names:
        src = model_dir / name.replace(".fp16", "")
        dst = model_dir / name
        if not src.exists():
            continue
        if not imported:
            from onnxconverter_common import float16
            imported = True
        model = onnx.load(str(src))
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, str(dst))
        size_mb = os.path.getsize(dst) / 1024 / 1024
        print(f"  {name}  ({size_mb:.1f} MB)")


def convert_int8(model_dir, names):
    """Quantize to int8 using ONNX Runtime dynamic quantization."""
    for name in names:
        src = model_dir / name.replace("_int8", "")
        dst = model_dir / name
        if not src.exists():
            continue
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        size_mb = os.path.getsize(dst) / 1024 / 1024
        print(f"  {name}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Tokenizer / config files
# ---------------------------------------------------------------------------

def copy_tokenizer_files(model_id, output_dir):
    for filename in ["tokenizer.json", "generation_config.json", "config.json"]:
        try:
            local = hf_hub_download(model_id, filename)
            shutil.copy(local, output_dir / filename)
            print(f"  {filename}")
        except Exception as e:
            print(f"  SKIP {filename}: {e}")


def generate_manifest(config, output_dir, variants=None):
    manifest = {
        "format_version": 1,
        "architecture": "whisper",
        "model_id": config["model_id"],
        "audio": {
            "sample_rate": 16000,
            "chunk_length_seconds": 30,
            "num_frames": 3000,
            "n_mels": config["n_mels"],
        },
        "model": {
            "d_model": config["d_model"],
            "decoder_layers": config["decoder_layers"],
            "decoder_attention_heads": config["decoder_attention_heads"],
            "vocab_size": config["vocab_size"],
        },
        "alignment": {
            "strategy": "cross_attention_dtw",
            "alignment_heads": config["alignment_heads"],
            "attention_output": "selected_cross_attentions",
            "median_filter_width": config["median_filter_width"],
        },
        "onnx": {
            "encoder": "encoder_model.onnx",
            "decoder": "decoder_model_merged.onnx",
            "decoder_align": "decoder_align_model.onnx",
        },
        "supports": {
            "transcription": True,
            "translation": True,
            "word_timestamps": True,
            "token_logprobs": True,
            "kv_cache": True,
        },
    }
    if variants:
        manifest["variants"] = variants

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export Whisper ONNX for ASR.js")
    parser.add_argument("model_id")
    parser.add_argument("output_dir")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_id)
    config = get_config(model, args.model_id)

    print(f"Exporting {args.model_id} -> {output_dir}")
    print(f"  n_mels={config['n_mels']}  layers={config['decoder_layers']}  heads={config['decoder_attention_heads']}")
    print(f"  alignment_heads={config['alignment_heads']}")
    print()

    # fp32 (always)
    print("Exporting fp32 models:")
    export_encoder(model, output_dir, config)
    export_decoder_merged(model, output_dir, config)
    export_decoder_align(model, output_dir, config)

    variants = {"fp32": {
        "encoder": "encoder_model.onnx",
        "decoder": "decoder_model_merged.onnx",
        "decoder_align": "decoder_align_model.onnx",
    }}

    # fp16
    if args.fp16:
        print("\nConverting to fp16:")
        fp16_names = ["encoder_model.fp16.onnx",
                       "decoder_model_merged.fp16.onnx",
                       "decoder_align_model.fp16.onnx"]
        convert_fp16(output_dir, fp16_names)
        variants["fp16"] = {
            "encoder": "encoder_model.fp16.onnx",
            "decoder": "decoder_model_merged.fp16.onnx",
            "decoder_align": "decoder_align_model.fp16.onnx",
        }

    # int8
    if args.int8:
        print("\nQuantizing to int8:")
        int8_names = ["encoder_model_int8.onnx",
                       "decoder_model_merged_int8.onnx",
                       "decoder_align_model_int8.onnx"]
        convert_int8(output_dir, int8_names)
        variants["int8"] = {
            "encoder": "encoder_model_int8.onnx",
            "decoder": "decoder_model_merged_int8.onnx",
            "decoder_align": "decoder_align_model_int8.onnx",
        }

    print("\nCopying config files:")
    copy_tokenizer_files(args.model_id, output_dir)

    print("\nGenerating manifest:")
    generate_manifest(config, output_dir, variants)

    print(f"\nDone! All artifacts in {output_dir}")


if __name__ == "__main__":
    main()
