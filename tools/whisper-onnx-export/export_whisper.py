#!/usr/bin/env python3
"""Export Whisper ONNX artifacts for ASR.js with alignment support.

Generates:
  encoder_model.onnx         - mel features to encoder hidden states
  decoder_align_model.onnx   - forced alignment with selected cross-attention heads
  manifest.json              - model metadata
  Plus copies tokenizer.json, generation_config.json, config.json

The decoder_model_merged.onnx is sourced from onnx-community/*_timestamped repos.

Usage:
  python export_whisper.py openai/whisper-tiny ./output/whisper-tiny
"""

import argparse, json, os, shutil
from pathlib import Path
import torch
from torch.export import Dim
from transformers import WhisperForConditionalGeneration
from huggingface_hub import hf_hub_download


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


class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, mel):
        return self.encoder(mel).last_hidden_state


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

        if selected:
            selected_cross = torch.cat(selected, dim=1)
        else:
            selected_cross = torch.zeros(1, 1, 1, 1)

        return logits, selected_cross


def export_encoder(model, output_dir, config):
    wrapper = EncoderWrapper(model)
    wrapper.eval()
    n_mels = config["n_mels"]
    mel = torch.randn(1, n_mels, 3000)

    path = output_dir / "encoder_model.onnx"
    torch.onnx.export(
        wrapper, (mel,), str(path),
        input_names=["mel"],
        output_names=["encoder_hidden_states"],
        dynamic_shapes={"mel": {0: Dim("batch"), 2: Dim("num_frames", min=1, max=3000)}},
        opset_version=18,
    )
    print(f"  encoder_model.onnx  ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")


def export_decoder_align(model, output_dir, config):
    ah = config.get("alignment_heads", [])
    # Force eager attention to get cross_attentions (sdpa does not support output_attentions)
    model.config._attn_implementation = "eager"
    wrapper = DecoderAlignWrapper(model, ah)
    wrapper.eval()

    n_audio_ctx = config["max_source_positions"]
    d_model = config["d_model"]
    input_ids = torch.zeros(1, 10, dtype=torch.int64)
    enc_out = torch.randn(1, n_audio_ctx // 2, d_model)

    path = output_dir / "decoder_align_model.onnx"
    torch.onnx.export(
        wrapper, (input_ids, enc_out), str(path),
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["logits", "selected_cross_attentions"],
        opset_version=18,
    )
    print(f"  decoder_align_model.onnx  ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")


def copy_tokenizer_files(model_id, output_dir):
    for filename in ["tokenizer.json", "generation_config.json", "config.json"]:
        try:
            local = hf_hub_download(model_id, filename)
            shutil.copy(local, output_dir / filename)
            print(f"  {filename}")
        except Exception as e:
            print(f"  SKIP {filename}: {e}")


def generate_manifest(config, output_dir):
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
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_id)
    config = get_config(model, args.model_id)

    print(f"Exporting {args.model_id} -> {output_dir}")
    print(f"  n_mels={config['n_mels']}, decoder_layers={config['decoder_layers']}")
    print(f"  alignment_heads={config['alignment_heads']}")

    export_encoder(model, output_dir, config)
    export_decoder_align(model, output_dir, config)

    print("\nCopying config files:")
    copy_tokenizer_files(args.model_id, output_dir)

    print("\nGenerating manifest:")
    generate_manifest(config, output_dir)

    print(f"\nDone! Output in {output_dir}")
    short_id = args.model_id.split("/")[-1]
    print(f"Tip: Copy decoder_model_merged.onnx from onnx-community/{short_id}_timestamped")


if __name__ == "__main__":
    main()
