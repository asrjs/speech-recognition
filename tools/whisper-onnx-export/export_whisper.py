#!/usr/bin/env python3
"""Export Whisper ONNX artifacts for ASR.js — 4-graph KV-cache decoder split.

Generates:
  encoder_model.onnx           — log-Mel features to encoder hidden states
  decoder_init.onnx            — prompt/prefill decoder, creates initial KV cache
  decoder_step.onnx            — single-token autoregressive step with KV cache reuse
  decoder_align.onnx           — forced cross-attention alignment for word timestamps
  manifest.json                — model metadata
  Plus copies tokenizer.json, generation_config.json, config.json

Options:
  --fp16    Also generate fp16 variants (float16 conversion)
  --int8    Also generate int8 variants (dynamic quantization)

Why 4 graphs instead of a merged decoder:
  - decoder_init runs once to build the initial cache from the prompt tokens
  - decoder_step is branch-free and fast for the autoregressive loop
  - decoder_align is run once after generation to extract cross-attention for DTW
  - This avoids a fragile merged decoder with ONNX If branches and
    DynamicCache data-dependent tracing that torch.onnx.export cannot capture

Usage:
  python export_whisper.py openai/whisper-tiny ./output/whisper-tiny
  python export_whisper.py openai/whisper-base ./output/whisper-base --fp16 --int8
  python export_whisper.py openai/whisper-base ./output --alignment-heads "2:3,3:5"
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, WhisperForConditionalGeneration
from huggingface_hub import hf_hub_download

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# ---------------------------------------------------------------------------
# Cache type aliases and helpers
# ---------------------------------------------------------------------------

LegacyLayerCache = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
LegacyCache = Tuple[LegacyLayerCache, ...]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def has_forward_arg(module: nn.Module, name: str) -> bool:
    try:
        return name in inspect.signature(module.forward).parameters
    except Exception:
        return False


def to_legacy_cache(past_key_values: Any) -> LegacyCache:
    """Convert HF cache outputs into legacy tuple format:
       tuple[layer] = (self_k, self_v, cross_k, cross_v)

    Supports:
      - legacy tuple/list cache
      - modern Cache objects exposing to_legacy_cache()
      - transformers 5.x EncoderDecoderCache wrapping DynamicCache
    """
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()

    if not isinstance(past_key_values, (tuple, list)):
        # HF 5.x EncoderDecoderCache is iterable but not a tuple/list.
        # It yields 6-element tuples: (self_k, self_v, None, cross_k, cross_v, None)
        # where the Nones are DynamicCache metadata slots.
        try:
            legacy: List[LegacyLayerCache] = []
            for layer_idx, layer_cache in enumerate(past_key_values):
                if isinstance(layer_cache, (tuple, list)) and len(layer_cache) == 6:
                    self_k, self_v, _none1, cross_k, cross_v, _none2 = layer_cache
                    legacy.append((self_k, self_v, cross_k, cross_v))
                elif isinstance(layer_cache, (tuple, list)) and len(layer_cache) == 4:
                    self_k, self_v, cross_k, cross_v = layer_cache[:4]
                    legacy.append((self_k, self_v, cross_k, cross_v))
                else:
                    raise ValueError(
                        f"Layer cache {layer_idx}: expected 4 or 6 elements, "
                        f"got {len(layer_cache) if hasattr(layer_cache, '__len__') else 'N/A'}"
                    )
            return tuple(legacy)
        except TypeError:
            pass

        raise TypeError(
            "Unsupported past_key_values type. Expected tuple/list or object with "
            f"to_legacy_cache(), got: {type(past_key_values)}"
        )

    legacy: List[LegacyLayerCache] = []
    for layer_idx, layer_cache in enumerate(past_key_values):
        if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) < 4:
            raise ValueError(
                f"Layer cache {layer_idx} is not a 4-tuple. "
                "Expected (self_k, self_v, cross_k, cross_v). "
                f"Got type={type(layer_cache)}, len={len(layer_cache) if hasattr(layer_cache, '__len__') else 'N/A'}"
            )
        self_k, self_v, cross_k, cross_v = layer_cache[:4]
        legacy.append((self_k, self_v, cross_k, cross_v))

    return tuple(legacy)


def build_encoder_decoder_cache_from_flat(
    flat: Sequence[torch.Tensor], num_layers: int
) -> Any:
    """Build EncoderDecoderCache from flat [self_k, self_v, cross_k, cross_v] per layer."""
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache

    expected = num_layers * 4
    if len(flat) != expected:
        raise ValueError(f"Expected {expected} flat cache tensors, got {len(flat)}")

    self_cache = DynamicCache()
    cross_cache = DynamicCache()
    for i in range(num_layers):
        offset = i * 4
        self_k = flat[offset + 0]
        self_v = flat[offset + 1]
        cross_k = flat[offset + 2]
        cross_v = flat[offset + 3]
        self_cache.update(self_k, self_v, i)
        cross_cache.update(cross_k, cross_v, i)

    return EncoderDecoderCache(self_cache, cross_cache)


def flatten_init_cache_outputs(pkv: LegacyCache) -> List[torch.Tensor]:
    """Flatten all 4 KV tensors per layer for decoder_init outputs."""
    flat: List[torch.Tensor] = []
    for self_k, self_v, cross_k, cross_v in pkv:
        flat.extend([self_k, self_v, cross_k, cross_v])
    return flat


def flatten_step_cache_outputs(pkv: LegacyCache) -> List[torch.Tensor]:
    """Flatten only self-attention KV for decoder_step outputs (cross-attn KV is static)."""
    flat: List[torch.Tensor] = []
    for self_k, self_v, _cross_k, _cross_v in pkv:
        flat.extend([self_k, self_v])
    return flat


# ---------------------------------------------------------------------------
# Model wrapper classes
# ---------------------------------------------------------------------------

class WhisperEncoderWrapper(nn.Module):
    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_features=input_features, return_dict=True)
        return out.last_hidden_state


class WhisperDecoderInitWrapper(nn.Module):
    """Prefill decoder graph.

    Outputs:
      logits
      present.{i}.decoder.key
      present.{i}.decoder.value
      present.{i}.encoder.key
      present.{i}.encoder.value
    """

    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
        self.num_layers = model.config.decoder_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=True,
            return_dict=True,
        )

        logits = self.proj_out(outputs.last_hidden_state)
        pkv = to_legacy_cache(outputs.past_key_values)
        flat_cache = flatten_init_cache_outputs(pkv)
        return tuple([logits] + flat_cache)


class WhisperDecoderStepWrapper(nn.Module):
    """Single-token autoregressive decoder graph.

    Inputs:
      input_ids: [B, 1]
      encoder_hidden_states
      cache_position
      flat past cache tensors:
        past_key_values.{i}.decoder.key
        past_key_values.{i}.decoder.value
        past_key_values.{i}.encoder.key
        past_key_values.{i}.encoder.value

    Outputs:
      logits
      updated self-attention KV only:
        present.{i}.decoder.key
        present.{i}.decoder.value
    """

    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
        self.num_layers = model.config.decoder_layers
        self.accepts_cache_position = has_forward_arg(self.decoder, "cache_position")

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        *flat_past_key_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # encoder_hidden_states is used inside decoder(**kwargs) for cross-attention
        # when past_key_values lacks cross KV (should not happen for step, but the
        # decoder API still requires it). cache_position may be used by the decoder
        # internally depending on the HF version; we pass it through unconditionally.
        pkv = build_encoder_decoder_cache_from_flat(flat_past_key_values, self.num_layers)

        kwargs = dict(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=pkv,
            use_cache=True,
            return_dict=True,
        )

        if self.accepts_cache_position:
            kwargs["cache_position"] = cache_position

        outputs = self.decoder(**kwargs)
        logits = self.proj_out(outputs.last_hidden_state)
        next_pkv = to_legacy_cache(outputs.past_key_values)
        flat_self_cache = flatten_step_cache_outputs(next_pkv)
        return tuple([logits] + flat_self_cache)


class WhisperDecoderAlignWrapper(nn.Module):
    """Manual teacher-forced alignment graph.

    Directly runs decoder blocks and captures only selected encoder_attn
    cross-attention weights. No DTW, torch.diff, timestamp extraction,
    word grouping, or jump detection — all post-processing lives in TypeScript.

    Returns averaged alignment matrix:
      alignment: [B, target_tokens, source_frames]
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        alignment_heads: Sequence[Tuple[int, int]],
    ):
        super().__init__()
        self.decoder = model.model.decoder
        self.alignment_heads = list(alignment_heads)
        self.num_layers = model.config.decoder_layers

        if not self.alignment_heads:
            raise ValueError(
                "alignment_heads is empty. Do not use a blind fallback for production; "
                "load official alignment_heads from generation_config or an OpenAI checkpoint."
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        decoder = self.decoder

        # Token embedding + positional encoding (no torch.diff here)
        positions = decoder.embed_positions(input_ids)
        hidden_states = decoder.embed_tokens(input_ids) + positions

        captured: List[torch.Tensor] = []

        for layer_idx, layer in enumerate(decoder.layers):
            residual = hidden_states

            # Self-attention (no need for attention weights)
            normed = layer.self_attn_layer_norm(hidden_states)
            self_out, _self_weights = layer.self_attn(
                hidden_states=normed,
                attention_mask=None,
                past_key_values=None,
                output_attentions=False,
            )
            hidden_states = residual + self_out

            # Cross-attention — capture weights for alignment heads
            residual = hidden_states
            normed = layer.encoder_attn_layer_norm(hidden_states)

            cross_out, cross_weights = layer.encoder_attn(
                hidden_states=normed,
                key_value_states=encoder_hidden_states,
                attention_mask=None,
                past_key_values=None,
                output_attentions=True,
            )

            hidden_states = residual + cross_out

            # Feed-forward
            residual = hidden_states
            hidden_states = residual + layer.fc2(
                layer.activation_fn(layer.fc1(
                    layer.final_layer_norm(hidden_states)
                ))
            )

            # Capture selected alignment heads from this layer
            for selected_layer, selected_head in self.alignment_heads:
                if selected_layer == layer_idx:
                    # cross_weights: [B, H, T, S]
                    captured.append(cross_weights[:, selected_head, :, :])

        hidden_states = decoder.layer_norm(hidden_states)

        if not captured:
            raise RuntimeError("No alignment heads captured.")

        # [N, B, T, S] -> [B, N, T, S] -> average heads -> [B, T, S]
        stacked = torch.stack(captured, dim=0).permute(1, 0, 2, 3)
        alignment = stacked.mean(dim=1)

        return alignment


# ---------------------------------------------------------------------------
# ONNX I/O name generators
# ---------------------------------------------------------------------------

def output_names_for_init(num_layers: int) -> List[str]:
    names = ["logits"]
    for i in range(num_layers):
        names.extend([
            f"present.{i}.decoder.key",
            f"present.{i}.decoder.value",
            f"present.{i}.encoder.key",
            f"present.{i}.encoder.value",
        ])
    return names


def input_names_for_step(num_layers: int) -> List[str]:
    names = ["input_ids", "encoder_hidden_states", "cache_position"]
    for i in range(num_layers):
        names.extend([
            f"past_key_values.{i}.decoder.key",
            f"past_key_values.{i}.decoder.value",
            f"past_key_values.{i}.encoder.key",
            f"past_key_values.{i}.encoder.value",
        ])
    return names


def output_names_for_step(num_layers: int) -> List[str]:
    names = ["logits"]
    for i in range(num_layers):
        names.extend([
            f"present.{i}.decoder.key",
            f"present.{i}.decoder.value",
        ])
    return names


def build_init_dynamic_axes(num_layers: int) -> dict:
    axes = {
        "input_ids": {0: "batch", 1: "prompt_sequence"},
        "encoder_hidden_states": {0: "batch"},
        "logits": {0: "batch", 1: "prompt_sequence"},
    }

    for i in range(num_layers):
        axes[f"present.{i}.decoder.key"] = {0: "batch", 2: "prompt_sequence"}
        axes[f"present.{i}.decoder.value"] = {0: "batch", 2: "prompt_sequence"}
        axes[f"present.{i}.encoder.key"] = {0: "batch"}
        axes[f"present.{i}.encoder.value"] = {0: "batch"}

    return axes


def build_step_dynamic_axes(num_layers: int) -> dict:
    axes = {
        "input_ids": {0: "batch"},
        "encoder_hidden_states": {0: "batch"},
        "cache_position": {0: "cache_position_length"},
        "logits": {0: "batch"},
    }

    for i in range(num_layers):
        axes[f"past_key_values.{i}.decoder.key"] = {0: "batch", 2: "past_sequence"}
        axes[f"past_key_values.{i}.decoder.value"] = {0: "batch", 2: "past_sequence"}
        axes[f"past_key_values.{i}.encoder.key"] = {0: "batch"}
        axes[f"past_key_values.{i}.encoder.value"] = {0: "batch"}
        axes[f"present.{i}.decoder.key"] = {0: "batch", 2: "present_sequence"}
        axes[f"present.{i}.decoder.value"] = {0: "batch", 2: "present_sequence"}

    return axes


# ---------------------------------------------------------------------------
# Alignment heads
# ---------------------------------------------------------------------------

def get_required_alignment_heads(model: WhisperForConditionalGeneration) -> List[Tuple[int, int]]:
    """Production rule:
      1. First trust model.generation_config.alignment_heads.
      2. Then check model.config.alignment_heads.
      3. If absent, fail loudly unless the caller intentionally supplies a manual list.

    Do not silently use arbitrary heads for production.
    """
    candidates = [
        getattr(getattr(model, "generation_config", None), "alignment_heads", None),
        getattr(model.config, "alignment_heads", None),
    ]

    for heads in candidates:
        if heads:
            return [(int(layer), int(head)) for layer, head in heads]

    raise RuntimeError(
        "No alignment_heads found in model.generation_config or model.config. "
        "For original OpenAI/HF Whisper checkpoints, update/download generation_config.json. "
        "For custom checkpoints, provide a manual verified alignment_heads list; "
        "do not use a blind fallback."
    )


def parse_manual_alignment_heads(value: str | None) -> List[Tuple[int, int]] | None:
    if not value:
        return None
    # Format: "2:3,3:5,4:1"
    pairs: List[Tuple[int, int]] = []
    for item in value.split(","):
        layer, head = item.split(":")
        pairs.append((int(layer), int(head)))
    return pairs


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def convert_fp16(model_dir: Path, names: list[str]):
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


def convert_int8(model_dir: Path, names: list[str]):
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

def copy_tokenizer_files(model_id: str, output_dir: Path):
    for filename in ["tokenizer.json", "generation_config.json", "config.json", "preprocessor_config.json"]:
        try:
            local = hf_hub_download(model_id, filename)
            shutil.copy(local, output_dir / filename)
            print(f"  {filename}")
        except Exception as e:
            print(f"  SKIP {filename}: {e}")


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export_all(
    model_id: str,
    output_dir: str | Path,
    opset: int = 17,
    prompt_len: int = 4,
    past_len: int = 4,
    manual_alignment_heads: List[Tuple[int, int]] | None = None,
    fp16: bool = False,
    int8: bool = False,
    device: str | None = None,
    dtype: str = "float32",
):
    out_dir = ensure_dir(output_dir)

    # Use eager attention so output_attentions=True works for alignment export
    load_kwargs: Dict[str, Any] = dict(attn_implementation="eager")
    if dtype == "float16":
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    if device == "cpu":
        load_kwargs["device_map"] = "cpu"
    elif device == "cuda":
        load_kwargs["device_map"] = "cuda"
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        **load_kwargs,
    )
    model.eval()
    model.config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    cfg = model.config
    num_layers = cfg.decoder_layers
    num_heads = cfg.decoder_attention_heads
    head_dim = cfg.d_model // cfg.decoder_attention_heads
    num_mel_bins = cfg.num_mel_bins
    max_source_positions = cfg.max_source_positions
    max_target_positions = cfg.max_target_positions

    alignment_heads = manual_alignment_heads or get_required_alignment_heads(model)

    print(f"Exporting 4-graph Whisper: {model_id} -> {out_dir}")
    print(f"  layers={num_layers}  heads={num_heads}  head_dim={head_dim}")
    print(f"  mel_bins={num_mel_bins}  max_source={max_source_positions}  max_target={max_target_positions}")
    print(f"  alignment_heads={alignment_heads}")
    print()

    dummy_mel = torch.randn(1, num_mel_bins, 3000, dtype=torch.float32)
    dummy_hidden = torch.randn(1, max_source_positions, cfg.d_model, dtype=torch.float32)
    dummy_prompt_ids = torch.ones(1, prompt_len, dtype=torch.long)
    dummy_single_id = torch.ones(1, 1, dtype=torch.long)
    dummy_cache_position = torch.tensor([prompt_len], dtype=torch.long)

    # ---- 1. Encoder ----
    print("Exporting encoder_model.onnx ...")
    encoder = WhisperEncoderWrapper(model)
    torch.onnx.export(
        encoder,
        (dummy_mel,),
        str(out_dir / "encoder_model.onnx"),
        input_names=["input_features"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_features": {0: "batch"},
            "last_hidden_state": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = os.path.getsize(out_dir / "encoder_model.onnx") / 1024 / 1024
    print(f"  encoder_model.onnx  ({size_mb:.1f} MB)")

    # ---- 2. Decoder init ----
    print("Exporting decoder_init.onnx ...")
    init_wrapper = WhisperDecoderInitWrapper(model)
    init_output_names = output_names_for_init(num_layers)

    with torch.no_grad():
        _ = init_wrapper(dummy_prompt_ids, dummy_hidden)

    torch.onnx.export(
        init_wrapper,
        (dummy_prompt_ids, dummy_hidden),
        str(out_dir / "decoder_init.onnx"),
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=init_output_names,
        dynamic_axes=build_init_dynamic_axes(num_layers),
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = os.path.getsize(out_dir / "decoder_init.onnx") / 1024 / 1024
    print(f"  decoder_init.onnx  ({size_mb:.1f} MB)")

    # ---- 3. Decoder step ----
    print("Exporting decoder_step.onnx ...")
    flat_dummy_pkv: List[torch.Tensor] = []
    for _layer in range(num_layers):
        flat_dummy_pkv.append(torch.randn(1, num_heads, past_len, head_dim, dtype=torch.float32))
        flat_dummy_pkv.append(torch.randn(1, num_heads, past_len, head_dim, dtype=torch.float32))
        flat_dummy_pkv.append(torch.randn(1, num_heads, max_source_positions, head_dim, dtype=torch.float32))
        flat_dummy_pkv.append(torch.randn(1, num_heads, max_source_positions, head_dim, dtype=torch.float32))

    step_wrapper = WhisperDecoderStepWrapper(model)

    with torch.no_grad():
        _ = step_wrapper(dummy_single_id, dummy_hidden, dummy_cache_position, *flat_dummy_pkv)

    torch.onnx.export(
        step_wrapper,
        (dummy_single_id, dummy_hidden, dummy_cache_position, *flat_dummy_pkv),
        str(out_dir / "decoder_step.onnx"),
        input_names=input_names_for_step(num_layers),
        output_names=output_names_for_step(num_layers),
        dynamic_axes=build_step_dynamic_axes(num_layers),
        opset_version=opset,
        do_constant_folding=False,
        dynamo=False,
    )
    size_mb = os.path.getsize(out_dir / "decoder_step.onnx") / 1024 / 1024
    print(f"  decoder_step.onnx  ({size_mb:.1f} MB)")

    # ---- 4. Decoder align ----
    print("Exporting decoder_align.onnx ...")
    align_wrapper = WhisperDecoderAlignWrapper(model, alignment_heads=alignment_heads)
    dummy_align_ids = torch.ones(1, 16, dtype=torch.long)

    with torch.no_grad():
        _ = align_wrapper(dummy_align_ids, dummy_hidden)

    # Debug: verify no torch.diff in alignment wrapper
    import traceback as _tb
    _orig_diff = torch.diff
    _calls: list[str] = []
    def _traced_diff(*a, **kw):
        _calls.append("".join(_tb.format_stack(limit=12)))
        return _orig_diff(*a, **kw)
    torch.diff = _traced_diff
    with torch.no_grad():
        _ = align_wrapper(dummy_align_ids, dummy_hidden)
    torch.diff = _orig_diff
    if _calls:
        print(f"  WARNING: torch.diff called {len(_calls)} times during align forward:")
        for i, s in enumerate(_calls[:3]):
            print(f"  --- call {i} ---\n{s}")
    else:
        print(f"  ✓ No torch.diff calls in align wrapper forward pass")

    torch.onnx.export(
        align_wrapper,
        (dummy_align_ids, dummy_hidden),
        str(out_dir / "decoder_align.onnx"),
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["alignment"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "target_sequence"},
            "encoder_hidden_states": {0: "batch"},
            "alignment": {0: "batch", 1: "target_sequence"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = os.path.getsize(out_dir / "decoder_align.onnx") / 1024 / 1024
    print(f"  decoder_align.onnx  ({size_mb:.1f} MB)")
    align_exported = True

    # ---- Quantization variants ----
    all_names = [
        "encoder_model.onnx",
        "decoder_init.onnx",
        "decoder_step.onnx",
    ]
    if align_exported:
        all_names.append("decoder_align.onnx")

    if fp16:
        print("\nConverting to fp16:")
        fp16_names = [n.replace(".onnx", ".fp16.onnx") for n in all_names]
        convert_fp16(out_dir, fp16_names)

    if int8:
        print("\nQuantizing to int8:")
        int8_names = [n.replace(".onnx", "_int8.onnx") for n in all_names]
        convert_int8(out_dir, int8_names)

    # ---- Tokenizer / config files ----
    print("\nCopying config files:")
    copy_tokenizer_files(model_id, out_dir)

    # ---- Manifest ----
    print("\nGenerating manifest.json ...")

    vocab = tokenizer.get_vocab()

    def token_id(tok: str):
        return tokenizer.convert_tokens_to_ids(tok) if tok in vocab else None

    special_tokens = {
        "eos_token_id": cfg.eos_token_id,
        "bos_token_id": cfg.bos_token_id,
        "pad_token_id": cfg.pad_token_id,
        "decoder_start_token_id": cfg.decoder_start_token_id,
        "forced_decoder_ids": getattr(cfg, "forced_decoder_ids", None),
        "suppress_tokens": getattr(cfg, "suppress_tokens", None),
        "begin_suppress_tokens": getattr(cfg, "begin_suppress_tokens", None),
        "no_timestamps_token_id": token_id("<|notimestamps|>"),
        "no_speech_token_id": token_id("<|nospeech|>"),
        "timestamp_begin": token_id("<|0.00|>"),
    }

    artifacts: Dict[str, str] = {
        "encoder": "encoder_model.onnx",
        "decoder_init": "decoder_init.onnx",
        "decoder_step": "decoder_step.onnx",
    }
    if align_exported:
        artifacts["decoder_align"] = "decoder_align.onnx"

    manifest = {
        "model_id": model_id,
        "format": "whisper-browser-self-export-v1",
        "opset": opset,
        "num_mel_bins": num_mel_bins,
        "max_source_positions": max_source_positions,
        "max_target_positions": max_target_positions,
        "d_model": cfg.d_model,
        "decoder_layers": num_layers,
        "decoder_attention_heads": num_heads,
        "head_dim": head_dim,
        "vocab_size": cfg.vocab_size,
        "alignment_heads": alignment_heads,
        "alignment_heads_source": "manual" if manual_alignment_heads else "generation_config_or_config",
        "special_tokens": special_tokens,
        "artifacts": artifacts,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest.json")

    print(f"\nDone! All 4-graph artifacts in {out_dir}")
    print(f"  encoder_model.onnx  decoder_init.onnx  decoder_step.onnx  decoder_align.onnx")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export Whisper ONNX with 4-graph KV-cache decoder architecture"
    )
    parser.add_argument("model_id", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--prompt-len", type=int, default=4)
    parser.add_argument("--past-len", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to load model on: 'cpu' or 'cuda'. Default: auto.",
    )
    parser.add_argument(
        "--dtype", type=str, default="float32",
        help="Model dtype: 'float32' or 'float16'. Default: float32.",
    )
    parser.add_argument(
        "--alignment-heads",
        type=str,
        default=None,
        help="Manual verified heads as 'layer:head,layer:head'. Use only when official metadata is absent.",
    )
    args = parser.parse_args()

    export_all(
        model_id=args.model_id,
        output_dir=args.output_dir,
        opset=args.opset,
        prompt_len=args.prompt_len,
        past_len=args.past_len,
        manual_alignment_heads=parse_manual_alignment_heads(args.alignment_heads),
        fp16=args.fp16,
        int8=args.int8,
        device=args.device,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
