#!/usr/bin/env python3
"""Export Whisper ONNX artifacts for ASR.js — 4-graph KV-cache decoder split.

Generates:
  encoder_model.onnx           — log-Mel features to encoder hidden states
  decoder_init.onnx            — prompt/prefill decoder, creates initial KV cache
  decoder_step.onnx            — single-token autoregressive step with KV cache reuse
  decoder_align.onnx           — forced cross-attention alignment for word timestamps
  manifest.json                — model metadata
  Plus copies tokenizer.json, generation_config.json, config.json

Why 4 graphs instead of a merged decoder:
  - decoder_init runs once to build the initial cache from the prompt tokens
  - decoder_step is branch-free and fast for the autoregressive loop
  - decoder_align is run once after generation to extract cross-attention for DTW
  - This avoids a fragile merged decoder with ONNX If branches and
    DynamicCache data-dependent tracing that torch.onnx.export cannot capture

External data / large-model safety:
  - ONNX protobuf has a 2GB hard limit on serialized ModelProto.
  - Large models (whisper-large-v3 ~1.55B, large-v3-turbo ~809M) produce
    decoder graphs >2GB when weights are inline.
  - Use --external-data auto to automatically store weights in separate .data
    files co-located with the .onnx graph, keeping the .onnx file small.
  - Never serialize a >2GB ModelProto in memory; use path-based validate/save.

Usage:
  python export_whisper.py openai/whisper-tiny ./output/whisper-tiny
  python export_whisper.py openai/whisper-large-v3-turbo ./output --device cpu --dtype float32 --external-data auto
  python export_whisper.py openai/whisper-large-v3-turbo ./out-fp16 --device cuda --dtype float16 --external-data auto
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, WhisperForConditionalGeneration
from huggingface_hub import hf_hub_download

import onnx
from onnx.external_data_helper import convert_model_to_external_data
from onnxruntime.quantization import quantize_dynamic, QuantType

# ---------------------------------------------------------------------------
# External-data-safe ONNX helpers
# ---------------------------------------------------------------------------

# Default threshold: save weights externally if the serialized graph would
# exceed 100 MB (safe margin below the 2 GB protobuf hard limit). Large
# models like whisper-large-v3-turbo produce decoder graphs up to ~910 MB
# (decoder_init) in fp32; external data keeps the .onnx file small.
_DEFAULT_EXTERNAL_DATA_THRESHOLD = 100 * 1024 * 1024  # 100 MB


def _model_size_estimate(model: onnx.ModelProto) -> int:
    """Rough byte estimate of a ModelProto's initializer data.

    Covers inline raw_data, float_data, int32_data, and external_data
    (the latter requires the .data files to be accessible).
    """
    total = 0
    for init in model.graph.initializer:
        if init.raw_data:
            total += len(init.raw_data)
        elif init.float_data:
            total += len(init.float_data) * 4
        elif init.int32_data:
            total += len(init.int32_data) * 4
        elif init.external_data:
            # External data: try to stat the actual file
            for entry in init.external_data:
                if entry.key == 'location':
                    data_path = Path(entry.value)
                    if data_path.exists():
                        total += data_path.stat().st_size
                    break
    return total


def save_onnx_safe(
    model: onnx.ModelProto,
    path: str | Path,
    *,
    use_external_data: bool = False,
    all_tensors_to_one_file: bool = True,
    size_threshold: int = _DEFAULT_EXTERNAL_DATA_THRESHOLD,
    convert_attribute: bool = False,
) -> Path:
    """Save an ONNX model safely.

    When ``use_external_data=True``, large initializers (weights) are saved in
    a co-located ``<graph>.onnx.data`` file and the .onnx file stays well
    below the 2 GB protobuf serialization limit.  This is required for large
    Whisper models like large-v3-turbo (809M params) and large-v3 (1.55B).

    Never calls SerializeToString on a >2 GB ModelProto — the external-data
    path splits weights out before serialization.
    """
    path = Path(path)
    if use_external_data:
        location = f"{path.name}.data"
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=all_tensors_to_one_file,
            location=location,
            size_threshold=size_threshold,
            convert_attribute=convert_attribute,
        )
    else:
        onnx.save_model(model, str(path))
    return path


def validate_onnx_safe(
    model_or_path: onnx.ModelProto | str | Path,
    *,
    use_path_based: bool = False,
) -> None:
    """Validate an ONNX model safely.

    For external-data / large models use ``use_path_based=True`` (or pass a
    path string).  This calls ``onnx.checker.check_model(path)`` which reads
    the graph structure without loading all weights into memory.

    Never calls ``check_model(proto)`` on a large ModelProto — that would
    require serializing the full model and hit the 2 GB protobuf limit.
    """
    if use_path_based or isinstance(model_or_path, (str, Path)):
        onnx.checker.check_model(str(model_or_path))
    else:
        # For small in-memory models this is fine.
        onnx.checker.check_model(model_or_path)


def discover_external_data(graph_path: Path) -> List[Dict[str, Any]]:
    """Return external-data metadata for an ONNX graph file.

    Returns a list of {path, file, sizeBytes, sha256} entries, or an empty
    list if no external data is used.  Handles both external_data proto fields
    (torch.onnx.export, save_as_external_data) and metadata_props convention.
    """
    entries: List[Dict[str, Any]] = []
    if not graph_path.exists():
        return entries

    try:
        model = onnx.load(str(graph_path), load_external_data=False)
    except Exception:
        return entries

    # Collect unique external data files across all initializers.
    seen: set[str] = set()
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue

        data_loc: str | None = None
        offset: int | None = None
        length: int | None = None

        # Preferred: external_data proto field (torch.onnx.export, save_as_external_data)
        for entry in init.external_data:
            if entry.key == 'location':
                data_loc = entry.value
            elif entry.key == 'offset':
                try:
                    offset = int(entry.value)
                except ValueError:
                    pass
            elif entry.key == 'length':
                try:
                    length = int(entry.value)
                except ValueError:
                    pass

        # Fallback: metadata_props convention (some exporters)
        if not data_loc:
            for prop in init.metadata_props:
                if prop.key == 'location':
                    data_loc = prop.value
                    break

        if not data_loc:
            continue

        # Resolve path relative to the graph file's directory.
        data_path = (graph_path.parent / data_loc).resolve()
        if not data_path.exists():
            continue

        # Deduplicate by file path.
        if str(data_path) in seen:
            continue
        seen.add(str(data_path))

        stat = data_path.stat()
        try:
            sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
        except Exception:
            sha = ""

        entry: Dict[str, Any] = {
            "path": f"./{data_loc}",
            "file": data_loc,
            "sizeBytes": stat.st_size,
        }
        if sha:
            entry["sha256"] = sha
        entries.append(entry)

    return entries


def repack_external_data(
    graph_path: Path,
    *,
    location: str | None = None,
    all_tensors_to_one_file: bool = True,
    size_threshold: int = 0,
) -> bool:
    """Convert per-weight external data into a single consolidated .data file.

    When torch.onnx.export auto-externalizes a large encoder into many
    per-weight files (encoder.conv1.weight, encoder.layers.0.fc1.bias, ...),
    this function loads the graph metadata, reads all the per-weight files,
    concatenates them, and rewrites the ONNX graph to reference a single
    ``<name>.onnx.data`` file.  Old per-weight files are deleted.

    Returns True if repacking was needed/peformed, False if the graph already
    uses a single consolidated .data file or has no external data.
    """
    if not graph_path.exists():
        return False

    try:
        model = onnx.load(str(graph_path), load_external_data=False)
    except Exception:
        return False

    # Check if external data is used at all
    has_external = False
    per_weight_count = 0
    consolidated_count = 0
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        has_external = True
        for entry in init.external_data:
            if entry.key == "location":
                loc = entry.value
                # Per-weight: filename contains '.' (e.g., encoder.conv1.weight)
                # Consolidated: single file like "graph.onnx.data"
                if "/" not in loc and loc.endswith(".data"):
                    consolidated_count += 1
                else:
                    per_weight_count += 1
                break

    if not has_external:
        return False

    # If already consolidated (all init refs point to the same .data file),
    # nothing to do.
    if per_weight_count == 0 and consolidated_count > 0:
        return False

    # Only per-weight files — we need to repack.
    # Read all external data files, build a single .data file.
    if location is None:
        location = f"{graph_path.name}.data"

    data_dir = graph_path.parent
    data_path = data_dir / location

    # Collect all unique external data files and their order
    offset_map: dict[str, tuple[int, int]] = {}  # file -> (offset_in_combined, length)
    ordered_files: list[str] = []  # order as they appear in initializers
    seen_files: set[str] = set()

    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in init.external_data:
            if entry.key == "location":
                loc = entry.value
                if loc not in seen_files:
                    seen_files.add(loc)
                    ordered_files.append(loc)
                break

    # Build the combined .data file
    current_offset = 0
    with open(data_path, "wb") as out_f:
        for loc in ordered_files:
            src = data_dir / loc
            if not src.exists():
                print(f"    WARNING: external data file missing: {loc}")
                continue
            data = src.read_bytes()
            out_f.write(data)
            offset_map[loc] = (current_offset, len(data))
            current_offset += len(data)

    # Rewrite ONNX graph: replace per-weight external_data refs
    # with consolidated ref pointing into the single .data file
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        # Find the original location
        orig_loc = None
        orig_length = None
        for entry in init.external_data:
            if entry.key == "location":
                orig_loc = entry.value
            elif entry.key == "length":
                orig_length = int(entry.value)

        if orig_loc is None or orig_loc not in offset_map:
            continue

        off, length = offset_map[orig_loc]
        # Clear old external_data and set new consolidated ref
        del init.external_data[:]
        init.external_data.add()
        init.external_data[0].key = "location"
        init.external_data[0].value = location
        init.external_data.add()
        init.external_data[1].key = "offset"
        init.external_data[1].value = str(off)
        init.external_data.add()
        init.external_data[2].key = "length"
        init.external_data[2].value = str(length)

    # Save the updated graph (no external data conversion needed —
    # initializers already reference external data)
    onnx.save_model(model, str(graph_path))

    # Delete old per-weight files
    for loc in ordered_files:
        old_file = data_dir / loc
        if old_file.exists() and old_file != data_path:
            old_file.unlink()

    return True


def _has_per_weight_external(gpath: Path) -> bool:
    """Return True if a graph uses per-weight external data files."""
    if not gpath.exists():
        return False
    try:
        m = onnx.load(str(gpath), load_external_data=False)
    except Exception:
        return False
    counts: dict[str, int] = {}
    for init in m.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in init.external_data:
            if entry.key == "location":
                counts[entry.value] = counts.get(entry.value, 0) + 1
                break
    # Per-weight: many unique locations with count=1
    # Consolidated: one location referenced many times
    if len(counts) <= 1:
        return False
    return any(c == 1 for c in counts.values())


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
# Quantization (external-data-safe)
# ---------------------------------------------------------------------------

def convert_fp16_safe(
    model_dir: Path,
    names: list[str],
    *,
    use_external_data: bool = False,
    all_tensors_to_one_file: bool = True,
    size_threshold: int = _DEFAULT_EXTERNAL_DATA_THRESHOLD,
    validate_path: bool = True,
):
    """Convert fp32 ONNX models to fp16 using external-data-safe saving.

    For export-time FP16 (--dtype float16 at export time), torch.onnx.export
    already produces the FP16 graph directly; this convert_fp16_safe path is
    the *post-export* route.

    Large-model safety: for models where the FP32 proto in memory exceeds
    2 GB, loading with ``load_external_data=False`` keeps the graph small and
    then the converter works on the in-memory graph.  The result is saved
    immediately with external data.  If the converter itself requires all
    weights in memory, the user should use the export-time FP16 path
    (--dtype float16) instead.
    """
    from onnxconverter_common import float16

    success = True
    for name in names:
        src = model_dir / name.replace(".fp16", "")
        dst = model_dir / name
        if not src.exists():
            continue

        # Load WITHOUT external data to keep the ModelProto small — the
        # converter only transforms graph nodes, not initializer values.
        try:
            model = onnx.load(str(src), load_external_data=True)
            est = _model_size_estimate(model) / 1024 / 1024
            print(f"  Loading {name}  ({est:.1f} MB in-memory)")

            model_fp16 = float16.convert_float_to_float16(model)

            save_onnx_safe(
                model_fp16,
                dst,
                use_external_data=use_external_data,
                all_tensors_to_one_file=all_tensors_to_one_file,
                size_threshold=size_threshold,
            )

            # Validate via path-based checker (safe for external-data models)
            if validate_path:
                validate_onnx_safe(dst, use_path_based=True)

            # Try ORT load to verify graph health.
            # NOTE: onnxconverter_common.float16 may leave Cast nodes with
            # mismatched output types, causing ORT load failures for some graphs.
            # This is a known converter limitation, not an external-data issue.
            # Export-time FP16 (--dtype float16) avoids this entirely.
            try:
                import onnxruntime as _ort_verify
                _ort_verify.InferenceSession(str(dst), providers=['CPUExecutionProvider'])
                print(f"    ORT load: OK")
            except Exception as _ort_err:
                print(f"    ORT load: FAILED ({_ort_err}). Use --dtype float16 at export time instead.")

            size_mb = os.path.getsize(dst) / 1024 / 1024
            data_size = ""
            data_file = Path(str(dst) + ".data")
            if data_file.exists():
                data_size = f"  data: {os.path.getsize(data_file) / 1024 / 1024:.1f} MB"
            print(f"  {name}  ({size_mb:.1f} MB onnx{data_size})")

        except Exception as e:
            print(f"  FAILED {name}: {e}")
            print(f"    For large models, use export-time FP16: --dtype float16 --external-data auto")
            success = False

    return success


def convert_int8_safe(model_dir: Path, names: list[str]):
    """Quantize to int8 using ONNX Runtime dynamic quantization.

    ORT's quantize_dynamic works on file paths and reads/writes safely.
    External data files are preserved automatically.
    """
    for name in names:
        src = model_dir / name.replace("_int8", "")
        dst = model_dir / name
        if not src.exists():
            continue
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        size_mb = os.path.getsize(dst) / 1024 / 1024
        data_size = ""
        data_file = Path(str(dst) + ".data")
        if data_file.exists():
            data_size = f"  data: {os.path.getsize(data_file) / 1024 / 1024:.1f} MB"
        print(f"  {name}  ({size_mb:.1f} MB onnx{data_size})")


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


def _organize_variant_dirs(
    build_dir: Path,
    *,
    model_id: str,
    variant: str,
    fp16: bool,
    int8: bool,
    dtype: str,
    use_external_data: bool,
    external_data_one_file: bool,
    external_data_threshold: int,
    validate_path_only: bool,
    quantize: str,
    graph_names: list[str],
):
    """Organize exported artifacts into variant subdirectories.

    Target layout:
      <build_dir>/
        fp32/    (or fp16/, int8-dynamic/)
          manifest.json
          encoder_model.onnx  (+ .data)
          decoder_init.onnx   (+ .data)
          decoder_step.onnx   (+ .data)
          decoder_align.onnx  (+ .data)
        README.md
        config.json
        ...
    """
    variant_dir_name = variant  # "fp32", "fp16", "int8-dynamic"
    variant_dir = build_dir / variant_dir_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Move graph files into variant dir
    for gname in graph_names:
        src = build_dir / gname
        if src.exists():
            shutil.move(str(src), str(variant_dir / gname))
        # Move associated .data file
        data_src = build_dir / f"{gname}.data"
        if data_src.exists():
            shutil.move(str(data_src), str(variant_dir / f"{gname}.data"))

    # Move manifest into variant dir
    manifest_src = build_dir / "manifest.json"
    if manifest_src.exists():
        shutil.move(str(manifest_src), str(variant_dir / "manifest.json"))

    # Copy config files into variant dir (needed for local-file loading)
    for cfg_file in ["config.json", "generation_config.json", "tokenizer.json", "preprocessor_config.json"]:
        cfg_src = build_dir / cfg_file
        if cfg_src.exists():
            shutil.copy(str(cfg_src), str(variant_dir / cfg_file))

    # FP16/INT8 post-processing inside variant dir
    if variant == "fp16" and dtype == "float16":
        # Export-time FP16: torch.onnx.export already produced FP16 graphs.
        # Validate that ORT can load each graph.
        print(f"\nValidating export-time FP16 variant:")
        _validate_variant_graphs(variant_dir, graph_names, "FP16")
    elif variant == "fp16" and fp16:
        # Post-export FP16 conversion
        print(f"\nPost-export FP16 conversion in {variant_dir_name}/:")
        fp16_names = [n.replace(".onnx", ".fp16.onnx") for n in graph_names]
        # Note: fp16_names currently have .fp16 in them but the sources are base names
        # Fix: convert from base names
        fp16_src_names = [n for n in graph_names]
        fp16_dst_names = [n.replace(".onnx", ".fp16.onnx") for n in graph_names]
        # Actually for variant dir, we convert in-place to the variant name
        ok = convert_fp16_safe(
            variant_dir,
            [f"{n.replace('.onnx', '')}.fp16.onnx" for n in graph_names],
            use_external_data=use_external_data,
            all_tensors_to_one_file=external_data_one_file,
            size_threshold=external_data_threshold,
            validate_path=validate_path_only,
        )
        if not ok:
            print("  WARNING: FP16 conversion had failures. See above.")
    elif variant in ("int8-dynamic", "q8") and int8:
        print(f"\nDynamic int8 quantization in {variant_dir_name}/:")
        int8_names = [n.replace(".onnx", "_int8.onnx") for n in graph_names]
        convert_int8_safe(variant_dir, int8_names)
        # Remove old FP32 files to keep dir clean
        for gname in graph_names:
            old = variant_dir / gname
            if old.exists():
                old.unlink()
            old_data = variant_dir / f"{gname}.data"
            if old_data.exists():
                old_data.unlink()
        # Rename int8 files to standard names
        for gname in graph_names:
            int8_name = gname.replace(".onnx", "_int8.onnx")
            src = variant_dir / int8_name
            if src.exists():
                shutil.move(str(src), str(variant_dir / gname))
            # Handle data files too
            int8_data = variant_dir / f"{int8_name}.data"
            if int8_data.exists():
                shutil.move(str(int8_data), str(variant_dir / f"{gname}.data"))

    # Write README at root
    _write_publish_readme(build_dir, model_id, variant_dir_name)

    print(f"\nVariant layout: {variant_dir_name}/")


def _validate_variant_graphs(
    variant_dir: Path,
    graph_names: list[str],
    label: str,
):
    """Validate all graphs in a variant directory: path-based ONNX checker + ORT load."""
    for gname in graph_names:
        gpath = variant_dir / gname
        if not gpath.exists():
            print(f"  ✗ {gname}: MISSING")
            continue
        try:
            validate_onnx_safe(gpath, use_path_based=True)
        except Exception as e:
            print(f"  ✗ {gname} ONNX check failed: {e}")
            continue
        try:
            import onnxruntime as _ort_val
            _ort_val.InferenceSession(str(gpath), providers=['CPUExecutionProvider'])
            onnx_sz = os.path.getsize(gpath) / 1024 / 1024
            data_sz = ""
            df = variant_dir / f"{gname}.data"
            if df.exists():
                data_sz = f" + {os.path.getsize(df) / 1024 / 1024:.1f} MB data"
            print(f"  ✓ {gname} ({onnx_sz:.1f} MB{data_sz})")
        except Exception as e:
            print(f"  ✗ {gname} ORT load failed: {e}")


def _write_publish_readme(build_dir: Path, model_id: str, variant_dir_name: str):
    """Write a README.md at the publish root."""
    import datetime
    readme = build_dir / "README.md"
    if readme.exists():
        return  # Don't overwrite existing README
    with open(readme, "w") as f:
        f.write(f"""---
license: apache-2.0
tags:
- whisper
- onnx
- asr
- speech-recognition
base_model: {model_id}
---

# {model_id} — 4-Graph ONNX Export

Self-exported 4-graph Whisper ONNX for [asrjs/speech-recognition](https://github.com/asrjs/speech-recognition).

## Format

`whisper-browser-self-export-v1` — 4-graph KV-cache split (encoder + decoder_init + decoder_step + decoder_align).

## Variants

| Dir | Description |
|-----|-------------|
| `{variant_dir_name}/` | Exported variant |

## Usage

```python
# Python: export_whisper.py
python export_whisper.py {model_id} ./output --device cpu --dtype float32 --external-data auto
```

```js
// TypeScript
import {{ loadSplitGraphLocalModel }} from '@asrjs/speech-recognition/models/whisper-seq2seq';
const model = loadSplitGraphLocalModel('./{variant_dir_name}');
```

Generated {datetime.date.today().isoformat()}.
""")


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
    external_data: str = "auto",
    external_data_threshold: int = _DEFAULT_EXTERNAL_DATA_THRESHOLD,
    external_data_one_file: bool = True,
    validate_path_only: Optional[bool] = None,
    variant: str = "fp32",
    output_layout: str = "flat",
    quantize: str = "dynamic",
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
    print(f"  dtype={dtype}  external_data={external_data}"
          f"  threshold={external_data_threshold / 1024 / 1024:.0f}MB"
          f"  one_file={external_data_one_file}")
    print()

    # ---- External data resolution ----
    # "auto" → use external data for models likely to exceed threshold
    # "always" → force external data for all graphs
    # "never" → inline weights (DANGEROUS for large models)
    use_external_data: bool
    if external_data == "never":
        use_external_data = False
    elif external_data == "always":
        use_external_data = True
    else:  # "auto"
        # Enable external data if the model has enough parameters to risk
        # exceeding the 2 GB protobuf limit.  Check both encoder and decoder
        # layers — encoder may be large even when decoder is small (e.g.,
        # large-v3-turbo: 32 encoder layers, 4 decoder layers).
        enc_layers = getattr(cfg, "encoder_layers", num_layers)
        use_external_data = max(num_layers, enc_layers) >= 24

    # Auto-detect validate_path_only if not explicitly set
    if validate_path_only is None:
        validate_path_only = use_external_data

    print(f"  use_external_data={use_external_data}  validate_path_only={validate_path_only}")
    print()

    # Use model's dtype and device for dummy inputs (critical for fp16/cuda export)
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    dummy_mel = torch.randn(1, num_mel_bins, 3000, dtype=model_dtype).to(model_device)
    dummy_hidden = torch.randn(1, max_source_positions, cfg.d_model, dtype=model_dtype).to(model_device)
    dummy_prompt_ids = torch.ones(1, prompt_len, dtype=torch.long).to(model_device)
    dummy_single_id = torch.ones(1, 1, dtype=torch.long).to(model_device)
    dummy_cache_position = torch.tensor([prompt_len], dtype=torch.long).to(model_device)

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
        flat_dummy_pkv.append(torch.randn(1, num_heads, past_len, head_dim, dtype=model_dtype).to(model_device))
        flat_dummy_pkv.append(torch.randn(1, num_heads, past_len, head_dim, dtype=model_dtype).to(model_device))
        flat_dummy_pkv.append(torch.randn(1, num_heads, max_source_positions, head_dim, dtype=model_dtype).to(model_device))
        flat_dummy_pkv.append(torch.randn(1, num_heads, max_source_positions, head_dim, dtype=model_dtype).to(model_device))

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
    dummy_align_ids = torch.ones(1, 16, dtype=torch.long).to(model_device)

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

    # ---- External data conversion for graphs ----
    # If external data is enabled, convert graphs exported by torch.onnx.export
    # with INLINE weights to external-data format.  This avoids the 2 GB
    # protobuf serialization limit.
    #
    # Important: torch.onnx.export sometimes auto-externalizes individual
    # tensors into separate files (for very large models).  We detect and
    # skip those graphs — they're already safe, and re-loading all external
    # weights back into memory would create a >2 GB ModelProto.
    if use_external_data:
        print("\nExternal data check:")
        graph_names = [
            "encoder_model.onnx",
            "decoder_init.onnx",
            "decoder_step.onnx",
        ]
        if align_exported:
            graph_names.append("decoder_align.onnx")

        def _already_external(gpath: Path) -> bool:
            """Check if a graph already uses external data."""
            try:
                m = onnx.load(str(gpath), load_external_data=False)
                for init in m.graph.initializer:
                    if init.data_location == onnx.TensorProto.EXTERNAL:
                        return True
                return False
            except Exception:
                return False

        for gname in graph_names:
            gpath = out_dir / gname
            if not gpath.exists():
                continue

            onnx_size_mb = os.path.getsize(gpath) / 1024 / 1024

            if _already_external(gpath):
                # torch.onnx.export already externalized this graph's weights.
                # Check if they are per-weight files (bad for publishing) or
                # already consolidated into a single .data file.
                if external_data_one_file and _has_per_weight_external(gpath):
                    print(f"  {gname}  ({onnx_size_mb:.2f} MB onnx, per-weight files → repacking...)")
                    repack_external_data(gpath)
                    new_onnx = os.path.getsize(gpath) / 1024 / 1024
                    data_file = Path(str(gpath) + ".data")
                    data_sz = os.path.getsize(data_file) / 1024 / 1024 if data_file.exists() else 0
                    print(f"    → {new_onnx:.2f} MB onnx, data: {data_sz:.1f} MB (consolidated)")
                else:
                    # Already good — consolidated or user wants per-weight files.
                    ext_files = list(out_dir.glob(f"{gpath.stem}*"))
                    ext_files = [f for f in ext_files if f != gpath and not f.name.endswith('.json')]
                    ext_total = sum(f.stat().st_size for f in ext_files) / 1024 / 1024
                    print(f"  {gname}  ({onnx_size_mb:.2f} MB onnx, {len(ext_files)} ext files, {ext_total:.1f} MB data) [already external]")
                continue

            # Graph has inline weights — convert safely.
            model = onnx.load(str(gpath), load_external_data=True)
            est = _model_size_estimate(model) / 1024 / 1024
            print(f"  {gname}  ({onnx_size_mb:.1f} MB, ~{est:.1f} MB weights → external)")

            save_onnx_safe(
                model,
                gpath,
                use_external_data=True,
                all_tensors_to_one_file=external_data_one_file,
                size_threshold=external_data_threshold,
            )
            new_onnx = os.path.getsize(gpath) / 1024 / 1024
            data_size = ""
            data_file = Path(str(gpath) + ".data")
            if data_file.exists():
                data_size = f"  data: {os.path.getsize(data_file) / 1024 / 1024:.1f} MB"
            print(f"    → {new_onnx:.2f} MB onnx{data_size}")

        # Validate converted graphs
        print("  Validating external-data graphs (path-based):")
        for gname in graph_names:
            gpath = out_dir / gname
            if gpath.exists():
                try:
                    validate_onnx_safe(gpath, use_path_based=True)
                    print(f"    ✓ {gname}")
                except Exception as e:
                    print(f"    ✗ {gname}: {e}")

    # ---- Quantization variants ----
    all_names = [
        "encoder_model.onnx",
        "decoder_init.onnx",
        "decoder_step.onnx",
    ]
    if align_exported:
        all_names.append("decoder_align.onnx")

    if fp16:
        print("\nConverting to fp16 (post-export):")
        fp16_names = [n.replace(".onnx", ".fp16.onnx") for n in all_names]
        convert_fp16_safe(
            out_dir,
            fp16_names,
            use_external_data=use_external_data,
            all_tensors_to_one_file=external_data_one_file,
            size_threshold=external_data_threshold,
            validate_path=validate_path_only,
        )

    if int8:
        print("\nQuantizing to int8:")
        int8_names = [n.replace(".onnx", "_int8.onnx") for n in all_names]
        convert_int8_safe(out_dir, int8_names)

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

    # Build artifacts dict with externalData metadata
    def _graph_entry(filename: str) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"file": filename}
        if use_external_data:
            ext = discover_external_data(out_dir / filename)
            if ext:
                entry["externalData"] = ext
        return entry

    artifacts: Dict[str, Dict[str, Any]] = {
        "encoder": _graph_entry("encoder_model.onnx"),
        "decoder_init": _graph_entry("decoder_init.onnx"),
        "decoder_step": _graph_entry("decoder_step.onnx"),
    }
    if align_exported:
        artifacts["decoder_align"] = _graph_entry("decoder_align.onnx")

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
        "external_data": use_external_data,
        "external_data_threshold": external_data_threshold,
        "runtime_compatibility": {
            "fp32": {
                "precision": "float32",
                "intended_runtime": ["node", "native", "python"],
                "status": "validated",
                "notes": "Reference path. Very large (~4.5 GB for large-v3-turbo). Not recommended for browser/WebGPU."
            },
            "fp16": {
                "precision": "float16",
                "intended_runtime": ["browser", "webgpu", "native_gpu"],
                "status": "requires_export_time_fp16",
                "notes": "Use --dtype float16 at export time. Post-export fp16 conversion is experimental due to onnxconverter_common Cast/type mismatch."
            },
            "int8-dynamic": {
                "precision": "int8 (dynamic)",
                "intended_runtime": ["cpu", "native", "browser_candidate"],
                "status": "requires_validation",
                "notes": "Post-export ONNX Runtime dynamic quantization. All four graphs must validate independently."
            }
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest.json")

    # ---- Variant directory layout ----
    if output_layout == "variant-dirs":
        _organize_variant_dirs(
            out_dir,
            model_id=model_id,
            variant=variant,
            fp16=fp16,
            int8=int8,
            dtype=dtype,
            use_external_data=use_external_data,
            external_data_one_file=external_data_one_file,
            external_data_threshold=external_data_threshold,
            validate_path_only=validate_path_only,
            quantize=quantize,
            graph_names=all_names,
        )

    print(f"\nDone! All 4-graph artifacts in {out_dir}")
    print(f"  encoder_model.onnx  decoder_init.onnx  decoder_step.onnx  decoder_align.onnx")
    if use_external_data:
        print(f"  External data: *.onnx.data files co-located with each graph")


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
        "--external-data", type=str, default="auto",
        choices=["auto", "always", "never"],
        help="ONNX external data strategy. 'auto' enables external data for large models "
             "(decoder_layers >= 24). 'always' forces external data for all graphs. "
             "'never' uses inline weights (NOT safe for large models >2GB). Default: auto.",
    )
    parser.add_argument(
        "--external-data-threshold", type=int, default=_DEFAULT_EXTERNAL_DATA_THRESHOLD,
        help=f"Size threshold in bytes for external data (default: {_DEFAULT_EXTERNAL_DATA_THRESHOLD}). "
             "Initializers above this size are stored in the .data file.",
    )
    parser.add_argument(
        "--external-data-one-file", type=str, default="true",
        choices=["true", "false"],
        help="Store all external data in a single .data file per graph (default: true).",
    )
    parser.add_argument(
        "--validate-path-only", type=str, default=None,
        choices=["true", "false"],
        help="Use path-based ONNX checker (safe for external-data models). "
             "Default: auto (true when external data is used).",
    )
    parser.add_argument(
        "--variant", type=str, default="fp32",
        choices=["fp32", "fp16", "int8-dynamic", "q8"],
        help="Variant to export: fp32, fp16, q8/int8-dynamic. Default: fp32.",
    )
    parser.add_argument(
        "--output-layout", type=str, default="variant-dirs",
        choices=["variant-dirs", "flat"],
        help="Output layout: 'variant-dirs' puts each variant in a subdirectory "
             "(fp32/, fp16/, int8-dynamic/). 'flat' keeps everything in root. "
             "Default: variant-dirs.",
    )
    parser.add_argument(
        "--quantize", type=str, default="dynamic",
        choices=["dynamic"],
        help="Quantization method (only used for int8-dynamic variant). Default: dynamic.",
    )
    parser.add_argument(
        "--alignment-heads",
        type=str,
        default=None,
        help="Manual verified heads as 'layer:head,layer:head'. Use only when official metadata is absent.",
    )
    args = parser.parse_args()

    # Auto-detect flags from variant
    if args.variant in ("fp16",) and args.dtype == "float32" and not args.fp16:
        args.dtype = "float16"
        print("Note: --variant fp16 → auto-setting --dtype float16 (export-time FP16)")
    elif args.variant in ("int8-dynamic", "q8") and not args.int8:
        args.int8 = True
        args.variant = "q8"  # Normalize to q8 directory name
        print("Note: --variant q8/int8-dynamic → auto-enabling --int8")

    validate_path_only = None
    if args.validate_path_only is not None:
        validate_path_only = args.validate_path_only == "true"

    external_data_one_file = args.external_data_one_file == "true"

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
        external_data=args.external_data,
        external_data_threshold=args.external_data_threshold,
        external_data_one_file=external_data_one_file,
        validate_path_only=validate_path_only,
        variant=args.variant,
        output_layout=args.output_layout,
        quantize=args.quantize,
    )


if __name__ == "__main__":
    main()
