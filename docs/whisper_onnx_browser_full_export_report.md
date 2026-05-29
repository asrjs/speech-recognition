# Self-Owned Whisper ONNX Export + Browser Runtime

**Target:** Build a complete Whisper ONNX export and TypeScript browser runtime with KV-cache, segment timestamps, word timestamps, and confidence metrics, without depending on third-party exported ONNX artifacts such as Optimum-generated model files.

This document keeps the deeper architecture from the prior report, but corrects the broken snippets and clarifies why the complexity is needed. The reason is **not** long-duration audio cache. Whisper is still bounded around 30-second chunks and a decoder context of 448 tokens. The deeper split is needed because **fast generation, KV-cache, segment timestamps, word-level alignment, and confidence metrics need different graph/runtime paths**.

## Source context

Whisper's official README says `transcribe()` processes audio using a sliding 30-second window and performs autoregressive sequence-to-sequence predictions on each window. It also shows the lower-level path where audio is padded or trimmed to fit 30 seconds before log-Mel extraction and decoding.[^openai-readme]

Hugging Face's Whisper docs list `max_source_positions=1500`, `max_target_positions=448`, `use_cache=True`, `median_filter_width=7`, `chunk_length=30`, and timestamp-related configuration fields.[^hf-whisper-docs]

Hugging Face's cache documentation explains that KV cache stores key/value states during autoregressive generation to avoid recomputing previous token states; it also notes that `DynamicCache` is the default cache and grows dynamically during generation.[^hf-kv-cache]

Hugging Face's Whisper generation implementation normalizes and median-filters selected cross-attention weights, averages heads, and applies DTW to compute token timestamps; it also describes `time_precision=0.02` and `time_precision_features=0.01`.[^hf-generation-whisper]

Hugging Face's OpenAI-to-HF conversion script explicitly warns that alignment heads are available only for the original OpenAI checkpoints when converting custom Whisper variants; this is why alignment-head handling must be treated as a first-class export/manifest concern.[^hf-convert-alignment]

---

## Architecture summary

Export four graph files:

```text
encoder_model.onnx

decoder_init.onnx

decoder_step.onnx

decoder_align.onnx
```

### Why four graphs?

| Graph | Purpose | Runs | Main reason to keep separate |
|---|---|---:|---|
| `encoder_model.onnx` | Log-Mel features to encoder hidden states | Once per chunk | Clean fixed encoder path |
| `decoder_init.onnx` | Prompt/prefill decoder, creates initial KV cache | Once per chunk | Avoid empty-cache branch in step graph |
| `decoder_step.onnx` | Single-token autoregressive loop with KV cache | Many times | Fast branch-free token generation |
| `decoder_align.onnx` | Teacher-forced cross-attention extraction | Once after generation | Word timestamps need attention maps, generation loop should not return huge attentions |

This split avoids a fragile merged decoder with ONNX `If` branches. It also avoids carrying large attention outputs through every autoregressive step.

---

## Important constraints

### Do not implement long-duration KV cache

Reset decoder KV cache for every Whisper chunk. Long audio should be handled by chunking/VAD/stitching:

```text
audio stream
  -> VAD / chunking / 30-second windows
  -> log-Mel
  -> encoder
  -> decoder init + step loop
  -> segment timestamps
  -> decoder align + word timestamps
  -> stitching
```

Previous transcript text may be injected as prompt tokens, but decoder KV cache should **not** be carried across unrelated audio windows.

### KV cache is small for Whisper

For `whisper-base`:

```text
decoder_layers = 6
decoder_attention_heads = 8
head_dim = 64
max_target_positions = 448
dtype = FP32
```

One self-attention K tensor:

```text
1 * 8 * 448 * 64 * 4 bytes = 917,504 bytes ≈ 896 KiB
```

K + V for all 6 layers:

```text
917,504 * 2 * 6 = 11,010,048 bytes ≈ 10.5 MiB
```

FP16 is roughly half. So the memory issue is not the active Whisper KV cache; the real issue is clean export and runtime orchestration.

---

## Correct graph interfaces

### `encoder_model.onnx`

Inputs:

```text
input_features: [B, num_mel_bins, 3000]
```

Outputs:

```text
last_hidden_state: [B, 1500, d_model]
```

### `decoder_init.onnx`

Inputs:

```text
input_ids: [B, P]
encoder_hidden_states: [B, 1500, d_model]
```

Outputs:

```text
logits: [B, P, vocab_size]

present.{layer}.decoder.key:   [B, decoder_heads, P, head_dim]
present.{layer}.decoder.value: [B, decoder_heads, P, head_dim]
present.{layer}.encoder.key:   [B, decoder_heads, 1500, head_dim]
present.{layer}.encoder.value: [B, decoder_heads, 1500, head_dim]
```

### `decoder_step.onnx`

Inputs:

```text
input_ids: [B, 1]
encoder_hidden_states: [B, 1500, d_model]
cache_position: [1]

past_key_values.{layer}.decoder.key:   [B, decoder_heads, L, head_dim]
past_key_values.{layer}.decoder.value: [B, decoder_heads, L, head_dim]
past_key_values.{layer}.encoder.key:   [B, decoder_heads, 1500, head_dim]
past_key_values.{layer}.encoder.value: [B, decoder_heads, 1500, head_dim]
```

Outputs:

```text
logits: [B, 1, vocab_size]

present.{layer}.decoder.key:   [B, decoder_heads, L + 1, head_dim]
present.{layer}.decoder.value: [B, decoder_heads, L + 1, head_dim]
```

Do **not** output cross-attention KV from every step. Cross-attention K/V is static for the same encoder hidden states, so keep it from `decoder_init.onnx` in the runtime cache dictionary.

### `decoder_align.onnx`

Inputs:

```text
input_ids: [B, T]
encoder_hidden_states: [B, 1500, d_model]
```

Outputs:

```text
alignment: [B, T, 1500]
```

This graph runs teacher-forced over the final generated sequence after token generation is complete.

---

## Critical correction from earlier code

This is wrong:

```python
flat_outputs[f"present.{i}.decoder.key"] = pkv[i]
```

`pkv[i]` is normally a full layer cache tuple. Use explicit tuple indexing:

```python
flat_outputs[f"present.{i}.decoder.key"] = pkv[i][0]
flat_outputs[f"present.{i}.decoder.value"] = pkv[i][1]
flat_outputs[f"present.{i}.encoder.key"] = pkv[i][2]
flat_outputs[f"present.{i}.encoder.value"] = pkv[i][3]
```

Also fix all incomplete placeholders like:

```python
past_key_values =
flat_pkv_inputs =
generatedTokens =
BigInt64Array.from()
```

---

# Python exporter

Create:

```text
tools/whisper-onnx-export/
  export_whisper.py
  requirements.txt
```

Suggested `requirements.txt`:

```text
torch
transformers
onnx
onnxruntime
numpy
```

Pin exact versions after the first successful export. Do not allow arbitrary Transformers upgrades to silently change cache behavior.

## `export_whisper.py`

```python
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, WhisperForConditionalGeneration

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
    """
    Convert HF cache outputs into legacy tuple format:
      tuple[layer] = (self_k, self_v, cross_k, cross_v)

    Supports:
      - legacy tuple/list cache
      - modern Cache objects exposing to_legacy_cache()
    """
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()

    if not isinstance(past_key_values, (tuple, list)):
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


def make_legacy_cache_from_flat(flat: Sequence[torch.Tensor], num_layers: int) -> LegacyCache:
    expected = num_layers * 4
    if len(flat) != expected:
        raise ValueError(f"Expected {expected} flat cache tensors, got {len(flat)}")

    layers: List[LegacyLayerCache] = []
    for i in range(num_layers):
        offset = i * 4
        self_k = flat[offset + 0]
        self_v = flat[offset + 1]
        cross_k = flat[offset + 2]
        cross_v = flat[offset + 3]
        layers.append((self_k, self_v, cross_k, cross_v))

    return tuple(layers)


def flatten_init_cache_outputs(pkv: LegacyCache) -> List[torch.Tensor]:
    flat: List[torch.Tensor] = []
    for self_k, self_v, cross_k, cross_v in pkv:
        flat.extend([self_k, self_v, cross_k, cross_v])
    return flat


def flatten_step_cache_outputs(pkv: LegacyCache) -> List[torch.Tensor]:
    flat: List[torch.Tensor] = []
    for self_k, self_v, _cross_k, _cross_v in pkv:
        flat.extend([self_k, self_v])
    return flat


class WhisperEncoderWrapper(nn.Module):
    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_features=input_features, return_dict=True)
        return out.last_hidden_state


class WhisperDecoderInitWrapper(nn.Module):
    """
    Prefill decoder graph.

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
    """
    Single-token autoregressive decoder graph.

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
        pkv = make_legacy_cache_from_flat(flat_past_key_values, self.num_layers)

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
    """
    Teacher-forced decoder pass used only for word-level timestamps.

    Returns selected cross-attention heads as:
      alignment_heads: [B, selected_heads, target_tokens, source_frames]

    The TypeScript side can normalize, median-filter, average heads, and run DTW.
    Returning selected heads instead of an already-averaged map makes debugging easier.
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        alignment_heads: Sequence[Tuple[int, int]],
    ):
        super().__init__()
        self.decoder = model.model.decoder
        self.alignment_heads = list(alignment_heads)

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
        outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

        cross_attentions = outputs.cross_attentions
        if cross_attentions is None:
            raise RuntimeError(
                "Decoder did not return cross_attentions. "
                "Load model with attn_implementation='eager' if needed."
            )

        selected: List[torch.Tensor] = []
        for layer_idx, head_idx in self.alignment_heads:
            # cross_attentions[layer] shape:
            # [B, num_heads, target_tokens, source_frames]
            selected.append(cross_attentions[layer_idx][:, head_idx, :, :])

        # [B, selected_heads, target_tokens, source_frames]
        return torch.stack(selected, dim=1)


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


def get_required_alignment_heads(model: WhisperForConditionalGeneration) -> List[Tuple[int, int]]:
    """
    Production rule:
      1. First trust model.generation_config.alignment_heads.
      2. Then check model.config.alignment_heads.
      3. If absent, fail loudly unless the caller intentionally supplies a manual list.

    Do not silently use arbitrary heads for production. Word timestamp quality depends heavily
    on using the correct alignment heads.
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
        "For custom checkpoints, provide a manual verified alignment_heads list; do not use a blind fallback."
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


def export_all(
    model_id: str,
    output_dir: str | Path,
    opset: int = 17,
    prompt_len: int = 4,
    past_len: int = 4,
    manual_alignment_heads: List[Tuple[int, int]] | None = None,
):
    out_dir = ensure_dir(output_dir)

    # Use eager attention so output_attentions=True works reliably for alignment export.
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        attn_implementation="eager",
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

    dummy_mel = torch.randn(1, num_mel_bins, 3000, dtype=torch.float32)
    dummy_hidden = torch.randn(1, max_source_positions, cfg.d_model, dtype=torch.float32)
    dummy_prompt_ids = torch.ones(1, prompt_len, dtype=torch.long)
    dummy_single_id = torch.ones(1, 1, dtype=torch.long)
    dummy_cache_position = torch.tensor([prompt_len], dtype=torch.long)

    # 1. Encoder
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

    # 2. Decoder init
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

    # 3. Decoder step
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
        do_constant_folding=True,
        dynamo=False,
    )

    # 4. Decoder align
    align_wrapper = WhisperDecoderAlignWrapper(model, alignment_heads=alignment_heads)
    dummy_align_ids = torch.ones(1, 16, dtype=torch.long)

    with torch.no_grad():
        _ = align_wrapper(dummy_align_ids, dummy_hidden)

    torch.onnx.export(
        align_wrapper,
        (dummy_align_ids, dummy_hidden),
        str(out_dir / "decoder_align.onnx"),
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["alignment_heads"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "target_sequence"},
            "encoder_hidden_states": {0: "batch"},
            "alignment_heads": {0: "batch", 2: "target_sequence"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

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
        "artifacts": {
            "encoder": "encoder_model.onnx",
            "decoder_init": "decoder_init.onnx",
            "decoder_step": "decoder_step.onnx",
            "decoder_align": "decoder_align.onnx",
        },
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    tokenizer.save_pretrained(out_dir)
    print(f"Exported Whisper ONNX artifacts to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--prompt-len", type=int, default=4)
    parser.add_argument("--past-len", type=int, default=4)
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
    )


if __name__ == "__main__":
    main()
```

---

# TypeScript runtime

Create:

```text
src/whisper/
  WhisperEngine.ts
  logits.ts
  timestamps.ts
  alignment.ts
  confidence.ts
```

## `logits.ts`

```typescript
export interface LogitProcessorOptions {
  suppressTokens?: number[] | null;
  beginSuppressTokens?: number[] | null;
  timestampBegin?: number | null;
  noTimestampsTokenId?: number | null;
}

export function suppressTokenIds(
  logits: Float32Array,
  tokenIds: number[] | null | undefined,
): void {
  if (!tokenIds) return;
  for (const id of tokenIds) {
    if (id >= 0 && id < logits.length) logits[id] = -Infinity;
  }
}

export function argmaxWithLogprob(logits: Float32Array): {
  tokenId: number;
  logprob: number;
  prob: number;
} {
  let maxVal = -Infinity;
  let maxIdx = 0;

  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (v > maxVal) {
      maxVal = v;
      maxIdx = i;
    }
  }

  let sumExp = 0;
  for (let i = 0; i < logits.length; i++) {
    sumExp += Math.exp(logits[i] - maxVal);
  }

  const logsumexp = maxVal + Math.log(sumExp);
  const logprob = logits[maxIdx] - logsumexp;

  return {
    tokenId: maxIdx,
    logprob,
    prob: Math.exp(logprob),
  };
}

export function extractLastStepLogits(logitsTensor: any): Float32Array {
  const data = logitsTensor.data as Float32Array;
  const dims = logitsTensor.dims as number[];

  if (dims.length !== 3) {
    throw new Error(`Expected logits dims [B, T, V], got ${JSON.stringify(dims)}`);
  }

  const batch = dims[0];
  const sequence = dims[1];
  const vocab = dims[2];

  if (batch !== 1) {
    throw new Error("This runtime currently supports batch=1 only.");
  }

  const offset = (sequence - 1) * vocab;
  return data.slice(offset, offset + vocab);
}
```

## `timestamps.ts`

```typescript
export interface SegmentTimestamp {
  start: number;
  end: number;
  tokens: number[];
  text?: string;
}

export function isTimestampToken(tokenId: number, timestampBegin: number): boolean {
  return tokenId >= timestampBegin;
}

export function timestampTokenToSeconds(tokenId: number, timestampBegin: number): number {
  return (tokenId - timestampBegin) * 0.02;
}

export function splitSegmentsByTimestampTokens(
  tokens: number[],
  timestampBegin: number,
): SegmentTimestamp[] {
  const segments: SegmentTimestamp[] = [];

  let currentStart: number | null = null;
  let currentTextTokens: number[] = [];

  for (const token of tokens) {
    if (isTimestampToken(token, timestampBegin)) {
      const t = timestampTokenToSeconds(token, timestampBegin);

      if (currentStart === null) {
        currentStart = t;
      } else {
        segments.push({
          start: currentStart,
          end: t,
          tokens: currentTextTokens,
        });
        currentStart = t;
        currentTextTokens = [];
      }
    } else {
      currentTextTokens.push(token);
    }
  }

  if (currentStart !== null && currentTextTokens.length > 0) {
    segments.push({
      start: currentStart,
      end: currentStart,
      tokens: currentTextTokens,
    });
  }

  return segments;
}
```

## `confidence.ts`

```typescript
export interface TokenScore {
  tokenId: number;
  logprob: number;
  prob: number;
}

export interface SegmentConfidence {
  averageLogprob: number;
  noSpeechProbability: number | null;
  compressionRatio: number | null;
  isReliable: boolean;
  reasons: string[];
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function averageLogprob(scores: TokenScore[]): number {
  if (scores.length === 0) return -Infinity;
  return scores.reduce((sum, s) => sum + s.logprob, 0) / scores.length;
}

export async function estimateCompressionRatio(text: string): Promise<number | null> {
  // For browser production, implement gzip using CompressionStream when available,
  // or a small gzip/deflate package. Return null until implemented.
  if (!text || text.length === 0) return null;
  return null;
}

export async function evaluateSegmentConfidence(args: {
  tokenScores: TokenScore[];
  noSpeechProbability: number | null;
  text: string;
  logprobThreshold?: number;
  noSpeechThreshold?: number;
  compressionRatioThreshold?: number;
}): Promise<SegmentConfidence> {
  const logprobThreshold = args.logprobThreshold ?? -1.0;
  const noSpeechThreshold = args.noSpeechThreshold ?? 0.6;
  const compressionRatioThreshold = args.compressionRatioThreshold ?? 2.4;

  const avg = averageLogprob(args.tokenScores);
  const compression = await estimateCompressionRatio(args.text);

  const reasons: string[] = [];

  if (avg < logprobThreshold) {
    reasons.push(`averageLogprob ${avg.toFixed(3)} < ${logprobThreshold}`);
  }

  if (
    args.noSpeechProbability !== null &&
    args.noSpeechProbability > noSpeechThreshold &&
    avg < logprobThreshold
  ) {
    reasons.push(
      `noSpeechProbability ${args.noSpeechProbability.toFixed(3)} > ${noSpeechThreshold}`,
    );
  }

  if (compression !== null && compression > compressionRatioThreshold) {
    reasons.push(`compressionRatio ${compression.toFixed(3)} > ${compressionRatioThreshold}`);
  }

  return {
    averageLogprob: avg,
    noSpeechProbability: args.noSpeechProbability,
    compressionRatio: compression,
    isReliable: reasons.length === 0,
    reasons,
  };
}
```

## `alignment.ts`

```typescript
export interface WordTimestamp {
  word: string;
  start: number;
  end: number;
  tokenStart: number;
  tokenEnd: number;
  probability: number | null;
}

export function medianFilter1D(values: Float32Array, width: number): Float32Array {
  if (width <= 1) return values.slice();
  if (width % 2 === 0) throw new Error("median filter width must be odd");

  const radius = Math.floor(width / 2);
  const out = new Float32Array(values.length);

  for (let i = 0; i < values.length; i++) {
    const window: number[] = [];
    const lo = Math.max(0, i - radius);
    const hi = Math.min(values.length - 1, i + radius);

    for (let j = lo; j <= hi; j++) window.push(values[j]);
    window.sort((a, b) => a - b);
    out[i] = window[Math.floor(window.length / 2)];
  }

  return out;
}

export function normalizeAcrossTokensThenAverageHeads(
  alignmentHeads: Float32Array,
  selectedHeads: number,
  targetTokens: number,
  sourceFrames: number,
  medianWidth = 7,
): Float32Array {
  // Input shape flattened from [H, T, F] for batch=1.
  // HF/OpenAI-style path normalizes across token dimension, median-filters, then averages heads.
  const averaged = new Float32Array(targetTokens * sourceFrames);

  for (let h = 0; h < selectedHeads; h++) {
    const headOffset = h * targetTokens * sourceFrames;

    const normalized = new Float32Array(targetTokens * sourceFrames);

    for (let f = 0; f < sourceFrames; f++) {
      let mean = 0;
      for (let t = 0; t < targetTokens; t++) {
        mean += alignmentHeads[headOffset + t * sourceFrames + f];
      }
      mean /= targetTokens;

      let variance = 0;
      for (let t = 0; t < targetTokens; t++) {
        const d = alignmentHeads[headOffset + t * sourceFrames + f] - mean;
        variance += d * d;
      }
      const std = Math.sqrt(variance / Math.max(1, targetTokens));
      const denom = std > 1e-6 ? std : 1e-6;

      for (let t = 0; t < targetTokens; t++) {
        normalized[t * sourceFrames + f] =
          (alignmentHeads[headOffset + t * sourceFrames + f] - mean) / denom;
      }
    }

    const filtered = new Float32Array(normalized.length);
    for (let t = 0; t < targetTokens; t++) {
      const row = normalized.slice(t * sourceFrames, (t + 1) * sourceFrames);
      const rowFiltered = medianFilter1D(row, medianWidth);
      filtered.set(rowFiltered, t * sourceFrames);
    }

    for (let i = 0; i < averaged.length; i++) averaged[i] += filtered[i] / selectedHeads;
  }

  return averaged;
}

export function dtwPathFromMatrix(
  matrix: Float32Array,
  targetTokens: number,
  sourceFrames: number,
): number[] {
  // Cost is negative alignment: higher alignment -> lower cost.
  const dp = new Float64Array(targetTokens * sourceFrames);
  const back = new Int8Array(targetTokens * sourceFrames);

  for (let i = 0; i < dp.length; i++) dp[i] = Infinity;
  dp[0] = -matrix[0];

  for (let t = 0; t < targetTokens; t++) {
    for (let f = 0; f < sourceFrames; f++) {
      if (t === 0 && f === 0) continue;

      const idx = t * sourceFrames + f;
      let best = Infinity;
      let move = 0;

      if (t > 0) {
        const v = dp[(t - 1) * sourceFrames + f];
        if (v < best) {
          best = v;
          move = 1;
        }
      }

      if (f > 0) {
        const v = dp[t * sourceFrames + (f - 1)];
        if (v < best) {
          best = v;
          move = 2;
        }
      }

      if (t > 0 && f > 0) {
        const v = dp[(t - 1) * sourceFrames + (f - 1)];
        if (v < best) {
          best = v;
          move = 3;
        }
      }

      dp[idx] = -matrix[idx] + best;
      back[idx] = move;
    }
  }

  let t = targetTokens - 1;
  let f = sourceFrames - 1;
  const tokenToFrame = new Array<number>(targetTokens).fill(0);

  while (t >= 0 && f >= 0) {
    tokenToFrame[t] = f;
    const move = back[t * sourceFrames + f];

    if (t === 0 && f === 0) break;
    if (move === 1) t -= 1;
    else if (move === 2) f -= 1;
    else {
      t -= 1;
      f -= 1;
    }
  }

  return tokenToFrame;
}

export function tokensToWordTimestamps(args: {
  generatedTokens: number[];
  tokenToText: (tokens: number[]) => string;
  tokenScores?: { tokenId: number; prob: number }[];
  alignmentHeads: Float32Array;
  selectedHeads: number;
  timestampBegin: number;
  sourceFrames?: number;
  frameSeconds?: number;
}): WordTimestamp[] {
  const sourceFrames = args.sourceFrames ?? 1500;
  const frameSeconds = args.frameSeconds ?? 0.02;

  const nonTimestampTokenIndices: number[] = [];
  const nonTimestampTokens: number[] = [];

  for (let i = 0; i < args.generatedTokens.length; i++) {
    const token = args.generatedTokens[i];
    if (token < args.timestampBegin) {
      nonTimestampTokenIndices.push(i);
      nonTimestampTokens.push(token);
    }
  }

  if (nonTimestampTokens.length === 0) return [];

  const compactHeads = new Float32Array(
    args.selectedHeads * nonTimestampTokens.length * sourceFrames,
  );

  // alignmentHeads is [H, full_generated_T, F] after prompt rows are removed.
  for (let h = 0; h < args.selectedHeads; h++) {
    for (let compactIdx = 0; compactIdx < nonTimestampTokenIndices.length; compactIdx++) {
      const originalIdx = nonTimestampTokenIndices[compactIdx];
      const src = h * args.generatedTokens.length * sourceFrames + originalIdx * sourceFrames;
      const dst = h * nonTimestampTokens.length * sourceFrames + compactIdx * sourceFrames;
      compactHeads.set(args.alignmentHeads.slice(src, src + sourceFrames), dst);
    }
  }

  const matrix = normalizeAcrossTokensThenAverageHeads(
    compactHeads,
    args.selectedHeads,
    nonTimestampTokens.length,
    sourceFrames,
    7,
  );

  const tokenToFrame = dtwPathFromMatrix(matrix, nonTimestampTokens.length, sourceFrames);

  const words: WordTimestamp[] = [];
  let currentTokens: number[] = [];
  let tokenStart = 0;

  function flush(tokenEndExclusive: number) {
    if (currentTokens.length === 0) return;

    const word = args.tokenToText(currentTokens).trim();
    if (!word) {
      currentTokens = [];
      tokenStart = tokenEndExclusive;
      return;
    }

    const startFrame = tokenToFrame[tokenStart] ?? 0;
    const endFrame = tokenToFrame[Math.max(tokenStart, tokenEndExclusive - 1)] ?? startFrame;

    let probability: number | null = null;
    if (args.tokenScores) {
      const probs = args.tokenScores.slice(tokenStart, tokenEndExclusive).map((s) => s.prob);
      if (probs.length > 0) probability = probs.reduce((a, b) => a + b, 0) / probs.length;
    }

    words.push({
      word,
      start: startFrame * frameSeconds,
      end: Math.max((endFrame + 1) * frameSeconds, startFrame * frameSeconds + frameSeconds),
      tokenStart,
      tokenEnd: tokenEndExclusive,
      probability,
    });

    currentTokens = [];
    tokenStart = tokenEndExclusive;
  }

  for (let i = 0; i < nonTimestampTokens.length; i++) {
    const token = nonTimestampTokens[i];
    const piece = args.tokenToText([token]);
    const startsNewWord = piece.startsWith(" ") && currentTokens.length > 0;

    if (startsNewWord) {
      flush(i);
      tokenStart = i;
    }

    currentTokens.push(token);
  }

  flush(nonTimestampTokens.length);
  return words;
}
```

## `WhisperEngine.ts`

```typescript
import * as ort from 'onnxruntime-web';
import { argmaxWithLogprob, extractLastStepLogits, suppressTokenIds } from './logits';
import { splitSegmentsByTimestampTokens } from './timestamps';
import { evaluateSegmentConfidence, sigmoid, TokenScore } from './confidence';
import { tokensToWordTimestamps, WordTimestamp } from './alignment';

export interface WhisperManifest {
  model_id: string;
  num_mel_bins: number;
  max_source_positions: number;
  max_target_positions: number;
  decoder_layers: number;
  decoder_attention_heads: number;
  head_dim: number;
  vocab_size: number;
  alignment_heads: Array<[number, number]>;
  special_tokens: {
    eos_token_id: number;
    bos_token_id: number;
    pad_token_id: number;
    decoder_start_token_id: number;
    forced_decoder_ids?: Array<[number, number]> | null;
    suppress_tokens?: number[] | null;
    begin_suppress_tokens?: number[] | null;
    no_timestamps_token_id?: number | null;
    no_speech_token_id?: number | null;
    timestamp_begin?: number | null;
  };
  artifacts: {
    encoder: string;
    decoder_init: string;
    decoder_step: string;
    decoder_align: string;
  };
}

export interface DecodeOptions {
  languageTokenId?: number;
  taskTokenId?: number;
  timestamps?: boolean;
  maxTokens?: number;
}

export interface DecodeResult {
  tokens: number[];
  tokenScores: TokenScore[];
  text: string;
  segments: Array<{
    start: number;
    end: number;
    text: string;
    tokens: number[];
  }>;
  words: WordTimestamp[];
  confidence: Awaited<ReturnType<typeof evaluateSegmentConfidence>>;
}

export class WhisperEngine {
  private manifest!: WhisperManifest;
  private encoderSession!: ort.InferenceSession;
  private initSession!: ort.InferenceSession;
  private stepSession!: ort.InferenceSession;
  private alignSession!: ort.InferenceSession;

  constructor(private readonly tokenToText: (tokens: number[]) => string) {}

  async load(manifest: WhisperManifest, baseUrl: string): Promise<void> {
    this.manifest = manifest;

    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: ['webgpu', 'wasm'],
    };

    this.encoderSession = await ort.InferenceSession.create(
      `${baseUrl}/${manifest.artifacts.encoder}`,
      sessionOptions,
    );

    this.initSession = await ort.InferenceSession.create(
      `${baseUrl}/${manifest.artifacts.decoder_init}`,
      sessionOptions,
    );

    this.stepSession = await ort.InferenceSession.create(
      `${baseUrl}/${manifest.artifacts.decoder_step}`,
      sessionOptions,
    );

    this.alignSession = await ort.InferenceSession.create(
      `${baseUrl}/${manifest.artifacts.decoder_align}`,
      sessionOptions,
    );
  }

  buildPrompt(options: DecodeOptions): number[] {
    const s = this.manifest.special_tokens;
    const prompt: number[] = [];

    prompt.push(s.decoder_start_token_id);

    if (options.languageTokenId !== undefined) prompt.push(options.languageTokenId);
    if (options.taskTokenId !== undefined) prompt.push(options.taskTokenId);

    if (
      !options.timestamps &&
      s.no_timestamps_token_id !== null &&
      s.no_timestamps_token_id !== undefined
    ) {
      prompt.push(s.no_timestamps_token_id);
    }

    return prompt;
  }

  async encode(inputFeatures: Float32Array): Promise<ort.Tensor> {
    const expected = this.manifest.num_mel_bins * 3000;
    if (inputFeatures.length !== expected) {
      throw new Error(`Expected ${expected} log-Mel values, got ${inputFeatures.length}`);
    }

    const input = new ort.Tensor('float32', inputFeatures, [1, this.manifest.num_mel_bins, 3000]);

    const outputs = await this.encoderSession.run({
      input_features: input,
    });

    return outputs.last_hidden_state;
  }

  async decodeFromEncoder(
    encoderHiddenStates: ort.Tensor,
    options: DecodeOptions = {},
  ): Promise<DecodeResult> {
    const s = this.manifest.special_tokens;

    if (s.timestamp_begin === null || s.timestamp_begin === undefined) {
      throw new Error('manifest.special_tokens.timestamp_begin is required for timestamp support.');
    }

    const prompt = this.buildPrompt(options);
    const maxTokens = options.maxTokens ?? this.manifest.max_target_positions;

    const promptTensor = new ort.Tensor(
      'int64',
      BigInt64Array.from(prompt.map(BigInt)),
      [1, prompt.length],
    );

    const initOutputs = await this.initSession.run({
      input_ids: promptTensor,
      encoder_hidden_states: encoderHiddenStates,
    });

    const activeCache: Record<string, ort.Tensor> = {};

    for (let i = 0; i < this.manifest.decoder_layers; i++) {
      activeCache[`past_key_values.${i}.decoder.key`] = initOutputs[`present.${i}.decoder.key`];
      activeCache[`past_key_values.${i}.decoder.value`] = initOutputs[`present.${i}.decoder.value`];
      activeCache[`past_key_values.${i}.encoder.key`] = initOutputs[`present.${i}.encoder.key`];
      activeCache[`past_key_values.${i}.encoder.value`] = initOutputs[`present.${i}.encoder.value`];
    }

    const generatedTokens: number[] = [];
    const tokenScores: TokenScore[] = [];

    const initLogits = extractLastStepLogits(initOutputs.logits);
    suppressTokenIds(initLogits, s.begin_suppress_tokens ?? null);

    if (!options.timestamps) {
      for (let i = s.timestamp_begin; i < initLogits.length; i++) initLogits[i] = -Infinity;
    }

    let selected = argmaxWithLogprob(initLogits);
    let nextInputToken = selected.tokenId;
    let currentSequenceLength = prompt.length;

    while (
      nextInputToken !== s.eos_token_id &&
      generatedTokens.length < maxTokens - prompt.length
    ) {
      generatedTokens.push(nextInputToken);
      tokenScores.push(selected);

      const stepInput = new ort.Tensor(
        'int64',
        BigInt64Array.from([BigInt(nextInputToken)]),
        [1, 1],
      );

      const cachePosition = new ort.Tensor(
        'int64',
        BigInt64Array.from([BigInt(currentSequenceLength)]),
        [1],
      );

      const stepOutputs = await this.stepSession.run({
        input_ids: stepInput,
        encoder_hidden_states: encoderHiddenStates,
        cache_position: cachePosition,
        ...activeCache,
      });

      for (let i = 0; i < this.manifest.decoder_layers; i++) {
        activeCache[`past_key_values.${i}.decoder.key`] = stepOutputs[`present.${i}.decoder.key`];
        activeCache[`past_key_values.${i}.decoder.value`] = stepOutputs[`present.${i}.decoder.value`];
        // Encoder cross KV stays from init.
      }

      currentSequenceLength += 1;

      const stepLogits = extractLastStepLogits(stepOutputs.logits);
      suppressTokenIds(stepLogits, s.suppress_tokens ?? null);

      if (!options.timestamps) {
        for (let i = s.timestamp_begin; i < stepLogits.length; i++) stepLogits[i] = -Infinity;
      }

      selected = argmaxWithLogprob(stepLogits);
      nextInputToken = selected.tokenId;
    }

    const text = this.tokenToText(generatedTokens);

    let noSpeechProbability: number | null = null;
    if (s.no_speech_token_id !== null && s.no_speech_token_id !== undefined) {
      const raw = initLogits[s.no_speech_token_id];
      if (Number.isFinite(raw)) noSpeechProbability = sigmoid(raw);
    }

    const segmentTokens = splitSegmentsByTimestampTokens(generatedTokens, s.timestamp_begin);
    const segments = segmentTokens.map((seg) => ({
      start: seg.start,
      end: seg.end,
      tokens: seg.tokens,
      text: this.tokenToText(seg.tokens),
    }));

    const fullIds = [...prompt, ...generatedTokens];
    const fullIdsTensor = new ort.Tensor(
      'int64',
      BigInt64Array.from(fullIds.map(BigInt)),
      [1, fullIds.length],
    );

    const alignOutputs = await this.alignSession.run({
      input_ids: fullIdsTensor,
      encoder_hidden_states: encoderHiddenStates,
    });

    const fullAlignmentHeads = alignOutputs.alignment_heads.data as Float32Array;
    const sourceFrames = this.manifest.max_source_positions;
    const selectedHeads = this.manifest.alignment_heads.length;

    // Shape is [1, H, fullIds.length, F]. Drop prompt rows for every selected head.
    const generatedAlignmentHeads = new Float32Array(
      selectedHeads * generatedTokens.length * sourceFrames,
    );

    for (let h = 0; h < selectedHeads; h++) {
      const srcBase = h * fullIds.length * sourceFrames + prompt.length * sourceFrames;
      const dstBase = h * generatedTokens.length * sourceFrames;
      generatedAlignmentHeads.set(
        fullAlignmentHeads.slice(srcBase, srcBase + generatedTokens.length * sourceFrames),
        dstBase,
      );
    }

    const words = tokensToWordTimestamps({
      generatedTokens,
      tokenToText: this.tokenToText,
      tokenScores,
      alignmentHeads: generatedAlignmentHeads,
      selectedHeads,
      timestampBegin: s.timestamp_begin,
      sourceFrames,
      frameSeconds: 0.02,
    });

    const confidence = await evaluateSegmentConfidence({
      tokenScores,
      noSpeechProbability,
      text,
    });

    return {
      tokens: generatedTokens,
      tokenScores,
      text,
      segments,
      words,
      confidence,
    };
  }

  async transcribeChunk(
    inputFeatures: Float32Array,
    options: DecodeOptions = {},
  ): Promise<DecodeResult> {
    const encoderHiddenStates = await this.encode(inputFeatures);
    return this.decodeFromEncoder(encoderHiddenStates, options);
  }
}
```

---

# Alignment-head rule

Do **not** trust a generic fallback alignment-head selection for production.

A fallback like “last half of decoder layers, all heads” may produce something, but word timestamp quality depends heavily on using the correct timing-sensitive cross-attention heads. The exporter should first look for official model-provided alignment heads from:

```text
model.generation_config.alignment_heads
model.config.alignment_heads
```

Then preserve the exact list in `manifest.json`:

```json
{
  "alignment_heads": [[2, 1], [3, 5]],
  "alignment_heads_source": "generation_config_or_config"
}
```

If official metadata is absent, the exporter should fail loudly by default. Allow a manual override only when the heads were verified:

```bash
python export_whisper.py openai/whisper-tiny ./out --alignment-heads "2:1,3:5,4:2"
```

This is not a cosmetic detail. It has one of the largest impacts on word-level timestamp quality.

---

# Timestamp priority order

Implement in this order:

```text
1. Encoder export and validation
2. Decoder init + step export
3. KV-cache generation loop
4. Basic text decoding
5. Segment timestamps from timestamp tokens
6. Token logprobs and average logprob
7. no_speech probability
8. decoder_align.onnx export
9. Official alignment-head extraction and manifest preservation
10. Word timestamps with normalization + median filter + DTW
11. Tokenizer-aware word grouping
12. Compression ratio / hallucination filters
13. WebGPU + WASM parity tests
```

---

# Validation requirements

## 1. ONNX checker

```python
import onnx

for path in [
    "encoder_model.onnx",
    "decoder_init.onnx",
    "decoder_step.onnx",
    "decoder_align.onnx",
]:
    model = onnx.load(path)
    onnx.checker.check_model(model)
```

## 2. PyTorch vs ONNX parity

Create a validation script that loads the same Whisper model in PyTorch and compares against ONNX Runtime CPU.

Test:

```text
encoder

decoder_init

decoder_step for at least 5 autoregressive steps

decoder_align
```

Target tolerances:

```text
FP32:
  logits max_abs_diff <= 1e-4
  KV tensors max_abs_diff <= 1e-4 or <= 1e-3

FP16:
  logits max_abs_diff <= 1e-2 may be acceptable
```

## 3. Browser tests

Run in headless Chromium:

```text
WASM backend
WebGPU backend
token ID comparison
segment timestamp comparison
word timestamp comparison within <= 40 ms where possible
```

## 4. Golden audio fixtures

Use:

```text
silence
single clean English sentence
multilingual sentence
noisy background
music + speech
speech shorter than 3 seconds padded to 30 seconds
```

Store expected outputs generated from the original PyTorch/HF model:

```json
{
  "model_id": "openai/whisper-tiny",
  "audio": "sample.wav",
  "tokens": [],
  "text": "",
  "segments": [],
  "words": [],
  "avg_logprob": 0.0,
  "no_speech_probability": 0.0
}
```

---

# Definition of done

The task is complete only when:

```text
- exporter creates all four ONNX files
- manifest.json is created
- tokenizer files are saved/copied
- official alignment_heads are extracted or manually verified
- alignment_heads are preserved in manifest.json
- exporter fails loudly if alignment_heads are missing and no manual override is provided
- ONNX checker passes
- PyTorch vs ONNX CPU validation passes
- browser runtime transcribes a 30-second chunk
- KV-cache generation works
- segment timestamps work
- decoder_align produces selected cross-attention heads
- word timestamps are produced through normalization + median filter + DTW
- confidence metrics are returned
- WASM and WebGPU backends both run
- golden audio tests pass
```

---

# References

[^openai-readme]: OpenAI Whisper README. https://github.com/openai/whisper
[^hf-whisper-docs]: Hugging Face Transformers Whisper documentation. https://huggingface.co/docs/transformers/en/model_doc/whisper
[^hf-kv-cache]: Hugging Face Transformers KV cache documentation. https://huggingface.co/docs/transformers/en/kv_cache
[^hf-generation-whisper]: Hugging Face Transformers `generation_whisper.py`. https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/generation_whisper.py
[^hf-convert-alignment]: Hugging Face Transformers `convert_openai_to_hf.py`, alignment-head handling. https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/convert_openai_to_hf.py
