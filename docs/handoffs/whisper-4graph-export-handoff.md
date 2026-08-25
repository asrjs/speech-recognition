# Whisper ONNX 4-Graph Export — Session Handoff

Date: 2026-05-29
Branch: `feat/asr-pipeline-output-formats`
Commits: `511fcee` → `35e9fcc` → (pending)

## What was done

Implemented self-contained 4-graph Whisper ONNX export with proper KV-cache decoder split, replacing the merged-decoder approach.

### 4-Graph Architecture

| Graph | Purpose | Size (tiny) | Key inputs | Key outputs |
|-------|---------|-------------|------------|-------------|
| `encoder_model.onnx` | Mel → hidden states | 31 MB | `input_features` [1,n_mels,3000] | `last_hidden_state` [1,1500,d] |
| `decoder_init.onnx` | Prompt/prefill, creates KV cache | 189 MB | `input_ids`, `encoder_hidden_states` | `logits` + 4×4 KV per layer |
| `decoder_step.onnx` | Single-token autoregressive | 108 MB | `input_ids` [1,1] + 4×4 past KV | `logits` [1,1,vocab] + self-attn KV |
| `decoder_align.onnx` | Cross-attention for DTW | 107 MB | `input_ids`, `encoder_hidden_states` | current: raw `alignment` [1,N,T,1500]; legacy: averaged [1,T,1500] |

### Key Design Decisions

1. **Split init/step avoids DynamicCache tracing** — HF 5.x `EncoderDecoderCache` with data-dependent branching cannot be traced by `torch.onnx.export(dynamo=False)`
2. **`decoder_step` needs NO `encoder_hidden_states`** — cross-attention K/V come from `past_key_values.{i}.encoder.{key,value}`. `cache_position` derived from cache length.
3. **Manual alignment wrapper** — runs decoder blocks directly, captures only the selected `encoder_attn` cross-attention logits. Avoids `aten::diff` which has no ONNX lowering. All softmax, DTW, median-filter, and timestamp logic stays in TypeScript.
4. **HF 5.x compat** — `EncoderDecoderCache` yields 6-element tuples `(self_k, self_v, None, cross_k, cross_v, None)`. `build_encoder_decoder_cache_from_flat()` constructs proper `DynamicCache` + `EncoderDecoderCache` objects.
5. **Current `decoder_align` returns selected raw logits `[B, N, T, S]`** — N is the selected `alignment_heads` count. The runtime performs crop, per-head softmax/normalization, median filtering, and head averaging. Older averaged `[B, T, S]` post-softmax artifacts remain a compatibility format.

### HF 5.x Pitfalls Encountered

- `EncoderDecoderCache.__init__(*caches)` takes TWO positional args: `EncoderDecoderCache(self_cache, cross_cache)` — NOT keyword args
- `Decoder.forward()` expects `Cache` objects, not raw tuples — must wrap legacy tuples
- `DynamicCache` entries are 3-element tuples `(k, v, None)` — the None is a slot for sliding window
- `to_legacy_cache()` handles both 4-element (legacy) and 6-element (HF 5.x) tuples

### aten::diff Resolution

- `output_attentions=True` path activates HF Whisper internal code that uses `torch.diff` in positional encoding helpers
- `aten::diff` has NO ONNX lowering in any opset (17, 18, 21 tested)
- `torch.export` (`dynamo=True`) also fails due to data-dependent guards in HF Whisper
- Solution: manual decoder block iteration that captures only `encoder_attn(..., output_attentions=True)` per layer
- Verified with monkey-patch: 0 `torch.diff` calls in align wrapper forward pass

## Verification Results

### Tests Created
- `test_kv_export.py` — validates ONNX graph structure (input/output names, shapes, ORT loading)
- `test_e2e_tokens.py` — synthetic audio ONNX vs PyTorch token comparison
- `test_comprehensive.py` — real speech, alignment, quantization parity

### E2E Token Match
| Test | Result |
|------|--------|
| Synthetic (440Hz sine) | 5/5 tokens exact match |
| Real speech (JFK, 11s) | 27/27 tokens (100%) exact match |
| Alignment shape | [1, 27, 1500] ✓ |
| Legacy attention normalization | row sums = 1.0000 ✓ |
| Alignment values | [0.0000, 0.1796] non-negative ✓ |
| fp16 vs fp32 | 1/1 tokens (100%) ✓ |
| int8 vs fp32 | 1/1 tokens (100%) ✓ |

### TypeScript Gate
- `npm run typecheck` ✓
- `npm run lint` (0e 4w) ✓
- `npm test` (76 files, 366 tests) ✓
- `npm run build` ✓

## Remaining Work

1. **Wire 4-graph format in TypeScript executor** — currently uses merged decoder with `use_cache_branch`; needs separate init/step sessions for self-exported models
2. **4-graph artifact source type** — add preset/artifact source for local file-based 4-graph models
3. **Timestamp logit processor** (Task 16) — suppression rules during generation; deferred, not needed for DTW word timestamps
4. **Larger model validation** — currently tested with whisper-tiny; whisper-base/small/large-v3-turbo need config-driven dimension fix in executor.ts

## How to Resume

```bash
cd ~/github/asrjs/speech-recognition
git checkout feat/asr-pipeline-output-formats

# Python export tool
cd tools/whisper-onnx-export
.venv/bin/python export_whisper.py openai/whisper-tiny ./output/tiny

# Run tests
.venv/bin/python test_kv_export.py
.venv/bin/python test_e2e_tokens.py
.venv/bin/python test_comprehensive.py
.venv/bin/python test_comprehensive.py --quantize

# TypeScript gate
cd ~/github/asrjs/speech-recognition
npm run typecheck && npm run lint && npm test && npm run build
```
