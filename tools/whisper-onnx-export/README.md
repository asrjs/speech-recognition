# Whisper ONNX Export Tool — 4-Graph KV-Cache Architecture

Produces self-contained Whisper ONNX artifacts for ASR.js with proper KV-cache decoder support.

## Usage

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Export whisper-tiny (4 graphs)
python export_whisper.py openai/whisper-tiny ./output/whisper-tiny

# Export with quantization variants
python export_whisper.py openai/whisper-base ./output/whisper-base --fp16 --int8

# Custom alignment heads (when official metadata missing)
python export_whisper.py openai/whisper-tiny ./output --alignment-heads "2:2,3:0"
```

## Output Structure

```
output/whisper-tiny/
  manifest.json                  — model metadata (format: whisper-browser-self-export-v1)
  tokenizer.json                 — BPE tokenizer
  generation_config.json         — alignment_heads, suppress tokens
  config.json                    — model config (layers, heads, dims)
  encoder_model.onnx             — mel → encoder hidden states (31 MB)
  decoder_init.onnx              — prompt/prefill decoder with KV cache init (189 MB)
  decoder_step.onnx              — single-token autoregressive with KV reuse (108 MB)
  decoder_align.onnx             — cross-attention alignment for word timestamps (107 MB)
```

All sizes for whisper-tiny fp32. Multiply by ~1.9x for whisper-base.

## 4-Graph Architecture

| Graph | Purpose | Runs | Key design |
|-------|---------|------|------------|
| `encoder_model.onnx` | Mel → hidden states | Once per chunk | Clean fixed path |
| `decoder_init.onnx` | Prompt/prefill, creates KV cache | Once per chunk | 4 KV tensors per layer output |
| `decoder_step.onnx` | Single-token autoregressive | Many times | Branch-free, only self-attn KV updated |
| `decoder_align.onnx` | Cross-attention for DTW alignment | Once after gen | Manual decoder block capture, no aten::diff |

### Graph Details

#### encoder_model.onnx
- Input: `input_features` [batch, n_mels, 3000]
- Output: `last_hidden_state` [batch, 1500, d_model]

#### decoder_init.onnx
- Inputs: `input_ids` [batch, prompt_length], `encoder_hidden_states`
- Outputs: `logits` + `present.{i}.decoder.key/.value` + `present.{i}.encoder.key/.value` (4 per layer)

#### decoder_step.onnx
- Inputs: `input_ids` [batch, 1] + all `past_key_values.*.{key,value}` (decoder + encoder K/V)
- Outputs: `logits` [batch, 1, vocab] + `present.{i}.decoder.key/.value` (self-attn only)
- Cross-attention KV preserved from init, never output by step

#### decoder_align.onnx
- Inputs: `input_ids` [batch, T], `encoder_hidden_states`
- Output from the current exporter: `alignment` [batch, N, T, 1500], where N is the
  selected alignment-head count and the values are raw cross-attention logits
- The manifest declares this contract as `attention_values: "logits"` and
  `attention_layout: "selected_heads"`. The TypeScript runtime crops the padded
  frame axis, softmaxes each head, normalizes all teacher-forced rows, median
  filters, averages heads, and then selects the no-timestamps anchor plus text
  rows. This preserves the reference Whisper order of operations.
- Older exports may still return `[batch, T, 1500]` post-softmax probabilities
  with `attention_layout: "mean"`; the runtime keeps that compatibility path.
- No DTW, timestamp logic, or torch.diff in ONNX — all post-processing in TypeScript
- The manually unrolled decoder supplies Whisper's causal self-attention mask;
  re-export alignment graphs after changing Transformers versions and validate
  them against the regular decoder before publishing.

## Verification

```bash
# Structural validation
python test_kv_export.py

# E2E token match vs PyTorch (synthetic audio)
python test_e2e_tokens.py

# Comprehensive: real speech (JFK), alignment, quantization parity
python test_comprehensive.py
python test_comprehensive.py --quantize
```

### HF reference artifact

`generate_hf_reference.py` records greedy no-timestamps and timestamped HF
tokens, optionally runs the exported split graphs, and can save the exact mel
input used by Transformers:

```bash
python generate_hf_reference.py \
  --model-dir ./output/whisper-tiny \
  --audio ./fixtures/jfk.wav \
  --output ./output/jfk.reference.json \
  --export-mel
```

Encoder and decoder variants may live in different directories:

```bash
python generate_hf_reference.py \
  --model-dir ./output/whisper-large-v3-turbo \
  --encoder-dir ./variants/fp32 \
  --decoder-dir ./variants/fp32 \
  --model-id openai/whisper-large-v3-turbo \
  --audio ./fixtures/jfk.wav \
  --output ./output/jfk.reference.json \
  --export-mel --skip-onnx
```

Use `--skip-onnx` when the installed Python ONNX Runtime does not support the
exported graph IR. The generated HF tokens and mel can still be consumed by
`tests/whisper-reproducibility-harness.test.ts`, which runs the graphs with
Node ORT. The TypeScript harness reads graph input/output dimensions directly;
for large-v3-turbo, `input_features` has 3000 mel frames while the encoder emits
1500 positions.

## Validation Results (whisper-tiny)

| Test | Result |
|------|--------|
| Synthetic (440Hz sine) tokens | 5/5 exact match ONNX vs PyTorch |
| Real speech (JFK, 11s) tokens | 27/27 (100%) exact match |
| Alignment shape | legacy validation: [1, 27, 1500] |
| Attention normalization | legacy post-softmax rows sum to 1.0000; current raw-logit rows are normalized at runtime |
| fp16 parity | 100% token match |
| int8 parity | 100% token match |

## Model Config Reference

| Model | Params | n_mels | d_model | layers | heads | alignment_heads |
|-------|--------|--------|---------|--------|-------|-----------------|
| whisper-tiny | ~39M | 80 | 384 | 4 | 6 | 6 heads (layers 2-3) |
| whisper-base | ~74M | 80 | 512 | 6 | 8 | 5 heads (layers 3-5) |
| whisper-small | ~244M | 80 | 768 | 12 | 12 | 10 heads (layers 5-10) |
| whisper-large-v3-turbo | ~809M | 128 | 1280 | 32 | 20 | from generation_config |
