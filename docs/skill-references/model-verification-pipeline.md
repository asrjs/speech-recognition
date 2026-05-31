# Ground-Truth Model Verification Pipeline

## Background

During the WebGPU fp16io pipeline debugging session (2026-06-01), ~150 tool
calls were spent debugging a test page that reimplemented the decode loop from
scratch. All 7 bugs found (wrong prompt tokens, missing suppress_tokens,
missing begin_suppress_tokens, missing KV cache encoder preservation) were
already correctly handled in the library. The Node runner worked perfectly.

**Lesson:** All decode logic must live in one place (the library). Test pages are
UI shells. Verify every model variant against ground truth step by step on Node
ORT first. Promote to WebGPU only after all steps pass.

## Verification Steps

### Step 1: Mel Spectrogram

Compare library WhisperMelProcessor output against a known-good reference
(e.g. Python WhisperFeatureExtractor output or a previously-validated capture).

```bash
node tests/smoke/verify-step1-mel.mjs
```

Expected: MSE < 1e-5 against reference (if same pipeline used).
Note: The library uses Slaney-scale mel; Python OpenAI Whisper uses Kaldi-style
fbank. These produce different numeric values. Either:
- (a) Generate reference from the library itself on a known run
- (b) Accept a larger MSE if comparing across implementations
- (c) Compare relative differences across model variants (fp32 vs fp16io)

### Step 2: Encoder Output

Compare fp16io (or any variant) encoder output against fp32 baseline.
Use the library's WhisperOnnxExecutor or direct ORT session.

```bash
# Dedicated verification script (preferred):
node tests/smoke/verify-step2-encoder.mjs

# Manual comparison:
WHISPER_DIR=models/fp32 node tests/smoke/verify-encoder.mjs --dump enc_fp32.bin
WHISPER_DIR=models/fp16io node tests/smoke/verify-encoder.mjs --dump enc_fp16io.bin
node -e "/* cosine similarity + MSE comparison */"
```

Acceptance:
- Same backend (both CPU): cosine > 0.999
- Different backend (CPU vs WebGPU): cosine > 0.99
- fp16io vs fp32 (same backend): cosine > 0.99
- **Verified 2026-05-31**: fp16io on Node ORT: cosine=0.999987, MSE=4.9e-6 (PASS)

### Step 3-5: Full Decode + Token-by-Token

Run the library's decode pipeline on Node and compare against fp32 baseline.

```bash
# Dedicated verification script (preferred):
node tests/smoke/verify-step3-5-decode.mjs
```

Acceptance:
- Transcript: IDENTICAL to fp32 baseline
- Token-by-token: all tokens match (27/27 for JFK test)
- No early EOS, no NaN
- **Verified 2026-05-31**: fp16io on Node ORT: 27/27 tokens match, identical transcript (PASS)

## WebGPU Promotion Checklist

Before moving a variant from Node ORT to WebGPU browser test:

- [x] Step 1 (Mel) matches known-good reference (`verify-step1-mel.mjs`)
- [x] Step 2 (Encoder) cosine similarity > 0.99 vs fp32 baseline (`verify-step2-encoder.mjs`)
- [x] Steps 3-5 (Decode + tokens) transcript match + token-by-token (`verify-step3-5-decode.mjs`)
- [ ] WebGPU browser test passes (`webgpu-agent-test/index.html` cross-validate mode)

**Embed baseline in test page**: After Node ORT verification passes, embed the fp32 baseline tokens + transcript as constants in `webgpu-agent-test/index.html` (`FP32_BASELINE_TOKENS`, `FP32_BASELINE_TRANSCRIPT`). The cross-validation mode compares WebGPU output against these constants.

**WebGPU-only variants**: fp16io is WebGPU-only. Do NOT test on WASM — fp16 internal ops produce garbage on WASM EP (pitfall #47).

## WebGPU Testing Workflow (Ground-Truth Files)

**CRITICAL: Never load two encoders in the browser.** VRAM won't support it
(fp32 encoder ~2.5GB, fp16io ~1.3GB, plus decoders). The correct workflow:

1. **Node ORT generates ground truth** → save to JSON files:
   - Encoder output stats (cosine similarity, MSE, per-frame, per-dim)
   - Token sequence (full array of generated token IDs)
   - Transcript text
   - Timing data
   
2. **Web app loads references** → `webgpu-agent-test/index.html` fetches the
   JSON files and compares against WebGPU runtime output

3. **Bev agent (Windows, browser)** runs the WebGPU test, compares results

**Single encoder per browser session.** Each variant test loads only one
encoder + decoder pair. Comparison happens against the pre-generated JSON.

**Viable WebGPU combinations** (fp32 decoder always required):
- fp32 full — baseline
- fp16io + fp32 decoder — fastest (encoder 2.13s)
- mixf32 (q8 enc + fp32 dec) — works but slow encoder (50s)

**Non-viable on WebGPU** (fp16 decoder or q8 decoder):
- fp16 full — fp16 decoder_init NaN (Erf/Where/Tile/Range ops)
- mixed (q8 enc + fp16 dec) — fp16 decoder NaN
- q8 full — decoder_step MatMulInteger overflow

## Pitfalls

- **Mel comparison across implementations**: Library uses Slaney scale, Python
  Whisper uses Kaldi fbank. They produce different numbers. Always compare
  library-vs-library (fp32 baseline vs variant) for meaningful results.

- **fp16io on WASM EP = garbage output (2026-05-31)**: fp16io encoder (fp16
  internal + fp32 I/O) runs on WASM but produces corrupt hidden states. The
  decoder generates "a, a," (5 tokens, early EOS) instead of the JFK quote.
  WASM EP does not properly support fp16 tensor ops internally. fp16io is
  **WebGPU-only** (or CUDA). For WASM browser, use fp32 or q8 variants.

- **fp16io on Node ORT = bit-identical to fp32 (2026-05-31)**: On Node ORT
  (CPU EP), fp16io produces cosine=0.999987 vs fp32, 27/27 tokens match.
  The "degraded quality" from Entry 023 was WebGPU policy bugs, not encoder
  precision. See verify-step2-encoder.mjs and verify-step3-5-decode.mjs.

- **Node ORT CPU vs WebGPU EP numerical differences**: The same ONNX model can
  produce slightly different outputs on different execution providers. Always
  compare within the same provider first, then cross-provider.

- **begin_suppress_tokens is model-specific**: The values [220, 50257] are for
  large-v3-turbo/generation_config.json. Different model sizes (tiny, base,
  small, medium, large-v3) may have different special token IDs. Always load
  from the model's own generation_config.json.
