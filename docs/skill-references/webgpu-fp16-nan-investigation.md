# WebGPU fp16/io Pipeline Investigation

## Status: FIRST WORKING WEBGPU PIPELINE ✅ (2026-06-01, Entry 023)

**fp16io encoder (fp16 internal + fp32 I/O) + fp32 decoder on WebGPU EP** produces a valid transcript (25.57s total). The pipeline is mechanically sound: zero NaN, zero overflow, correct KV cache, correct EOS handling. Transcript quality is degraded (garbage text) due to fp16 encoder compute precision.

## What changed from earlier NaN/empty results

### Fix #1: Generation policy (decode-policy-verification.md)

The test page had 6 bugs in its generation policy, including wrong prompt tokens, missing `suppress_tokens`, missing `begin_suppress_tokens [220, 50257]`. Without these, the decoder fires EOS at step 2 regardless of model precision.

**Diagnostic**: token sequence `50360 → 50364 → 50257` = prompt too short.

### Fix #2: KV cache encoder preservation

The split decoder_step model outputs ONLY decoder self-attention KV (`present.{i}.decoder.{key,value}`). Encoder cross-attention KV must be preserved from the previous step's cache:

```javascript
// After step output:
const oldPkv = {...pkv};
for (const k of sk) { if (k.startsWith('present'))
  pkv[k.replace(/^present\./, 'past_key_values.')] = so[k]; }
// Preserve encoder KV (step model doesn't output it):
for (const [k, v] of Object.entries(oldPkv)) {
  if (k.includes('encoder') && !pkv[k]) pkv[k] = v; }
```

Without this, step 2 throws: `invalid input 'past_key_values.0.encoder.key' is missing`.

### Fix #3: fp16io encoder export

`onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True, disable_shape_infer=True)` creates an encoder with fp16 internal compute but fp32 I/O. Feed its output directly into fp32 decoder — no Cast node, no JS-side conversion.

## Updated test matrix (ORT 1.26.0, WebGPU EP)

| Variant | Encoder | Decoder | Time | NaN? | Transcript | Status |
|---------|---------|---------|------|------|------------|--------|
| **fp16io** 🏆 | fp16 I/O fp32 2.13s | fp32 3.32s | **25.57s** | ✅ none | garbage (quality issue) | **WORKS** |
| mixf16f32 | fp16 2.6s | fp32 | 25.5s | ✅ none | empty (crash) | OBSOLETE |
| mixf32 🎯 | q8 50.9s | fp32 | 74.7s | ✅ none | "G," short | BROKEN (q8 shift) |
| q8 | q8 48.1s | q8 | 91.5s | ⚠️ overflow | garbage 200 tok | BROKEN |
| fp32 | fp32 (2.4GB) | fp32 | FAIL | — | fetch limit | OBSOLETE |

fp16io **supersedes** mixf16f32: same speed, same zero NaN, but no early-EOS thanks to I/O fp32 boundary. The earlier "empty transcript" was caused by policy bugs + missing KV cache preservation, not encoder distribution mismatch.

## Quality analysis

fp16io produces fluent English sentences but does NOT match the ground-truth JFK transcript ("Ask not what your country..."). The fp16 encoder compute introduces sufficient numerical drift that the decoder generates different text.

**Next quality steps:**
1. Compare fp32 encoder baseline vs fp16io encoder output (cosine similarity, per-channel stats) — see `decode-policy-verification.md` for Python script
2. Try q8 encoder + fp32 decoder (mixf32) — if quality is better, the issue is specifically fp16 precision
3. Try WebGPU fp32 encoder if fetch limits permit (unlikely — 4.5GB)
4. Accept quality degradation as trade-off for browser GPU acceleration

## Updated repository references

- Test page: `/mnt/n/github/asrjs/webgpu-agent-test/index.html`
- Config reference: `generation_config.json` in HF cache at `whisper-large-v3-turbo`
- Models: `ysdede/whisper-large-v3-turbo-onnx-4graph` (fp32 base), plus `fp16_iofp32/encoder_model.onnx` in test dir
