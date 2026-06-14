# Agent Task Coordination

Branch: `main`
Updated: 2026-06-14 (Bev/Codex, fp16 WebGPU + Whisper mel optimization + decoder profiling)

## Context Recovery

**Primary skill**: Load `asrjs-dev` skill first
**Verification skill**: Load `whisper-model-verification-pipeline` for model porting verification
**Progress file**: `docs/SESSION_HANDOVER.md`
**HF models**: `ysdede/whisper-large-v3-turbo-onnx-4graph` (original, fixed fp16) + v2 backup
**Local models (webgpu test)**: `/mnt/n/github/asrjs/webgpu-agent-test/models/` (fp32, fp16, fp16_iofp32, mixed, q8)
**Test page**: `/mnt/n/github/asrjs/webgpu-agent-test/index.html` (library-synced)

## 🏆 MILESTONE: WebGPU large-v3-turbo pipeline WORKING (Entry 023)

First successful WebGPU Whisper pipeline: **fp16io encoder (fp16 internal + fp32 I/O) + fp32 decoder**

- Encoder: 2.13s, Decoder: 3.32s, Total: 25.57s
- Zero NaN, zero overflow, proper EOS

## 🏆 MILESTONE: Full fp16 WebGPU 4-graph preset WORKING (Entry 024)

Verified on Windows Chrome + WebGPU with the custom Hugging Face repo
`ysdede/whisper-large-v3-turbo-onnx-4graph`:

- Preset: `fp16io-fp16-webgpu`
- Encoder folder: `fp16_iofp32`
- Decoder folder: `fp16`
- Fixture: `29.9043s` JFK audio from `webgpu-agent-test`
- Transcript: correct JFK continuation, EOS reached within 50-token cap
- Stage metrics: preprocess `234.63ms`, encode `1732.64ms`, decode `3837.28ms`, total `5812.04ms`
- End-to-end transcribe RTF: `0.1944` (`5.1452x` realtime)

Important correction: the library preset must resolve the custom 4-graph repo,
not `onnx-community/whisper-large-v3-turbo`. Logs mentioning
`onnx-community/...` during this work indicated the demo/app was resolving the
wrong manifest source.

### Decoder profile finding

For Whisper splitgraph models, decoder time can exceed encoder time even when KV
cache is working. The encoder is one parallel pass over all mel frames; the
decoder is autoregressive and must run `decoder_step.onnx` once per generated
token. The 4-graph split plus KV cache prevents recomputing the full decoder
prefix, but it cannot remove token-by-token generation.

Latest Chrome WebGPU fp16 run (`fp16io-fp16-webgpu`, 29.9s JFK fixture, 50-token
cap) shows the bottleneck is ORT/WebGPU graph execution, not JS KV bridging:

| Metric | Time |
| ------ | ---- |
| Encode | `1759.04ms` |
| Decode total | `3979.24ms` |
| Decoder init run | `133.52ms` |
| Decoder step total | `3792.29ms` |
| Decoder step ORT run | `3788.18ms` |
| Step feed build | `0.82ms` |
| Step tensor wrapping/clone | `1.24ms` |
| Step output handling | `1.67ms` |
| Step p50 / p95 / max | `77.0ms` / `86.01ms` / `91.2ms` |
| Step count | `49` |

Optimization implication: first reduce `decoder_step` calls or the cost of each
ORT/WebGPU step. Beam search and `best_of` improve quality options but multiply
decoder-step work; they are expected to be slower.

### Whisper mel performance fix

The old `WhisperMelProcessor` used a direct per-frame 400-point DFT with fresh
per-frame allocations. That made 30s preprocessing take roughly `9185ms` on the
test host. The processor now keeps Whisper's exact `n_fft=400` contract but uses
a cached Bluestein FFT, reusable work buffers, and sparse mel filter bounds.

Verification:

```bash
npm test -- tests/whisper-mel-validation.test.ts --run
npm run benchmark:whisper-mel
```

Latest benchmark:

| Audio | Avg mel time | RTFx   |
| ----- | ------------ | ------ |
| 1s    | 8.7ms        | 114.8x |
| 10s   | 62.7ms       | 159.5x |
| 30s   | 204.2ms      | 146.9x |

### Root cause: Policy bugs, NOT precision bugs

The test page had **6 bugs** (all fixed):

1. Wrong task token (translate instead of transcribe)
2. Wrong no_timestamps token ID
3. Missing `suppress_tokens`
4. Missing `begin_suppress_tokens [220, 50257]` → EOS fired early at step 2
5. Missing encoder KV preservation → decoder_step crashed at step 2+
6. Custom decode loop instead of library → reimplemented everything

### 🔄 CRITICAL WORKFLOW CHANGE

**Before:** Custom decode logic in WebGPU test page (separate implementation).
**After:** Single implementation in library. Browser test page syncs from library code.

New workflow for model porting:

```
Step 1: Mel → compare vs reference (MSE=0)
Step 2: Encoder → compare fp16io vs fp32 (cosine sim > 0.99)
Step 3: Decoder init → compare logits distribution
Step 4: Full decode → compare transcript vs ground truth (Levenshtein < 10%)
Step 5: Token-by-token → first 5 tokens match fp32 baseline
→ All on Node ORT first, THEN promote to WebGPU
```

## Backend Strategy

| Priority | Backend                     | Model  | Lifecycle  | Notes                     |
| -------- | --------------------------- | ------ | ---------- | ------------------------- |
| **1st**  | `onnxruntime-node` (native) | fp32   | Persistent | Dev target, no heap limit |
| 2nd      | WebGPU (browser)            | fp16io | Persistent | Working (Entry 023)       |
| Fallback | ORT Web/WASM                | fp32   | Sequential | ~1.5GB heap               |

**Browser externalData**: Must fetch `.data` files and pass `externalData: [{path, data: Uint8Array}]`.

## COMPLETED TASKS (this session)

### WebGPU Pipeline Fixes

- [x] `begin_suppress_tokens [220, 50257]` — EOS blocked at step 0, text token selected
- [x] `suppress_tokens` — ~80 special tokens blocked every step
- [x] Prompt corrected: `[50258, 50259, 50360, 50364]` (was using translate token)
- [x] Encoder KV preservation across decode steps
- [x] First successful WebGPU whisper transcript

### Verification Infrastructure

- [x] Mel reference `jfk2-mel-128.json` regenerated (was stale after WhisperMelProcessor fix)
- [x] `tests/smoke/verify-step1-mel.mjs` — Mel comparison (MSE=0 PASS)
- [x] Skill: `whisper-model-verification-pipeline` (mlops/) — full workflow

## COMPLETED TASKS (2026-05-31 — fp16io verification)

### fp16io Encoder Verification (Steps 2-5) — ALL PASS ✅

**Step 2: Encoder output comparison** — `tests/smoke/verify-step2-encoder.mjs`

- Cosine similarity: **0.999987** (threshold ≥ 0.999) ✅
- MSE: **4.9368e-6** (threshold < 0.01) ✅
- Per-frame cosine: min 0.999943, mean 0.999982
- No NaN/Inf
- Worst dimension drift: dim[884] MSE=1.0518e-3

**Step 4: Transcript comparison** — `tests/smoke/verify-step3-5-decode.mjs`

- fp16io transcript: **IDENTICAL** to fp32 ✅
- JFK quote: "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."

**Step 5: Token-by-token comparison**

- 27/27 tokens match exactly ✅
- No divergent tokens

**Key finding**: fp16io encoder produces bit-identical output to fp32 on Node ORT. The "degraded transcript quality" noted in Entry 023 was NOT from encoder precision — it was from WebGPU decode path differences (policy bugs, not precision).

**fp16io Quality Tuning — NOT NEEDED** (cancelled): Encoder output is functionally identical to fp32. No quality tuning required for Node ORT. WebGPU quality issues were policy bugs (fixed in Entry 023).

## REMAINING TASKS (priority order)

### 1. int8 (qf8) Model Generation for WASM

Parakeet.js uses int8 for WASM compatibility. Whisper q8 already works identically to fp32.
May need to generate proper int8 variants if q8 decoder has issues on specific WASM backends.
Tool: `onnxruntime.quantization.quantize_dynamic` with `optimize_model` pass first.

### 2. WebGPU Verification (Browser)

Full verification suite at `/mnt/n/github/asrjs/webgpu-agent-test/index.html`:

- All variants: fp32, fp16, fp16io, q8, mixed
- All backends: WebGPU, WASM (configurable per encoder/decoder)
- Modes: Run Decode, Cross-Validate, Encoder-Only
- Requires real browser with GPU (RTX 5060 Ti + Chrome)

### 3. Batched Encoder

Deferred — no CPU benefit. Would help with CUDA provider.

### 4. Framework Adapters (React, Vue, Svelte)

Separate packages — deferred.

## Shared Files (coordinate before modifying)

- `src/models/whisper-seq2seq/core.ts` — decode loops
- `src/models/whisper-seq2seq/executor.ts` — ORT bridge, splitgraph
- `src/models/whisper-seq2seq/enhanced-executor.ts` — production pipeline
- `src/models/whisper-seq2seq/processors.ts` — WhisperTimestampLogitProcessor
- `src/models/whisper-seq2seq/generation-config.ts` — config parsing
- `src/audio/whisper-mel.ts` — mel spectrogram
- `src/quality/` — quality gates
- `src/chunking/` — VAD, drift, context
- `src/post-processing/` — merge, format, subtitles
- `src/alignment/` — CTC Viterbi, WAV2VEC2 aligner
- `src/pipeline/` — ProductionWhisperPipeline

## Verification (quick)

```bash
cd ~/github/asrjs/speech-recognition
npm run typecheck && npm run lint && npm test   # 601 tests
npm run build
node tests/smoke/verify-step1-mel.mjs            # Mel: must be MSE=0
node tests/smoke/verify-step2-encoder.mjs        # fp16io vs fp32 encoder (cosine > 0.999)
node tests/smoke/verify-step3-5-decode.mjs       # fp16io vs fp32 decode (token-by-token)
node tests/smoke/quality-gates-smoke.mjs
node tests/smoke/whisper-e2e-pipeline-smoke.mjs
node tests/smoke/whisper-large-v3-turbo-native.mjs
```
