# Agent Task Coordination

Branch: `perf/whisper-webgpu-decode`
Updated: 2026-06-14 (Codex, WebGPU GPU-KV optimization plan refresh)

## Context Recovery

**Primary skill**: Load `asrjs-dev` skill first
**Verification skill**: Load `whisper-model-verification-pipeline` for model porting verification
**Progress file**: `docs/Whisper-Optimizations.md`
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

Optimization implication after the GPU-KV work: first preserve the measured
greedy fast path, then optimize encoder time and beam-specific decode separately.
Beam search and `best_of` improve quality options but multiply decoder-step
work; they are expected to be slower until batched beam decode exists.

### WebGPU GPU-KV fast path

The active WebGPU speedup is `experimentalGpuKvCache`. It keeps decoder KV
tensors on GPU between `decoder_init` and `decoder_step`, while keeping logits
on CPU through per-output placement so Whisper token suppression remains exact.

Latest known Chrome WebGPU run (`fp16io-fp16-webgpu`, 29.9s JFK fixture,
50-token cap, `experimentalGpuKvCache=true`):

| Metric | Time |
| ------ | ---- |
| Preprocess | `335.83ms` |
| Encode | `1980.085ms` |
| Decode total | `771.56ms` |
| Decoder init run | `72.08ms` |
| Decoder step run | `685.14ms` |
| Step p50 / p95 / max | `11.55ms` / `30.615ms` / `53.42ms` |
| Logit processing | `2.025ms` |
| Output handling | `1.525ms` |
| GPU tensor downloads | `0` |
| Total | `3095.505ms` |
| RTFx | `9.6606x` |

Do not use the older arithmetic that assumes a 75ms full-logit GPU download on
this path. Current per-output placement reports zero GPU tensor downloads.

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
| 1s    | 9.8ms        | 101.6x |
| 10s   | 58.1ms       | 172.1x |
| 30s   | 179.2ms      | 167.4x |

### Greedy score cleanup

The generic greedy splitgraph loop no longer computes cumulative
log-probability unless `bestOf` ranking requests it. This keeps ordinary greedy
decode token-equivalent while removing one full-vocabulary pass per token on
the stable CPU/WASM-style path.

Local helper benchmark, 50 tokens and 51,865 vocab entries: `49.18ms` average
before, `2.54ms` average after. This is a helper-level CPU result, not a
browser WebGPU GPU-KV end-to-end claim.

### ArgMax experiment status

GPU-side ArgMax is a future isolated experiment, not a committed optimization.
A raw `ArgMax(logits)` graph output is unsafe because it bypasses
`suppress_tokens`, `begin_suppress_tokens`, no-timestamp suppression, and
timestamp-state rules.

Only attempt this behind a new local/HF alternate artifact, and only for the
greedy no-timestamps path at first:

- `temperature=0`
- `numBeams=1`
- `bestOf=1`
- `noTimestamps=true`
- no `onTokenLogits` callback

The first safe graph variant needs a static suppression mask and must request
only `next_token_id` plus `present.*` outputs via ORT `fetches` for the A/B.
Reject it on any token mismatch or WebGPU memory growth.

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

### 1. Preserve and broaden WebGPU GPU-KV validation

Run the working `fp16io-fp16-webgpu` fast path on more fixtures up to 30s:

- fixed JFK 29.9s fixture
- shorter 10s fixture
- at least one non-JFK speech sample
- `maxNewTokens=50` and a longer cap for EOS behavior

Keep token IDs, transcript prefix, RTFx, p50/p95 step timing, and tensor
location metrics in results.

### 2. Encoder graph-capture A/B

The source flag `experimentalWebGpuEncoderGraphCapture` is wired, but the
Chrome automation retry was blocked by the Chrome extension not accepting
automation after a fresh-window retry. Reinstall the Chrome plugin/extension
from the Codex plugin UI before trying automation again.

Manual A/B URLs when the demo is running:

```text
http://localhost:8765/?auto=fp16io-fp16-webgpu&maxNewTokens=50&gpuKv=1
http://localhost:8765/?auto=fp16io-fp16-webgpu&maxNewTokens=50&gpuKv=1&encoderGraphCapture=1
```

Keep only if session creation succeeds, tokens match, and `encodeMs` improves
on the same fixture.

### 3. Masked GPU ArgMax alternate artifact

Create a separate local/HF model artifact before touching graph outputs. Do not
overwrite `ysdede/whisper-large-v3-turbo-onnx-4graph`.

The experiment must use a masked ArgMax output and ORT `fetches`; raw ArgMax is
not semantically equivalent to the library decoder.

### 4. Batched beam decode

Beam search is implemented but still uses the stable CPU/WASM-style KV bridge.
The measured WebGPU fast path is greedy-only. For beam speed, design a batched
decoder step plus KV reorder path instead of multiplying ORT calls per beam.

### 5. int8 (q8) Model Generation for WASM

Parakeet.js uses int8 for WASM compatibility. Whisper q8 already works identically to fp32.
May need to generate proper int8 variants if q8 decoder has issues on specific WASM backends.
Tool: `onnxruntime.quantization.quantize_dynamic` with `optimize_model` pass first.

### 6. WebGPU Verification (Browser)

Full verification suite at `/mnt/n/github/asrjs/webgpu-agent-test/index.html`:

- All variants: fp32, fp16, fp16io, q8, mixed
- All backends: WebGPU, WASM (configurable per encoder/decoder)
- Modes: Run Decode, Cross-Validate, Encoder-Only
- Requires real browser with GPU (RTX 5060 Ti + Chrome)

### 7. Batched Encoder

Deferred — no CPU benefit. Would help with CUDA provider.

### 8. Framework Adapters (React, Vue, Svelte)

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
