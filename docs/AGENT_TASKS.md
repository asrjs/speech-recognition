# Agent Task Coordination

Branch: `feat/whisper-webgpu-artifact-boundaries`
Updated: 2026-08-25 (causal alignment artifact gate, FP16 export, and beam-cache benchmark)

## POST-RESTART VALIDATION (2026-08-23)

The workstation restart restored the expected NVIDIA Blackwell WebGPU state.
The browser harness used the real custom target
`ysdede/whisper-large-v3-turbo-onnx-4graph`. The remote preset names the
encoder artifact `fp16_iofp32/encoder_model.onnx`; the local harness uses its
optimized fp16-output copy, `fp16_iofp32_fp16out`, paired with the `fp16`
decoder. Warmed 30-second JFK measurements reached:

- `22.7617x` RTFx, `1328.07ms` total, `199.625ms` encoder, `49` decoder steps,
  `0` GPU tensor downloads.
- `22.2738x` RTFx with the profiling-only encoder GPU drain enabled; the drain
  moved `208.955ms` of queue work into the encoder metric without changing the
  production path.
- A 10-second measurement reached `11.7391x` RTFx. Longer audio is the useful
  throughput signal because fixed preprocess/encoder/decoder-init costs are
  amortized.
- An independent manual repeat on the optimized local variant reached
  `25.6993x` RTFx (`1175.81ms` total) on the 29.9043-second clip, with
  `183.49ms` encoder time, `49` GPU-KV steps, p50/p95 step time of
  `13.395/15.430ms`, and `0` downloads. Its 10-second repeat reached
  `13.856x`.

The earlier post-restart failure-state run around `8x` was not used as a code
baseline. Historical best-case runs around `26-28x` remain plausible, but
future optimization comparisons must use repeated warmed runs on this same
custom model and browser configuration.

## Context Recovery

**Primary skill**: Load `asrjs-dev` skill first
**Verification skill**: Load `whisper-model-verification-pipeline` for model porting verification
**Progress file**: `docs/Whisper-Optimizations.md`
**HF models**: `ysdede/whisper-large-v3-turbo-onnx-4graph` (original, fixed fp16) + v2 backup
**Local models (webgpu test)**: `N:\github\asrjs\webgpu-agent-test\public\models\` (fp32, fp16, fp16_iofp32, mixed, q8)
**Test page**: `N:\github\asrjs\webgpu-agent-test` (library-synced Vite harness)

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

### Whisper continuation boundary closure

- [x] Add `alignment_export.causal_self_attention` manifest metadata and make
  legacy split-graph artifacts warn and fall back to generated timestamp
  interpolation instead of claiming verified word alignment.
- [x] Share immutable parent KV-cache objects during stable beam expansion;
  decoder adapters retain ownership of cloning/repacking.
- [x] Make the exporter compatible with Transformers 4.41 legacy cache tuples,
  local model-snapshot paths, and true `--external-data always` output.
- [x] Re-export the local FP16 4-graph artifact at
  `N:\models\whisper-large-v3-turbo-causal-fp16-20260825-r2`; all graphs pass
  ONNX checker and CPU ORT load, and the exact external-data alignment graph
  passes the browser WebGPU timestamp harness with first word `2.42s`.
- [x] Re-measure warmed 29.904s English beam 2: stable `3.1242x` RTFx and
  batched `3.8094x` RTFx, with exact stable/batched text parity.

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

> **Direction change (2026-06-19):** Pause broad optimization work. Finish practical
> Whisper compatibility first, but keep the model target straight: the browser
> WebGPU implementation target is the custom splitgraph repo
> `ysdede/whisper-large-v3-turbo-onnx-4graph`. Merged-decoder
> `onnx-community/*_timestamped` presets are secondary compatibility paths.
> Optimization (batched beam, GPU-KV extensions, encoder graph capture) stays
> experimental until correctness is proven on fixtures.

### 1. Reference Decode Parity

Compare asr.js splitgraph decode output token-by-token against OpenAI Whisper /
HF Transformers / faster-whisper on a curated fixture set (English + Turkish).
Use the existing `WHISPER_REFERENCE_JSON` reproducibility harness with
`ysdede/whisper-large-v3-turbo-onnx-4graph` artifacts or local exports from the
same 4-graph layout.

- [x] Verify no-timestamps greedy decode against the locally cached HF
  `openai/whisper-large-v3-turbo` oracle on JFK: 31/31 normalized tokens and
  identical text for both Python mel and the TypeScript WAV frontend.
- Verify beam search tokens match reference for `numBeams=2..5`.
- Verify temperature sampling behavior matches Whisper/faster-whisper semantics.
- Verify `bestOf` only applies to nonzero-temperature sampling.

The reproducibility harness now supports separate encoder/decoder variant
directories and reads `[mel frames]` and `[encoder positions]` from ONNX graph
metadata. For large-v3-turbo, keep these distinct: 3000 input mel frames and
1500 encoder output positions. Use `--skip-onnx` in the Python generator when
the installed Python ORT cannot load the graph IR, then execute the graphs with
the Node/Vitest harness.

### 1a. Beam Semantics Revalidation

- [x] Keep finished EOS hypotheses outside active beam slots.
- [x] Use `round(numBeams * patience)` as the finished-candidate budget.
- [x] Route survivor KV caches with explicit parent indexes.
- [x] Run stable and batched execution through the same candidate lifecycle.
- [x] Match Whisper final ranking: default length normalization and Google NMT
  penalty for explicit `lengthPenalty`; use `0` only for raw-score ranking.
- [x] Keep beam expansion bounded by `beamSize + 1` candidates without a
  full-vocabulary log-softmax allocation per active beam.
- [x] Revalidate stable versus batched tokens in Windows Chrome/WebGPU for
  English beam 5, timestamped English beam 2, and Turkish auto beam 2; all
  matched exactly, with decoder calls reduced 245→49, 40→20, and 158→79.
- [ ] Add an HF/OpenAI beam reference fixture for `numBeams=2` and `5`.

### 2. True Language Auto-Detection

`language: "auto"` is wired for the splitgraph path and the merged-decoder
compatibility path. Keep validating it against the custom 4-graph target first;
use merged-decoder tests only to prevent regressions in secondary presets.

- Ensure failed auto-detection falls back to a real language token, never
  `<|auto|>` or a hard-coded Turkish fallback.
- [x] Browser validation with a non-English splitgraph fixture: Turkish TDK 18s
  clip, `language=auto`, detected `tr`, GPU-KV greedy, zero GPU downloads.

### 3. Quality Gates + Temperature Fallback

The wrapper exists but needs fixture-based validation that it actually rejects
hallucinations and recovers with higher temperature.

- [x] Add fixture smoke tests for compression-ratio and logprob rejection.
- [x] Verify retry temperatures are passed through correctly in single-chunk and
  VAD-chunk paths.
- [x] Verify caller `onTokenLogits` survives wrapper collection.
- [x] Replace the hard-coded no-speech approximation in the Whisper runtime.
  The gate now receives raw decoder-init logits from the SOT position before
  suppression and resolves the token from generation config or the tokenizer.
- [x] Preserve raw-init quality context through temperature fallback and keep
  the generic `50362` behavior for direct gate callers.
- [x] Define selected-beam logprob/entropy metrics without retaining every
  full-vocabulary tensor for every hypothesis; add fixture validation for
  compression/logprob rejection and temperature recovery.

### 4. Word Timestamp Parity

- [x] Emit word timestamps on the WebGPU splitgraph greedy path. If decoder-align
  is missing or empty, interpolate from timestamp tokens (Whisper fallback).
  Browser 10s JFK produced 17 timed words with GPU-KV still on GPU.
- [x] Lazy-load `decoder_align` when word timestamps are requested, copy encoder
  states to CPU for the align session, and DTW only generated text-token rows
  (skip timestamp/special tokens). Interpolation remains the fallback.
- [x] Align each timestamp-token span against only that span's encoder frames,
  crop 30s padding to audio duration, and spread identical DTW jumps so tokens
  are not zero-duration. Turbo often emits a single 0–10s pair, so some internal
  boundaries still need faster-whisper / WhisperX (wav2vec2) comparison.
- [x] Clip DTW outlier word durations using OpenAI Whisper's median*2 cap, and
  add optional `wordAligner` (Wav2Vec2 CTC Viterbi) as a WhisperX-style refine
  pass after decode. GPU-KV greedy is unchanged unless an aligner is provided.
- [x] Compare DTW/attention word timestamps against faster-whisper on the JFK
  reference fixture; the first-word anchor now matches within normal DTW
  variation instead of starting at zero.
- [x] Fix the systematic leading-boundary error in the runtime postprocessor
  and alignment export.
- [x] Re-export and validate a complete local FP16 precision variant with
  causal alignment and external-data metadata.
- [x] Keep opt-in WebGPU graph-capture probes recoverable: retry normal ORT
  session creation when partitioning rejects capture and emit a warning; a
  headless Chrome run completed with exact transcript behavior and zero GPU
  downloads.
- [ ] Re-export the published precision variants and validate the remote
  preset; no model-hosting update has been performed.
- Validate that word timestamps work with both splitgraph and merged-decoder paths.
- Timestamp-token processing now includes `<|notimestamps|>` suppression and
  the aggregate timestamp-probability rule; retain focused tests for both.

### 5. WhisperX Runner End-to-End Validation

- [x] Fix runner CLI underscore flags (`--beam_size`, `--no-word_timestamps`,
  `--language auto`, `--output_format`) and load Wav2Vec2 only after language
  detection so Turkish uses the XLS-R aligner. DTW words now use span-limited
  alignment plus duration clipping; Wav2Vec2 refines those times when present.
- [x] Harden the runner's temperature attempts with raw decoder-init no-speech
  logits and selected-sequence quality traces, and make large-vocabulary
  sampling safe without argument spreading. CLI regressions cover the
  `--model-dir` spelling and the 51,865-token Whisper vocabulary.
- [x] Run `tests/smoke/whisperx-runner.mjs` on real speech files (English +
  Turkish). The runner now uses Windows-safe FFmpeg redirection and file URLs
  for dynamic ESM imports, and preserves per-beam KV dimensions for stable
  beam. English OGG conversion and Turkish WAV both complete with beam 2.
- [x] Verify `--beam_size`, `--language auto`, `--word_timestamps`, and
  `--output_format` produce sane output on those files. Temperature remains
  covered by the deterministic quality-gate tests; faster-whisper timestamp
  comparison is recorded as the remaining alignment-quality boundary.

### 6. Deprioritized / Experimental (do not prioritize)

- Batched beam decode: implemented as opt-in `experimentalBatchedBeam`. Keep stable
  CPU-KV beam as oracle. Only promote after more variants/fixtures prove parity.
- GPU-KV beam support: not feasible without batched beam decode graph changes.
- Encoder graph capture: fails on Reshape/Shape ops; revisit only after ORT updates.
- Decoder graph capture: remains diagnostic-only; failed partitioning now falls
  back to the regular session with a recoverable warning.
- Masked GPU ArgMax: requires alternate model artifact; not core compatibility.
- Batched encoder: no CPU benefit.
- FireRedASR2-AED and Qwen3-ASR-0.6B: artifact-backed porting boundary is
  tracked in `docs/handoffs/asr-candidate-boundaries-2026-08-25.md`.
- Framework adapters: separate packages.
- `condition_on_previous_text`, hotwords, numeral suppression: skip unless a fixture
  proves they help.

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
