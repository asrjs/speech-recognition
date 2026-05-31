# ASR.js Whisper Engine — Session Handover

**Branch**: `main`
**Date**: 2026-05-31 (Flexo, evening)
**Machine**: P520 (WSL2, RTX 5060 Ti 8GB)

## Quick Recall

Load `asrjs-dev` skill → read this file → `docs/AGENT_TASKS.md`.

## What We've Been Working On (Since Yesterday)

### The Core Problem

We're porting Whisper large-v3-turbo to run in the browser via WebGPU. The model uses a **4-graph splitgraph format** (encoder + decoder_init + decoder_step + decoder_align) published at `ysdede/whisper-large-v3-turbo-onnx-4graph` on HuggingFace. We have 5 quantization variants: fp32, fp16, fp16io (fp16 internal + fp32 I/O), q8 (int8), and mixed (q8 encoder + fp32 decoder).

### Yesterday's Session (Entry 023) — WebGPU Pipeline Working

After ~150 tool calls debugging, we achieved the **first successful WebGPU Whisper pipeline**:
- **fp16io encoder** (fp16 internal + fp32 I/O) + **fp32 decoder**
- JFK transcript: "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
- 25.57s total (encoder 2.13s, decoder 3.32s on WebGPU)

**Root cause of all failures**: 6 policy bugs in the test page, NOT precision issues:
1. Wrong task token (translate instead of transcribe)
2. Wrong no_timestamps token ID
3. Missing `suppress_tokens` (~80 tokens)
4. Missing `begin_suppress_tokens [220, 50257]` → EOS fired early
5. Missing encoder KV preservation across decode steps
6. Custom decode loop instead of library

**Critical workflow change**: All decode logic now lives in the library (`src/models/whisper-seq2seq/`). Browser test pages are UI shells only — they sync code from the library.

### Today's Session — Verification + Browser Testing

#### ✅ fp16io Encoder Verification Complete (Node ORT)

Created two verification scripts and ran full pipeline:

**Step 2: Encoder output** (`tests/smoke/verify-step2-encoder.mjs`)
- Cosine similarity: **0.999987** (≥0.999 threshold)
- MSE: **4.9368e-6** (<0.01 threshold)
- Per-frame cosine: min 0.999943, mean 0.999982
- No NaN/Inf

**Steps 3-5: Full decode** (`tests/smoke/verify-step3-5-decode.mjs`)
- fp16io transcript: **IDENTICAL** to fp32
- 27/27 tokens match exactly
- No divergent tokens at any position

**Key finding**: fp16io encoder produces bit-identical output to fp32 on Node ORT. The "degraded transcript quality" seen in Entry 023's WebGPU test was from the 6 policy bugs, NOT from encoder precision. **fp16io quality tuning is NOT needed.**

#### ❌ Browser Testing — BLOCKED

Rewrote `webgpu-agent-test/index.html` as a comprehensive verification suite (all variants, all backends, cross-validation mode). Then hit multiple blockers:

| Issue | Impact | Workaround |
|-------|--------|------------|
| fp32 encoder 2.4GB | `Failed to fetch` — browser per-request limit ~1.5-2GB | Need IndexedDB cache + streaming |
| WASM fp16 ops | fp16io encoder produces garbage ("a, a,") on WASM | fp16io is WebGPU-only |
| WASM heap limit | ~1.5GB total — can't load encoder + decoder together | Sequential lifecycle (dispose between stages) |
| Headless browser no WebGPU | API exists but no GPU adapter in WSL2 | Need real Windows Chrome with GPU |

**Tested combinations**:
- fp32 on WebGPU → fetch limit
- fp16io on WASM → garbage output (fp16 ops unsupported)
- q8 on WASM → `std::bad_alloc` (heap limit)

#### 📌 Next Step: Library Pipeline + IndexedDB Cache

The library already has infrastructure for this:
- `src/io/cache.ts` → `IndexedDbAssetCache` (caches in `asrjs-cache-db` IndexedDB)
- `src/io/handles.ts` → `resolveAssetHandle` + HuggingFace URL builder + `BlobAssetHandle`
- `src/presets/whisper/manifest.ts` → `onnx-community/whisper-large-v3-turbo` preset (line 101)
- `browser-demo/src/shared/modelLoader.js` → example using `createSpeechPipeline({ cacheModels: true })`

**Two paths**:
1. **Library pipeline**: `createSpeechPipeline({ cacheModels: true })` with built-in `onnx-community/whisper-large-v3-turbo` preset — IndexedDB auto, but uses merged decoder format (not our 4-graph splitgraph)
2. **Low-level API**: `IndexedDbAssetCache` + `resolveAssetHandle` for our custom `ysdede/whisper-large-v3-turbo-onnx-4graph` models — needs manual wiring but supports our quantization variants

## Model Sizes

| Variant | Encoder .data | Decoder Init .data | Decoder Step .data | External Data |
|---------|--------------|-------------------|-------------------|---------------|
| fp32 | 2.4GB | 507MB | 254MB | All external |
| fp16 | 1.2GB | 254MB | 127MB | All external |
| fp16io | 1.2GB | 507MB | 254MB | All external |
| q8 | inline (616MB) | inline (228MB) | inline (415MB) | None (smallest) |
| mixed | inline (616MB) | 455MB | 127MB | Partial |

## Commits (this session)

```
9de3da3  feat: fp16io encoder verification complete (Steps 2-5)
01f78cf  docs: add verification suite and int8 task to AGENT_TASKS
81767e6  docs: session handover — fp16io verified, browser testing blocked
```

## Files Created/Modified

| File | Purpose |
|------|---------|
| `tests/smoke/verify-step2-encoder.mjs` | Encoder cosine similarity + MSE + per-frame + per-dim drift |
| `tests/smoke/verify-step3-5-decode.mjs` | Full decode + transcript match + token-by-token comparison |
| `/mnt/n/github/asrjs/webgpu-agent-test/index.html` | Full verification suite (all variants + backends + cross-validation) |
| `/mnt/n/github/asrjs/webgpu-agent-test/INSTRUCTIONS.md` | Browser agent test instructions |
| `docs/AGENT_TASKS.md` | Updated: completed tasks, remaining tasks |
| `docs/SESSION_HANDOVER.md` | Updated: this file |
| `asrjs-dev` skill | Pitfall #43 updated (fp16io verified identical) |

## Project Structure

```
speech-recognition/
  src/
    models/whisper-seq2seq/
      core.ts              — decode loops (greedy, beam, bestOf, patience)
      processor.ts         — WhisperTimestampLogitProcessor
      executor.ts          — ORT bridge, splitgraph, KV management
      enhanced-executor.ts — production pipeline
      generation-config.ts — config parsing
    audio/whisper-mel.ts   — mel spectrogram + padToFrames
    io/cache.ts            — IndexedDbAssetCache
    io/handles.ts          — resolveAssetHandle, HF URL builder
    presets/whisper/manifest.ts — model presets (line 101: large-v3-turbo)
  tests/smoke/
    verify-step1-mel.mjs              — Mel verification (MSE=0 ✅)
    verify-step2-encoder.mjs          — Encoder fp16io vs fp32 ✅
    verify-step3-5-decode.mjs         — Full decode comparison ✅
    whisper-large-v3-turbo-native.mjs — Native ORT smoke
    whisperx-runner.mjs               — WhisperX-compatible runner

webgpu-agent-test/  (Windows N: drive)
  index.html    — Verification suite (all variants + backends)
  models/       — fp32, fp16, fp16_iofp32, q8, mixed
  jfk2.en.wav   — Test audio
```

## Remaining Tasks

| # | Task | Effort | Notes |
|---|------|--------|-------|
| 1 | Library pipeline + IndexedDB for browser testing | Medium | `createSpeechPipeline({ cacheModels: true })` or low-level `IndexedDbAssetCache` |
| 2 | WebGPU verification (browser) | Medium | Needs IndexedDB cache first, then real Chrome + GPU |
| 3 | int8 (qf8) model generation for WASM | Medium | `onnxruntime.quantization.quantize_dynamic` |
| 4 | Batched encoder | Deferred | No CPU benefit |
| 5 | Framework adapters | Deferred | Separate packages |

## Verification Commands

```bash
cd ~/github/asrjs/speech-recognition

# Node ORT verification pipeline
node tests/smoke/verify-step1-mel.mjs            # Mel: MSE=0
node tests/smoke/verify-step2-encoder.mjs        # Encoder: cosine 0.999987
node tests/smoke/verify-step3-5-decode.mjs       # Decode: 27/27 tokens match

# Standard tests
npm run typecheck && npm run lint && npm test     # 601 tests
npm run build
node tests/smoke/quality-gates-smoke.mjs
node tests/smoke/whisper-e2e-pipeline-smoke.mjs
node tests/smoke/whisper-large-v3-turbo-native.mjs

# Browser testing (Windows Chrome)
cd /mnt/n/github/asrjs/webgpu-agent-test && npx serve -l 8765
# Open http://localhost:8765/ in Chrome with GPU
```
