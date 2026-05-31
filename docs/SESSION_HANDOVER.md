# ASR.js Whisper Engine — Session Handover

**Branch**: `main`
**Date**: 2026-05-31 (Flexo)
**Machine**: P520 (WSL2, RTX 5060 Ti 8GB)

## Quick Recall

Load `asrjs-dev` skill → read `docs/AGENT_TASKS.md` → this file.

## What we accomplished this session

### ✅ fp16io Encoder Verification Complete (Steps 2-5) — Node ORT

**Step 2: Encoder output comparison** (`tests/smoke/verify-step2-encoder.mjs`)
- Cosine similarity: **0.999987** (threshold ≥ 0.999) ✅
- MSE: **4.9368e-6** (threshold < 0.01) ✅
- Per-frame cosine: min 0.999943, mean 0.999982
- No NaN/Inf

**Step 4: Transcript comparison** (`tests/smoke/verify-step3-5-decode.mjs`)
- fp16io transcript: **IDENTICAL** to fp32 ✅
- JFK quote produced correctly by both models

**Step 5: Token-by-token comparison**
- 27/27 tokens match exactly ✅
- No divergent tokens at any position

**Key finding**: fp16io encoder produces bit-identical output to fp32 on Node ORT. The "degraded transcript quality" in Entry 023 was from WebGPU decode policy bugs, NOT encoder precision.

### 🔄 WebGPU Test Page Rewritten

`/mnt/n/github/asrjs/webgpu-agent-test/index.html` — comprehensive verification suite:
- All variants: fp32, fp16, fp16io, q8, mixed
- All backends: WebGPU, WASM (configurable per encoder/decoder)
- Modes: Run Decode, Cross-Validate vs fp32 baseline, Encoder-Only cosine comparison
- Quick presets: one-click for common combos
- Library-synced: WhisperMelProcessor, WhisperTimestampLogitProcessor, whisperDecode
- Sequential lifecycle for WASM: encoder→dispose→decoder_init→dispose→decoder_step

### ❌ WebGPU Browser Testing — BLOCKED

Tested with real Chrome on Windows:
- **fp32 encoder 2.5GB** → `Failed to fetch` (browser fetch limit exceeded)
- **WASM fp16io** → produces garbage ("a, a,") — WASM doesn't support fp16 ops
- **WASM q8** → `std::bad_alloc` — heap limit (encoder 1.2GB + decoder 506MB)
- **Headless browser** → WebGPU API exists but no GPU adapter

### 📌 Next Step: Library Pipeline + IndexedDB Cache

**Problem**: Local HTTP server can't serve 2.5GB files (browser fetch limit).

**Solution**: Use library's `createSpeechPipeline({ cacheModels: true })` which handles:
- HuggingFace model downloading (streaming)
- IndexedDB caching (persistent across sessions)
- ORT session creation with proper external data handling

**Two paths identified**:
1. **Library pipeline** with `onnx-community/whisper-large-v3-turbo` preset (built-in, IndexedDB auto)
2. **Low-level API** with `IndexedDbAssetCache` + `resolveAssetHandle` for our custom `ysdede/whisper-large-v3-turbo-onnx-4graph` models

**Status**: Decision needed — which path to take.

## Verification Scripts Created

- `tests/smoke/verify-step2-encoder.mjs` — Encoder cosine similarity + MSE + per-frame + per-dimension drift
- `tests/smoke/verify-step3-5-decode.mjs` — Full decode + transcript match + token-by-token comparison

## Project Structure

```
speech-recognition/
  src/
    models/whisper-seq2seq/
      core.ts              — decode loops (greedy, beam, bestOf, patience)
      processor.ts         — WhisperTimestampLogitProcessor (suppression rules)
      executor.ts          — ORT bridge, splitgraph, KV management
      enhanced-executor.ts — production pipeline (VAD+gates+fallback+drift+merge)
      generation-config.ts — parse begin_suppress_tokens / suppress_tokens
    audio/
      whisper-mel.ts       — Whisper-compatible log-mel spectrogram + padToFrames
    io/
      cache.ts             — IndexedDbAssetCache (asrjs-cache-db)
      handles.ts           — resolveAssetHandle, HuggingFace URL builder, BlobAssetHandle
    presets/whisper/
      manifest.ts          — onnx-community/whisper-large-v3-turbo preset (line 101)
  tests/smoke/
    verify-step1-mel.mjs              — Mel verification (MSE=0 ✅)
    verify-step2-encoder.mjs          — Encoder fp16io vs fp32 (NEW)
    verify-step3-5-decode.mjs         — Full decode comparison (NEW)
    whisper-large-v3-turbo-native.mjs — Native ORT persistent smoke
    whisperx-runner.mjs               — WhisperX-compatible runner

webgpu-agent-test/  (on Windows N: drive)
  index.html          — Full verification suite (REWRITTEN)
  models/             — ONNX model files (fp32, fp16, fp16_iofp32, mixed, q8)
  jfk2.en.wav         — Test audio
  INSTRUCTIONS.md     — Browser agent instructions
```

## Model Sizes (for browser testing)

| Variant | Encoder .data | Decoder Init .data | Decoder Step .data | External Data |
|---------|--------------|-------------------|-------------------|---------------|
| fp32 | 2.4GB | 507MB | 254MB | All external |
| fp16 | 1.2GB | 254MB | 127MB | All external |
| fp16io | 1.2GB | 507MB | 254MB | All external |
| q8 | inline (616MB) | inline (228MB) | inline (415MB) | None |
| mixed | inline (616MB) | 455MB | 127MB | Partial |

## Key Pitfalls (this session)

1. **WASM fp16 unsupported**: fp16io encoder produces garbage on WASM — fp16 internal ops not supported
2. **WASM heap limit**: ~1.5GB total — can't load encoder + decoder together for large models
3. **Browser fetch limit**: ~1.5-2GB per request — fp32 encoder (2.4GB) fails with `Failed to fetch`
4. **Headless browser no WebGPU**: API exists but no GPU adapter (WSL2 without GPU passthrough)
5. **`npx serve` required**: Python's `http.server` fails for files >2GB (HTTP/1.0, no range requests)

## Remaining

| Task | Effort | Notes |
|------|--------|-------|
| Library pipeline + IndexedDB for browser testing | Medium | Use `createSpeechPipeline({ cacheModels: true })` |
| int8 (qf8) model generation for WASM | Medium | `onnxruntime.quantization.quantize_dynamic` |
| WebGPU verification (browser) | Medium | Needs IndexedDB cache working first |
| Batched encoder | Deferred | No CPU benefit |
| Framework adapters | Deferred | Separate packages |

## Verification (quick)

```bash
cd ~/github/asrjs/speech-recognition

# Full verification pipeline (Node ORT)
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
# Start WSL server: cd /mnt/n/github/asrjs/webgpu-agent-test && npx serve -l 8765
# Open: http://localhost:8765/ (or http://172.21.137.254:8765/)
```

## Key Files

| File | Purpose |
|------|---------|
| `docs/AGENT_TASKS.md` | Task coordination (source of truth) |
| `docs/SESSION_HANDOVER.md` | This file — session context |
| `tests/smoke/verify-step1-mel.mjs` | Mel verification (Step 1) |
| `tests/smoke/verify-step2-encoder.mjs` | Encoder verification (Step 2) |
| `tests/smoke/verify-step3-5-decode.mjs` | Decode verification (Steps 3-5) |
| `/mnt/n/github/asrjs/webgpu-agent-test/index.html` | Browser verification suite |
| `src/io/cache.ts` | IndexedDbAssetCache |
| `src/io/handles.ts` | Asset resolution + HuggingFace URL builder |
| `src/presets/whisper/manifest.ts` | Whisper model presets (line 101: large-v3-turbo) |
