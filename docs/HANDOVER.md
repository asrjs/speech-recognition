# Whisper WebGPU Optimization — Handover Report

**Date:** 2026-06-15
**Branch merged:** `perf/whisper-webgpu-decode` → `main`
**Commit on main:** `c1f50ce` (merge commit)
**Tests:** 633 passed, 4 skipped

---

## What We Did

Optimized Whisper large-v3-turbo ONNX 4-graph inference on WebGPU (ORT Web)
in the browser. Three optimizations deployed, benchmarked, and merged to main.

### Cumulative Results (RTX 5060 Ti, 29.9s JFK audio, fp16io-fp16, greedy)

| Stage | Before | After | Speedup |
|---|---|---|---|
| Preprocess (mel) | 237ms | 81ms | 2.9× |
| Encode | 1900ms | 277ms | 6.9× |
| Decode | 4000ms | 698ms | 5.7× |
| **Total** | **~6140ms** | **~1056ms** | **5.8×** |
| **RTFx** | **4.8×** | **27.6×** | |
| **Step P50** | **80ms** | **9.5ms** | **8.4×** |

All features preserved: timestamps, beam search, temperature sampling, logit processing.
All optimizations gated behind `experimentalGpuKvCache` flag.

---

## Three Deployed Optimizations

### 1. GPU KV Cache Bridge (`experimentalGpuKvCache`)
- **File:** `src/models/whisper-seq2seq/executor.ts`
- **Method:** `preferredOutputLocation` per-output map. Decoder KV tensors stay on GPU,
  fed directly between `decoder_init` and `decoder_step` without CPU round-trip.
  Logits intentionally kept on CPU (needed by timestamp processor).
- **Gating:** `experimentalGpuKvCache=true` + WebGPU backend. Greedy-only (rejects
  beam search, best_of, temperature > 0).

### 2. Stripped fp16 Encoder
- **Files:** `tools/whisper-onnx-export/strip_encoder_cast.py` (surgery script),
  `src/models/whisper-seq2seq/executor.ts` (async cast skip),
  `src/models/whisper-seq2seq/ort.ts` (encoder `preferredOutputLocation`)
- **Method:** Remove final `Cast(f16→f32)` from fp16_iofp32 encoder ONNX graph.
  Encoder outputs fp16 directly on GPU. Original fp16-input decoder_init works
  without modifications. Zero dtype casts in pipeline.
- **HF model:** `fp16_iofp32_fp16out/encoder_model.onnx` on
  `ysdede/whisper-large-v3-turbo-onnx-4graph`
- **Supersedes:** Cast-injected decoder_init approach (tool still exists:
  `inject_decoder_init_cast.py`)

### 3. Fast Mel Spectrogram (N_FFT=512)
- **File:** `src/audio/whisper-mel.ts`
- **Method:** Replace Bluestein FFT (N_FFT=400, non-power-of-2) with zero-padded
  512-point radix-2 FFT. 400-point Hann window centered in 512-point buffer.
  Matches parakeet.js approach.
- **Gating:** `fastFft` option (default true). Legacy Bluestein via `fastFft: false`.

---

## Key Architecture

```
Audio → Fast Mel (N_FFT=512, 81ms)
     → Stripped fp16 encoder (GPU output, 277ms)
     → Original fp16 decoder_init (GPU, no Cast needed)
     → GPU KV decoder_step (per-output location map)
     → Transcript (bit-identical to baseline)
```

**Zero dtype casts.** Pipeline is fp16 end-to-end on GPU.

---

## Not Yet Done (Next Session Priorities)

### High Priority
1. **GPU logit processing + ArgMax combined** — Move `WhisperTimestampLogitProcessor`
   suppression into ONNX graph. Apply mask before ArgMax. Output 4-byte token ID
   instead of 207KB logits per step. Expected: ~150ms decode reduction.
2. **Batched beam search** — Export decoder_step with batch=beam_size. Add
   `beam_indices` for GPU-side KV `Gather`. Expected: ~5× beam search speedup.
3. **Resolve decoder init regression** — Init time increased 69ms→195ms when
   encoder output went to GPU. Cross-session tensor handoff overhead. One-time
   cost per transcription but worth investigating.

### Medium Priority
4. **Static KV cache + graph capture** — Export decoder_step with fixed KV shapes.
   Enables `enableGraphCapture: true`.
5. **Parallel mel processing** — Chunk audio for parallel mel computation.
6. **Shared WebGPU device** — Create shared device with `shader-f16` feature.
   Currently reverted due to fp16 Cast failures (needs `requiredFeatures`).

---

## Tools Committed

| Script | Purpose |
|---|---|
| `tools/whisper-onnx-export/strip_encoder_cast.py` | Remove Cast(f16→f32) from encoder output |
| `tools/whisper-onnx-export/inject_decoder_init_cast.py` | Inject Cast(f32→f16) at decoder_init entry |
| `tools/whisper-onnx-export/append_argmax_to_decoder.py` | Append ArgMax+Cast to decoder_step output |
| `tools/whisper-onnx-export/verify_cast_parity.py` | Verify Cast model produces identical output |
| `tools/whisper-onnx-export/verify_argmax_parity.py` | Verify ArgMax model matches NumPy argmax |

---

## 14 Documented Pitfalls

See `docs/Whisper-Optimizations.md` § "Lessons Learned" for full details. Key ones:

1. `onnx.save(save_as_external_data=True)` **corrupts weights** — use plain `save()` + copy `.data`
2. **New ONNX outputs default to GPU-buffer** in per-output maps — must explicitly add to `'cpu'`
3. **Deployed `.onnx` filename must match internal `external_data.location`**
4. `preferredOutputLocation: 'gpu-buffer'` **eliminates GPU pipeline stalls** (the big encode win)
5. **GPU ArgMax alone is counterproductive** — only wins combined with GPU logit processing
6. **Stripped fp16 encoder > Cast-injected decoder_init** — cleaner architecture
7. **Always warm up before benchmarking** — first run includes model loading overhead
8. **Reuse browser tabs** — new tabs allocate fresh VRAM
9. **Vite `.vite` cache corruption** → kill node, `rm -rf .vite`, restart `--force`
10. **Vite on Windows needs PTY** for background processes

---

## Test Harness

- **App:** `N:\github\asrjs\webgpu-agent-test\` — Vite dev server on port 8765
- **Start:** `cd webgpu-agent-test && npx vite --host 0.0.0.0 --port 8765 --strictPort --force` (with `pty=true` on Windows)
- **Benchmark URL:** `http://localhost:8765/?auto=fp16io-fp16-webgpu&maxNewTokens=50&gpuKv=1`
- **Results:** `_results/*.json` + middleware at `GET /__test_results__`
- **Fix 504 errors:** Kill all node, `rm -rf node_modules/.vite`, restart `--force`
- **Models:** Local `public/models/` mirror + HF `ysdede/whisper-large-v3-turbo-onnx-4graph`
- **Custom ORT build:** Required for WebGPU (npm package is WASM-only). Built in WSL2 at `/home/steam/github/onnxruntime`

---

## Current Branch State

- **main:** `c1f50ce` — merged, pushed
- **perf/whisper-webgpu-decode:** 15 commits, merged into main
- **Backup tag:** `backup/whisper-fp16-webgpu-working-2026-06-14` (pre-optimization baseline)
- **Do NOT overwrite:** `ysdede/whisper-large-v3-turbo-onnx-4graph` (production models)
- **Experimental models go to:** separate repo like `ysdede/whisper-large-v3-turbo-onnx-4graph-webgpu-opt`

---

## Related Skills

- `porting-models-to-asrjs-webgpu` — Full fp16 porting reference, updated with optimizations
- `onnx-webgpu-dtype-bridge` — Cast node insertion for dtype mismatches
- `browser-inference-testing` — Automated browser-based ML inference testing
- `parakeet-js-dev` — parakeet.js, keet, asrjs ecosystem workflows
