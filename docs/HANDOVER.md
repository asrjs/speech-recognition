# Whisper WebGPU Optimization — Handover Report

**Date:** 2026-06-19 (updated)
**Active branch:** `main`
**Agent:** Bev (P520, Windows 11, RTX 5060 Ti)
**Hermes:** v0.14.0 (Windows native)

---

## Current Production State

4-graph splitgraph Whisper WebGPU pipeline:
```
encoder.onnx (fp16_iofp32_fp16out) → decoder_init.onnx → decoder_step.onnx → decoder_align.onnx
```

**Flags:** `experimentalGpuKvCache=true` (others off by default)
- `encoderGpuDrain`: PROFILING ONLY — forces GPU drain after encoder, adds ~18ms overhead. Off in production.
- `encoderGpuFlush`: DIAGNOSTIC (Edge B2) — same as encoderGpuDrain, kept for backward compat.

**Performance (RTX 5060 Ti, English JFK 29.9s, 50 tokens, GPU-warm, fp16 GPU KV):**

| Phase | Time | % |
|-------|------|---|
| Audio prep | 81ms | 7% |
| Mel preprocess | 97ms | 8.5% |
| Encoder (run) | 185ms | 16% |
| Decoder init | 15ms | 1.3% |
| Decoder steps (49 × 12.6ms) | 620ms | 54% |
| **Total** | **~1,137ms** | |
| **RTFx** | **26.3x** | |

**IMPORTANT: decoder_init is NOT a bottleneck.** The old ~196ms reading was the encoder's GPU async completion time being billed to decoder_init. This was a PROFILING ATTRIBUTION BUG, now fixed. See `docs/EDGE-HUNT-REPORT.md` and `docs/ORT-FLUSH-INVESTIGATION.md`.

---

## Profiling Attribution Fix (2026-06-18)

### The Bug
`decoderInitMs` was reported as ~196ms. In reality, this was the encoder's GPU execution time (~178ms) surfacing at the first synchronization point after encoder submission. ORT's `device_queue_.Submit()` is non-blocking — the GPU hadn't finished when `session.run()` returned.

### The Fix
Added `encoderGpuDrainMs` metric (gated behind `encoderGpuDrain` flag). When enabled, it calls `getData(false)` on the encoder output to force GPU completion, then re-wraps the GPUBuffer. This moves the ~178ms wait from `decoderInitMs` to `encoderGpuDrainMs` where it belongs.

### Why fp32 looked "fast"
`maybeCastEncoderHiddenStates()` calls `getData(true)` when casting fp32→fp16, which forces the same GPU flush. The cost was hidden in `encoderOutputCastMs` instead of `decoderInitMs`. The fp16 path is actually more efficient (no unnecessary CPU round-trip).

### New Metrics (always reported)
- `encoderRunMs` — `session.run()` wall time
- `encoderOutputCastMs` — time in `maybeCastEncoderHiddenStates` (0ms for fp16→fp16)
- `encoderGpuDrainMs` — GPU drain wait (0 when flag off, ~193ms when on)
- `encoderTotalMs` — `encoderRunMs + encoderGpuDrainMs`
- `encoderOutputLocation` — `'gpu-buffer'` or `'cpu'`
- `encoderOutputDtype` — `'float16'` or `'float32'`

### Correct Interpretation
```
encoderRunMs      = ORT run wall time (command setup/submission)
encoderGpuDrainMs = GPU async completion wait (PROFILING ONLY, gated)
encoderTotalMs    = sync-to-completion wall time (honest encoder cost)
decoderInitMs     = varies: ~196ms without drain (queue wait), ~15ms with drain (true cost)
```

---

## Multi-Token Decoder Step (2026-06-19)

### What was done
Patched `decoder_step.onnx` to support dynamic sequence length:
- Changed `input_ids` dim[1] from hardcoded `1` to dynamic `sequence_length`
- Model now accepts `[batch, K]` for any K (verified K=2,4,8)
- Backward-compatible: K=1 path unchanged
- Token parity confirmed: K=2 pos0 matches K=1 pos0 (fp16 tolerance)

### Verification (Python ORT, CPU EP)
| Test | Result |
|------|--------|
| K=2: logits shape | [1, 2, 51866] ✓ |
| K=4: logits shape | [1, 4, 51866] ✓ |
| K=8: logits shape | [1, 8, 51866] ✓ |
| Continuation (K=2→K=2→K=1) | KV 1→3→5→6 ✓ |
| Token parity (K=2 pos0 vs K=1) | max diff 0.0078 ✓ |

### Speed Impact
**None yet.** The multi-token model is infrastructure — it *enables* batching but doesn't speed up single-token decode. Speedup requires a draft model (smaller/faster) for speculative decoding.

### Code Ready
- `runDecoderStepMultiToken()` in executor.ts — feeds K tokens in one ORT call
- `secondArgmax()` helper — for future speculative decoding
- Speculative greedy decode loop was prototyped and **REVERTED** — self-speculation breaks token parity because rejection changes KV context

### Model Files
- `public/models/fp16/decoder_step.onnx` — multi-token version (109KB + 635MB .data)
- `public/models/fp16/decoder_step_k1.onnx` — original backup (hardcoded dim[1]=1)
- `public/models/fp16/decoder_step_multi.onnx` — embedded-data copy for Python testing

---

## Edge Hunt — Concluded (2026-06-18)

See `docs/EDGE-HUNT-REPORT.md` for full details.

| Edge | Hypothesis | Result | Decision |
|------|-----------|--------|----------|
| A | Buffer re-wrap (strips session callbacks) | 197ms → 197ms | **REJECT** |
| B2 | GPU pipeline flush via `getData()` | 196ms → 15ms | **DIAGNOSTIC ONLY** |
| B | Copy bridge (GPU→GPU copy) | Not implemented | **NOT FEASIBLE** |
| C | fp32 cast bridge (re-export decoder_init) | Not tested | **DEFER** |
| D | Graph Identity/Cast tricks | Not tested | **REJECT** (penalty is pre-computation) |

**Root cause:** ORT's `Submit()` is non-blocking. The encoder's GPU work hadn't finished when decoder_init started. Both sessions share the same `device_queue_`, so decoder's first compute dispatch waits behind encoder's pending work. This is correct behavior — the 178ms is real GPU time, not a bug.

**ORT C++ fix available** (reference only, not deployed): `docs/ort-flush-fence.patch` adds `OnSubmittedWorkDone` fence in `Flush()`. Saves ~18ms vs JS `getData()` approach. Requires ORT Web rebuild.

---

## Optimization Sprint Results (2026-06-19)

See `docs/OPTIMIZATION-SPRINT-REPORT.md` for full details.

### P1: Multi-Token Decoder Step — ACCEPT (infra), DEFER (speedup)
Model deployed. Speculative decode needs draft model.

### P1-B: Encoder Scan — ACCEPT (q8 next), DEFER (graph capture)
Encoder is 2,326 nodes, 1 Cast. q8 encoder exists (0.6GB). Graph capture only helps multi-chunk.

### P1-C: CPU Prep — DEFER
Mel processing already uses the exact Whisper N_FFT=400 contract through a
cached Bluestein FFT, precomputed twiddles, cached filterbank, and buffer reuse.
Audio mono copy is necessary. WASM/WebGPU mel is not justified yet.

---

## Three Deployed Optimizations (perf/whisper-webgpu-decode)

1. **GPU KV Cache Bridge** — `preferredOutputLocation` per-output map, KV stays on GPU
2. **Stripped fp16 Encoder** — removed Cast(f16→f32) from encoder output
3. **Exact Whisper Mel N_FFT=400** — cached Bluestein FFT with reusable buffers

---

## Rejected Experiments (branches preserved)

| # | Experiment | Branch | Verdict |
|---|-----------|--------|---------|
| 4 | Shared WebGPU device (shader-f16) | `perf/shared-webgpu-device` | ❌ Step +22% regression |
| 5 | Fused encoder_decoder_init ONNX | `perf/fused-encoder-decoder-init` | ❌ Init +19%, step +17% |
| 6 | GPU suppression mask + ArgMax | `perf/gpu-argmax` | ❌ Step +11% Turkish |
| 7 | ONNX "simple" buffer cache | main (reverted) | ❌ RTFx -12% |
| 8 | Hot-loop KV precompute | `perf/hot-loop` | ❌ Broke session reuse |

**Note:** Experiments 4-6 were tested BEFORE the harness fix (page reload degradation). Verdicts may shift with the fixed harness. Re-testing recommended.

### Additionally Rejected/Closed (2026-06-19)
- **decoder_init optimization** — only 15ms after profiling fix, not a bottleneck
- **fused encoder_decoder_init** — already rejected above
- **shared WebGPU device** — already rejected above
- **GPU ArgMax for decoder_init** — 15ms is fine, parity already achieved
- **Identity/Cast graph tricks** — penalty was profiling attribution, not real
- **CPU pass-through for encoder output** — fp16 GPU pass-through is optimal

---

## Harness & Production Infrastructure

### Test Harness
- **Location:** `N:\github\asrjs\webgpu-agent-test\` (separate directory, not in git)
- **Start:** `cd /n/github/asrjs/webgpu-agent-test && npm run dev` (port 8765)
- **Benchmark URL:** `http://localhost:8765/?auto=fp16io-fp16-webgpu&local=1&gpuKv=1`
- **Profiling URL:** Add `&encoderGpuDrain=1` for honest per-phase metrics
- **Results:** `_results/*.json` + `GET /__test_results__`
- **Auto-run-twice:** warmup + measurement in single page load
- **Available URL params:** `auto`, `local`, `gpuKv`, `encoderGpuDrain`, `encoderGpuFlush`, `encoderBufferRewrap`, `encoderOutputCpu`, `profiling`, `language`, `maxNewTokens`, `numBeams`, `bestOf`, `temperature`

### Production Additions (on main)
- `flushAllModels()` in `DefaultSpeechPipeline` — VRAM cleanup between audio files
- Profiling sub-buckets in `TranscriptMetrics`
- `encoderGpuDrainMs`, `encoderTotalMs`, `encoderOutputCastMs` metrics
- `encoderGpuDrain` flag (profiling only, off by default)
- `runDecoderStepMultiToken()` — multi-token decoder step infrastructure

---

## Known Pitfalls

1. `window.location.href` causes progressive GPU degradation (use auto-run-twice)
2. `browser_navigate` strips URL params — use `browser_console` to set `location.href` instead
3. ORT WebGPU "simple" buffer cache = 12% slowdown (reverted)
4. ORT WebGPU REQUIRES `wasmPaths` even for WebGPU sessions
5. `onnx.save(save_as_external_data=True)` corrupts weights — use `convert_model_to_external_data`
6. New ONNX outputs default to `gpu-buffer` — must explicitly add to `'cpu'`
7. Deployed `.onnx` filename must match internal `external_data.location`
8. Reuse browser tabs — new tabs allocate fresh VRAM
9. `flushModel()` between cache-key changes — pipeline leaks VRAM
10. `getData()` adds ~18ms staging buffer overhead vs native fence — use only for profiling
11. Self-speculative greedy decoding breaks token parity — needs draft model

---

## Key Artifacts

| Path | Purpose |
|------|---------|
| `docs/HANDOVER.md` | This document |
| `docs/EDGE-HUNT-REPORT.md` | Edge A/B/B2/C/D investigation — root cause proven |
| `docs/ORT-FLUSH-INVESTIGATION.md` | ORT C++ command buffer audit |
| `docs/PROFILING-REPORT-2026-06-19.md` | Honest profiling baseline (post-fix) |
| `docs/OPTIMIZATION-SPRINT-REPORT.md` | P1/P1-B/P1-C with ACCEPT/REJECT/DEFER |
| `docs/Whisper-Optimizations.md` | Full experiment tracker (11 experiments) |
| `docs/STRUCTURAL-OPTIMIZATION-REPORT.md` | Structural optimization report |
| `docs/ort-flush-fence.patch` | Fix A reference patch (C++ fence) |
| `src/models/whisper-seq2seq/executor.ts` | Main executor: decode loop, profiling, multi-token |
| `src/models/whisper-seq2seq/ort.ts` | ORT session creation, artifact resolution |
| `src/models/whisper-seq2seq/types.ts` | Type definitions (all flags) |
| `src/audio/whisper-mel.ts` | Exact 400-point Whisper mel preprocessing with cached FFT |
| `src/runtime/media.ts` | Audio decode + downmix |
| `tools/whisper-onnx-export/` | ONNX export/modification tools |
| `public/models/fp16/decoder_step.onnx` | Multi-token decoder_step (dynamic seq_len) |

---

## Remaining High-Impact Opportunities

1. **Speculative decoding with draft model** — multi-token model deployed, needs draft model (e.g., 2-layer Whisper-tiny). Expected ~43% decoder speedup.
2. **q8 encoder** — exists in `public/models/q8/`. Expected ~200ms encoder (half of fp16).
3. **Graph capture for multi-chunk** — precompile shaders once, reuse across chunks.
4. **VRAM optimization** — encoder/decoder sessions may share weight buffers (~1.85GB baseline).
5. **WASM SIMD mel** — 30-50ms potential reduction in CPU prep.
