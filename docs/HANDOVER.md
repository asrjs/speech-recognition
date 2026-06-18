# Whisper WebGPU Optimization — Handover Report

**Date:** 2026-06-17 (updated)
**Active branch:** `main` (ahead of origin by 7 commits)
**Branch merged:** `perf/whisper-webgpu-decode` → `main`
**Tests:** 633 passed, 4 skipped

---

## Current Production State

4-graph splitgraph Whisper WebGPU pipeline:
```
encoder.onnx (fp16_iofp32_fp16out) → decoder_init.onnx → decoder_step.onnx → decoder_align.onnx
```

**Flags:** `experimentalGpuKvCache=true` (all others disabled by default)

**Performance (RTX 5060 Ti, English JFK 29.9s, 50 tokens, GPU-warm):**
```
RTFx: 24.4x | Total: ~1225ms
├─ Preprocess: ~127ms
├─ Encoder:    195ms
├─ Decoder Init: 200ms (KNOWN REGRESSION, was ~69ms)
├─ Steps:      700ms (97% ORT execution)
└─ JS overhead: ~3ms
Session creation (cold): 16.5s (one-time)
```

### Profiling Available (merged from `perf/profile`)
Fine-grained sub-buckets in `stageMetrics`:
- `decoderInitTensorCreateMs`, `decoderInitLogitReadMs`, `decoderInitKvExtractMs`
- `decoderStepTensorCreateMs`, `decoderStepLogitReadMs`, `decoderStepKvMergeMs`
- `sessionCreateMs`

---

## Three Deployed Optimizations (perf/whisper-webgpu-decode)

1. **GPU KV Cache Bridge** — `preferredOutputLocation` per-output map, KV stays on GPU
2. **Stripped fp16 Encoder** — removed Cast(f16→f32) from encoder output
3. **Fast Mel N_FFT=512** — replaced Bluestein FFT with 512-point radix-2

---

## Rejected Experiments (branches preserved)

| # | Experiment | Branch | Verdict |
|---|---|---|---|
| 4 | Shared WebGPU device (shader-f16) | `perf/shared-webgpu-device` | ❌ Step +22% regression |
| 5 | Fused encoder_decoder_init ONNX | `perf/fused-encoder-decoder-init` | ❌ Init +19%, step +17% |
| 6 | GPU suppression mask + ArgMax | `perf/gpu-argmax` | ❌ Step +11% Turkish |
| 7 | ONNX "simple" buffer cache | main (reverted) | ❌ RTFx -12% |
| 8 | Hot-loop KV precompute | `perf/hot-loop` | ❌ Broke session reuse |

**Note:** Experiments 4-6 were tested BEFORE the harness fix (page reload degradation).
Verdicts may shift with the fixed harness. Re-testing recommended.

---

## Harness & Production Infrastructure

### Test Harness
- **Location:** `N:\github\asrjs\webgpu-agent-test\` (separate directory, not in git)
- **Start:** `npx vite --host 0.0.0.0 --port 8765 --strictPort --force` (pty=true on Windows)
- **Benchmark URL:** `http://localhost:8765/?auto=fp16io-fp16-webgpu&maxNewTokens=50&gpuKv=1&language=en`
- **Results:** `_results/*.json` + `GET /__test_results__`
- **Auto-run-twice:** warmup + measurement in single page load (no reload degradation)

### Production Additions (on main)
- `flushAllModels()` in `DefaultSpeechPipeline` — VRAM cleanup between audio files
- Profiling sub-buckets in `TranscriptMetrics`

---

## Known Pitfalls

1. `window.location.href` causes progressive GPU degradation (use auto-run-twice)
2. ORT WebGPU "simple" buffer cache = 12% slowdown (reverted)
3. ORT WebGPU REQUIRES `wasmPaths` even for WebGPU sessions
4. `onnx.save(save_as_external_data=True)` corrupts weights
5. New ONNX outputs default to `gpu-buffer` — must explicitly add to `'cpu'`
6. Deployed `.onnx` filename must match internal `external_data.location`
7. Reuse browser tabs — new tabs allocate fresh VRAM
8. `flushModel()` between cache-key changes — pipeline leaks VRAM

---

## Key Artifacts

| Path | Purpose |
|---|---|
| `src/utils/webgpu-context.ts` | Shared WebGPU device (on rejected branch) |
| `tools/whisper-onnx-export/fuse_encoder_decoder_init.py` | Fused ONNX graph merger |
| `tools/whisper-onnx-export/fuse_logit_argmax.py` | GPU suppression mask injector |
| `tools/whisper-onnx-export/consolidate_external_data.py` | External data merger |
| `tools/whisper-onnx-export/verify_fused_parity.py` | ONNX output parity checker |
| `tools/whisper-onnx-export/verify_decoder_step_fast_argmax.py` | Fast ArgMax parity checker |
| `docs/Whisper-Optimizations.md` | Full experiment tracker (11 experiments) |
| `docs/SHARED-DEVICE-REPORT.md` | Shared device rejection report |
| `docs/FUSED-INIT-REPORT.md` | Fused init rejection report |

---

## Remaining High-Impact Opportunities

1. **Fix decoder init regression (200ms → 69ms)** — cross-session GPU tensor handoff
2. **Static KV cache + graph capture** — requires new ONNX export
3. **Batched beam graph** — one ORT call per token for all beams
4. **VRAM optimization** — encoder/decoder sessions may share weight buffers (~1.85GB baseline)
