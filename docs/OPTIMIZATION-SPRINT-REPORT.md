# Optimization Sprint Report — Post Profiling Fix

**Date:** June 19, 2026  
**Agent:** Bev (P520, Windows 11, RTX 5060 Ti)  
**Baseline:** `docs/PROFILING-REPORT-2026-06-19.md`  
**Commits:** `206d870` through `79538b6`

---

## 1. Baseline Profile (fp16 GPU KV Cache, 30s JFK)

| Phase | Time | % |
|-------|------|---|
| Decoder steps (49 × 12.6ms) | 620ms | 54% |
| Encoder (run + drain) | 381ms | 34% |
| Mel preprocess | 97ms | 8.5% |
| Audio prep | 81ms | 7% |
| Decoder init | 15ms | 1.3% |
| **Total** | **1,137ms** | |

---

## 2. P1 — Multi-Token Decoder Step

### What was done
- Patched `decoder_step.onnx`: changed `input_ids` dim[1] from hardcoded `1` to dynamic `sequence_length`
- Verified K=2,4,8 inference in Python ORT (CPU)
- Verified token parity: K=2 pos0 matches K=1 pos0 (fp16 tolerance)
- Verified continuation: mixed K=2 and K=1 steps produce correct KV advancement
- Deployed as drop-in replacement (backward-compatible, K=1 unchanged)
- Added `runDecoderStepMultiToken()` infrastructure in executor.ts
- Prototyped K=2 greedy speculative decode → **REVERTED** (breaks token parity)

### Model verification

| Test | Result |
|------|--------|
| K=2: logits shape | [1, 2, 51866] ✓ |
| K=4: logits shape | [1, 4, 51866] ✓ |
| K=8: logits shape | [1, 8, 51866] ✓ |
| KV advancement (K=2) | seq_len 1→3 ✓ |
| Continuation (K=2→K=2→K=1) | 1→3→5→6 ✓ |
| Token parity (K=2 pos0 vs K=1) | max diff 0.0078 ✓ |
| Backward compat (K=1) | identical behavior ✓ |

### Speed impact

**None yet.** The multi-token model is infrastructure — it *enables* batching but doesn't speed up the current single-token decode loop. Speedup requires:

1. **Draft model** (smaller/faster Whisper, or n-gram model) that generates K candidate tokens
2. Main model verifies all K in one `runDecoderStepMultiToken()` call
3. Accept/reject per token (classic speculative decoding, Leviathan et al. 2023)

Without a draft model, greedy speculative decode with the same model breaks token parity because rejection changes the KV context. This was prototyped and reverted.

### Decision: **DEFER** (infrastructure ACCEPTED, speedup needs draft model)

The model change is deployed and backward-compatible. `runDecoderStepMultiToken()` and `secondArgmax()` are in place. The speculative decode loop is reverted but the pattern is documented. Next step: implement or integrate a draft model.

---

## 3. P1-B — Encoder Optimization Scan

### What was done
- Inspected encoder ONNX graph: 2,326 nodes, 1 Cast (fp32 input → fp16)
- Encoder output: fp16, shape [batch, 1500, 1280]
- q8 encoder exists in `public/models/q8/` (0.6GB vs 1.2GB fp16)
- Graph capture: 774 Constants + 2,326 nodes — feasible for multi-chunk audio

### Encoder graph assessment

| Check | Finding |
|-------|---------|
| Avoidable Cast nodes | Only 1 (fp32→fp16 at input) — necessary |
| Avoidable Shape/Reshape | None obvious (standard Whisper architecture) |
| Graph capture viable | Yes — large constant count, repeated encoder calls per chunk |
| q8 encoder | Exists, untested — expected ~200ms GPU time (half of fp16) |
| Preallocated output buffer | Not tested — would eliminate allocation per run |

### Decision: **ACCEPT q8 encoder as next test, DEFER graph capture**

q8 encoder is the quickest win — half the VRAM, potentially half the GPU time. Graph capture helps only for multi-chunk audio (same encoder called N times). For single-chunk transcription, it adds no value.

---

## 4. P1-C — CPU Prep Cleanup

### What was done
- Audited `whisper-mel.ts` (mel preprocessing) and `media.ts` (audio decode + downmix)
- Mel processing: already uses power-of-2 FFT (fast path), precomputed twiddles, cached filterbank, constructor-level buffer reuse
- Audio prep: mono copy is necessary (`AudioContext.close()` detaches internal buffers)
- `padToFrames`: zero-copy when frameCount == targetFrames (common case)

### Assessment

| Area | Finding | Potential |
|------|---------|-----------|
| Mel FFT | Already power-of-2, precomputed | — |
| Mel filterbank | Cached at construction | — |
| Work buffers | Reused (constructor allocation) | — |
| Audio mono copy | Necessary for AudioContext lifecycle | — |
| padToFrames | Zero-copy for exact-size audio | — |
| WASM SIMD mel | Not implemented | 30-50ms potential |
| WebGPU mel | Not implemented | 50-70ms potential |

### Decision: **DEFER — low ROI for JS-level changes**

The 97ms mel time is already well-optimized in pure JS. Reducing it further requires WASM SIMD or WebGPU compute shader — significant engineering for 30-50ms. Not justified given decoder steps (620ms) and encoder (381ms) dominate.

---

## 5. Closed Branches (Confirmed)

| Branch | Reason |
|--------|--------|
| decoder_init optimization | 15ms — not a bottleneck |
| fused encoder_decoder_init | Not needed |
| shared WebGPU device | Not needed |
| GPU ArgMax for decoder_init | Token parity already achieved, 15ms is fine |
| Identity/Cast graph tricks | Penalty was profiling attribution, not real |
| CPU pass-through for encoder | fp16 GPU pass-through is optimal |

---

## 6. Decisions Summary

| Track | Decision | Rationale |
|-------|----------|-----------|
| **P1** Multi-token model | **ACCEPT** (infra) | Model deployed, backward-compat, enables future speculative decode |
| **P1** Speculative decode loop | **DEFER** | Needs draft model; self-speculation breaks parity |
| **P1-B** q8 encoder | **ACCEPT** (next) | Quickest encoder win — half VRAM, likely faster |
| **P1-B** Graph capture | **DEFER** | Only helps multi-chunk audio |
| **P1-C** JS mel optimization | **DEFER** | Already well-optimized; WASM/WebGPU not justified yet |
| **P1-C** Audio prep copies | **REJECT** | Copy necessary for AudioContext lifecycle |

---

## 7. Recommended Next Branch

```
perf/speculative-decoding
```

**Prerequisites:**
1. Draft model (options):
   - 2-layer Whisper-tiny distilled decoder (~50M params, ~3ms/step)
   - N-gram statistical model (zero params, ~0.1ms/step, lower acceptance rate)
   - KV-cache-compatible architecture
2. Multi-token model already deployed (this sprint)
3. `runDecoderStepMultiToken()` already in place (this sprint)

**Expected speedup with K=4, 80% acceptance:**
- Serial: 49 steps × 12.6ms = 620ms
- Speculative: ~16 verify calls × 13ms + 49 draft calls × 3ms = 355ms
- **~43% decoder speedup, ~23% total speedup**

**Fallback if draft model not available:**
- P1-B: Test q8 encoder (expected ~200ms encoder, ~26.3x → ~35x rtfx)
- P1-B: Encoder output buffer preallocation (micro-optimization, ~2-5ms)

---

## 8. Artifacts

| File | Description |
|------|-------------|
| `docs/PROFILING-REPORT-2026-06-19.md` | Baseline profiling report (honest metrics) |
| `docs/ORT-FLUSH-INVESTIGATION.md` | ORT command buffer investigation |
| `docs/ort-flush-fence.patch` | Fix A reference patch (C++ fence) |
| `models/fp16/decoder_step.onnx` | Multi-token model (dynamic seq_len) |
| `models/fp16/decoder_step_k1.onnx` | Original model backup |
| `src/.../executor.ts` | `runDecoderStepMultiToken()`, `secondArgmax()`, `encoderGpuDrain` flag |

---

*Report auto-generated by Bev (hermes agent, P520).*
