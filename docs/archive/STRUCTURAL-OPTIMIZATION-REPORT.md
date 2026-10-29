# Whisper WebGPU Structural Optimization Report

**Date:** June 18, 2026  
**Agent:** Bev (P520, Windows 11, RTX 5060 Ti)  
**Branch:** main (8 commits ahead of origin)  
**ORT:** Custom build (internal 1.28.0, package.json 1.26.0)  
**Model:** whisper-large-v3-turbo, fp16_iofp32_fp16out encoder + fp16 decoder  

---

## 1. Baseline Profile Recap

**Fixture:** English JFK 30s (29.9s), GPU-warm, 50 tokens, gpuKv=1

```
Total: 904 ms, RTFx: 33.07x

Encoder:          180 ms (19.9%)  ← encoderRunMs: 180ms, encoderOutputMs: 0ms
Decoder Init:     197 ms (21.8%)  ← decoderInitRunMs: 197ms (ORT session.run)
Decoder Steps:    443 ms (49.0%)  ← 49 steps, P50: 9ms, P95: 10ms, Max: 11ms
Preprocess/other:  84 ms ( 9.3%)

Session creation (cold): 11,544 ms (one-time)
```

**Comparison with handover baseline (24.4x RTFx):**
The current RTFx (33x) is higher than the handover's 24.4x. This is likely due to:
- Custom ORT build (1.28.0 internal vs previous version)
- RTX 5060 Ti driver/runtime improvements
- Same measurement methodology (auto-run-twice, warmup + measurement)

---

## 2. Decoder Init Regression Diagnostics (Track A)

### Diagnostic Matrix Results

| Test | Encoder Output | Decoder Init RunMs | Encoder RunMs | Total Ms | RTFx | Token Parity |
|------|---------------|-------------------|---------------|----------|------|--------------|
| **A1** (baseline) | GPU fp16 | **197 ms** | 180 ms | 904 ms | 33.1x | ✓ identical |
| **A2** (force CPU) | CPU fp16 | **19 ms** | 374 ms | 1033 ms | 28.9x | ✓ identical |
| **A3** (fp32 path) | GPU fp32 | **23 ms** | 480 ms | 1252 ms | 23.9x | ✓ identical |
| **A4** (fp16io→fp32) | GPU fp16→fp32 | ERROR (dtype mismatch) | — | — | — | — |

### Key Findings

1. **The decoder_init 200ms regression IS caused by cross-session GPU tensor handoff.**
   - With CPU input (A2): decoderInitRunMs = **19ms** (10x faster)
   - With GPU fp16 input (A1): decoderInitRunMs = **197ms**
   - The handoff adds **~178ms** overhead

2. **The penalty is specific to fp16 GPU tensors.**
   - fp32 GPU→GPU (A3): decoderInitRunMs = **23ms** (fast!)
   - fp16 GPU→GPU (A1): decoderInitRunMs = **197ms** (slow)
   - fp16 CPU→CPU (A2): decoderInitRunMs = **19ms** (fast)
   - This suggests an ORT WebGPU issue with fp16 GPU buffer sharing between sessions

3. **The current production path (A1) is the best overall.**
   - A1: 904ms total, 33x RTFx — despite 197ms decoder init
   - A2: 1033ms total, 29x RTFx — decoder init fixed but encoder download adds 194ms
   - A3: 1252ms total, 24x RTFx — fp32 encoder too slow
   - The fp16 encoder speed advantage (180ms vs 480ms) offsets the decoder init penalty

4. **Evidence wording (per plan requirements):**
   > The current evidence strongly suggests a cross-session GPU tensor handoff/synchronization penalty specific to fp16 tensors between encoderSession and decoderInitSession. The fp32 path does not exhibit this penalty (23ms vs 197ms), and forcing the fp16 encoder output to CPU eliminates it (19ms). We do NOT claim "browser security forces a hard hardware fence" — the root cause within ORT WebGPU's fp16 buffer management requires further investigation.

### Required Measurements Collected

```
A1 (baseline):
  encoderRunMs:          180.005 ms
  encoderOutputMs:         0.015 ms
  encoderOutputLocation:   gpu-buffer
  encoderOutputDtype:      float16
  decoderInitMs:         197.170 ms
  decoderInitRunMs:      197.115 ms
  decoderInitInputMs:      0.010 ms
  decoderInitOutputMs:     0.025 ms
  gpuInputCount:         785
  cpuInputCount:          50
  gpuOutputCount:        408
  cpuOutputCount:         50
  gpuDownloadCount:        0

A2 (encoder output CPU):
  encoderRunMs:          374.275 ms  ← +194ms (GPU→CPU download)
  encoderOutputLocation:   cpu
  decoderInitRunMs:       19.490 ms  ← -178ms (no handoff penalty)
  gpuInputCount:         784
  cpuInputCount:          51

A3 (fp32 path):
  encoderRunMs:          479.770 ms  ← fp32 encoder is 2.6x slower
  encoderOutputLocation:   gpu-buffer
  encoderOutputDtype:      float32
  decoderInitRunMs:       23.340 ms  ← fp32 GPU→GPU is fast!
```

### Decision: DIAGNOSTIC ONLY

The A2 fix (force encoder output to CPU) solves the decoder init regression but creates a net-negative result due to the encoder download cost. No production change recommended.

---

## 3. Multi-Token Decoder Step Feasibility (Track B1)

### Result: REJECT (current model) / DEFER (re-export feasible)

**Test:** Attempted to pass `input_ids` shaped `[1, K]` for K=2, K=4 to the existing `decoder_step.onnx`.

```
K=1: SUCCESS — logits shape (1, 1, 51866), KV shape (1, 20, 2, 64)
K=2: FAILED — "Got invalid dimensions for input: input_ids for index 1 Got: 2 Expected: 1"
K=4: FAILED — "Got invalid dimensions for input: input_ids for index 1 Got: 4 Expected: 1"
```

**Root cause:** The ONNX export script (`export_whisper.py`, line 745-761) defines `input_ids` dynamic axes as `{0: "batch"}` only — dimension 1 is exported as static (value: 1).

**Re-export feasibility:** The HuggingFace Whisper decoder natively supports multi-token input. The fix is a one-line change in `build_step_dynamic_axes`:
```python
# Current:
"input_ids": {0: "batch"},
# Fix:
"input_ids": {0: "batch", 1: "seq_len"},
```
Plus updating the dummy input shape and logits output axis. PyTorch 2.7.1 + CUDA is available on this machine.

**Rejection criteria met:** "graph only supports [1, 1]" — the current model rejects K>1 input.

---

## 4. Static KV / Graph Capture Feasibility (Track B2)

### B2-A: Baseline Dynamic (no graph capture)
Already measured in A1: step P50 = 9ms, step run = 440ms total (49 steps)

### B2-B: Dynamic Model + freeDimensionOverrides {batch:1}
```
Session creation: SUCCESS
decoderStepP50Ms:   8.64 ms  (vs 9.0ms baseline — within noise)
decoderStepP95Ms:  11.39 ms  (vs 10.0ms baseline — slightly worse)
decoderStepRunMs: 438.46 ms  (vs 440ms baseline — neutral)
totalMs:          914.91 ms  (vs 904ms baseline — +1.2%)
rtfx:             32.69x     (vs 33.07x baseline — -1.1%)
Token parity:     ✓ identical
```
**Decision: NEUTRAL** — No meaningful improvement on WebGPU. The CPU test showed 15% improvement, but WebGPU kernels are already specialized for batch=1 internally.

### B2-C: Dynamic Model + enableGraphCapture
```
Session creation: FAILED
Error: "This session cannot use the graph capture feature as requested by the user as all compute gr..."
```
**Decision: REJECT** — Graph capture requires static shapes. The dynamic `past_sequence` dimension (which changes every step) prevents graph capture.

### B2-D: Static Exported Model + Graph Capture
**Not tested.** Would require re-exporting decoder_step with static `past_sequence` (e.g., 448). Key risk: every step would read/write the full max-length KV cache (~9MB), potentially increasing memory traffic enough to erase any graph capture benefit.

**Decision: DEFER** — The current step time (9ms P50) is already very fast. Static KV risks increasing memory traffic. Only worth pursuing if step time becomes the dominant bottleneck (currently it's 49% of total, but per-step is already optimized).

---

## 5. ORT Version Matrix (Track B3)

### Current State
```
webgpu-agent-test package.json:  onnxruntime-web 1.26.0
Internal version string in JS:   1.28.0 (custom build from WSL2)
speech-recognition package.json: onnxruntime-web ^1.27.0-dev.20260506-673c3320fc

npm latest stable:  1.26.0
npm latest dev:     1.27.0-dev.20260506-673c3320fc
```

### Key API Availability
```
enableGraphCapture:     ✓ Present in JS source (session options)
freeDimensionOverrides: ✓ Present in JS source (session options)
storageBufferCacheMode: ✗ NOT in JS source (0 matches in ort.webgpu.min.mjs)
                         Only in WASM binary — not exposed in JS API
```

### Decision: DEFER
ORT version matrix testing deferred. The current custom build (1.28.0 internal) is newer than both the latest stable (1.26.0) and latest dev (1.27.0-dev). Testing older versions is unlikely to reveal improvements. The `storageBufferCacheMode` option does not exist in the JS API and should not be used.

---

## 6. Decoder Graph Inspection Notes (Track B4)

### decoder_step.onnx (fp16, 185MB, 611 nodes)

**Inputs (17):**
```
input_ids:                              int64,  ['batch', 1]          ← STATIC dim 1
past_key_values.{0-3}.decoder.key:      float16, ['batch', 20, 'past_sequence', 64]  ← DYNAMIC
past_key_values.{0-3}.decoder.value:    float16, ['batch', 20, 'past_sequence', 64]  ← DYNAMIC
past_key_values.{0-3}.encoder.key:      float16, ['batch', 20, 1500, 64]  ← STATIC (encoder seq)
past_key_values.{0-3}.encoder.value:    float16, ['batch', 20, 1500, 64]  ← STATIC
```

**Outputs (9):**
```
logits:                                 float16, ['batch', 'MatMullogits_dim_1', 51866]
present.{0-3}.decoder.key:              float16, ['batch', 20, 'present_sequence', 64]
present.{0-3}.decoder.value:            float16, ['batch', 20, 'present_sequence', 64]
```

**Dynamic dimensions:**
- `batch` — symbolic (always 1 in practice)
- `past_sequence` — symbolic, grows each step (0, 1, 2, ..., N)
- `present_sequence` — symbolic, = past_sequence + 1
- `MatMullogits_dim_1` — symbolic but tied to input_ids[1] = 1 (static)

**Op distribution (top 10):**
```
Constant:           162
Unsqueeze:          103
Transpose:           65
Add:                 52
MatMul:              49
Concat:              44  ← 8 are real KV growth (axis=-2), rest are shape computation
Reshape:             25
Mul:                 25
Gather:              21
Shape:               21
LayerNormalization:  13
Softmax:              8
Cast:                 4  ← position indices only (int64↔float16)
```

**Key observations:**
- 49 MatMuls = 4 layers × (Q/K/V/O self-attn + Q/K/V/O cross-attn + FFN×2) ≈ 12 per layer
- 8 real KV growth Concat nodes (axis=-2 on sequence dimension) — these change output shape every step
- 4 Cast nodes: all for position index computation, not data type conversion
- No CPU fallback evidence: all ops are standard WebGPU-supported ONNX ops
- No suspicious or unsupported kernels detected

**Why ORT run time is ~9ms/step (WebGPU):**
The 49 MatMuls + 65 Transposes + 13 LayerNorms + 8 Softmaxes are the dominant cost. With 4 decoder layers, each layer processes ~12 MatMuls + 16 Transposes. The dynamic Concat for KV growth adds shape-dependent overhead. The JS overhead (tensor create, logit read, KV merge) is negligible (<0.1ms per step).

### decoder_init.onnx (fp16, 202MB, 589 nodes)

**Key difference from decoder_step:**
- 2 inputs only: `input_ids` ['batch', 'prompt_sequence'] (DYNAMIC!) + `encoder_hidden_states` ['batch', 1500, 1280]
- 17 outputs: logits + 16 KV (both decoder AND encoder KV)
- 589 nodes (slightly fewer than step's 611)
- `input_ids` has DYNAMIC `prompt_sequence` — the init model CAN accept multiple tokens

---

## 7. Memory / VRAM Behavior

```
A1 (baseline):     GPU inputs 785, CPU inputs 50, GPU outputs 408, CPU outputs 50, downloads 0
A2 (enc CPU):      GPU inputs 784, CPU inputs 51, GPU outputs 408, CPU outputs 50, downloads 0
A3 (fp32):         GPU inputs 785, CPU inputs 50, GPU outputs 408, CPU outputs 50, downloads 0
B2-B (freeDim):    Same as A1
```

No VRAM growth observed across repeated runs (flushModel/flushAllModels working correctly). The auto-run-twice harness ensures clean GPU state for measurement.

---

## 8. Token/Transcript Parity Table

| Test | Token IDs (first 10) | Transcript Match | Status |
|------|---------------------|------------------|--------|
| A1 (baseline) | [682, 264, 938, 2503, 295, 264, 1002, 11, 787, 257] | ✓ identical | pass |
| A2 (enc CPU) | [682, 264, 938, 2503, 295, 264, 1002, 11, 787, 257] | ✓ identical | pass |
| A3 (fp32) | [682, 264, 938, 2503, 295, 264, 1002, 11, 787, 257] | ✓ identical | pass |
| B2-B (freeDim) | [682, 264, 938, 2503, 295, 264, 1002, 11, 787, 257] | ✓ identical | pass |

All successful runs produce identical token IDs and transcripts. The transcript is:
"In the long history of the world, only a few generations have been granted the role of defending freedom in its hour of maximum danger. I do not shrink from this responsibility. I welcome it. I do not believe that any of us would exchange"

**Note:** The FP32 baseline text ("And so, my fellow Americans...") is a different part of the JFK speech. The 30s audio contains both passages. Status "check" (not "pass") is expected — both are valid transcriptions of different segments.

---

## 9. ACCEPT / REJECT / DEFER Decisions

| Track | Decision | Rationale |
|-------|----------|-----------|
| **A1** (baseline) | **ACCEPT** (current production) | Best overall: 904ms, 33x RTFx |
| **A2** (enc output CPU) | **DIAGNOSTIC ONLY** | Fixes decoder init (197→19ms) but net-negative (904→1033ms) |
| **A3** (fp32 path) | **DIAGNOSTIC ONLY** | Fast decoder init (23ms) but slow encoder (480ms), net-negative |
| **A4** (fp16io→fp32) | **REJECT** | Dtype mismatch error (fp16 encoder → fp32 decoder) |
| **B1** (multi-token, current model) | **REJECT** | Model has static input_ids dim=1, rejects K>1 |
| **B1** (multi-token, re-export) | **DEFER** | One-line fix in export script, PyTorch+CUDA available, requires model download + export |
| **B2-A** (baseline) | **ACCEPT** (current production) | 9ms P50 step time |
| **B2-B** (freeDimOverrides) | **NEUTRAL** | No meaningful improvement on WebGPU |
| **B2-C** (graph capture, dynamic) | **REJECT** | Session creation fails — dynamic shapes prevent graph capture |
| **B2-D** (static KV + graph capture) | **DEFER** | Requires re-export, risk of increased memory traffic, step time already fast |
| **B3** (ORT version matrix) | **DEFER** | Current build (1.28.0) is newer than npm stable/dev; storageBufferCacheMode not in JS API |
| **B4** (graph inspection) | **COMPLETE** | No suspicious kernels, no CPU fallback, bottleneck is MatMul/Transpose count |

---

## 10. Recommended Next Implementation Branch

### Primary recommendation: `perf/multi-token-decoder-step` (DEFER → IMPLEMENT)

**Rationale:** The B1 re-export is the highest-potential structural optimization:
- One-line change in `export_whisper.py` (`build_step_dynamic_axes`)
- Enables multi-token decoder_step calls (K tokens in one run instead of K serial runs)
- Prerequisite for speculative decoding
- PyTorch + CUDA available on this machine

**Implementation plan:**
1. Modify `build_step_dynamic_axes` to add `{1: "seq_len"}` to `input_ids` and `logits`
2. Re-export fp16 decoder_step with dynamic input_ids
3. Create browser test page that:
   - Runs encoder + decoder_init normally
   - Saves pre-K past KV state
   - Runs K serial decoder_step calls (baseline)
   - Resets KV and runs one [1, K] decoder_step call
   - Compares per-position argmax and continuation tokens
4. Test K=2, K=4, K=8
5. Measure: serialKStepRunMs vs multiTokenRunMs, logits shape, KV shapes, parity

### Secondary recommendation: Investigate fp16 cross-session handoff

**Rationale:** The Track A diagnostic revealed a 178ms penalty specific to fp16 GPU tensors crossing session boundaries. This is likely an ORT WebGPU bug or limitation.

**Investigation paths:**
1. Search ORT WebGPU source for fp16 buffer handling between sessions
2. Test with latest ORT dev builds (beyond 1.27.0-dev)
3. File an ORT issue with the diagnostic data (fp16 GPU→GPU: 197ms vs fp32 GPU→GPU: 23ms vs fp16 CPU→CPU: 19ms)
4. Explore ORT session options that might affect cross-session buffer sharing

### Not recommended:
- **Static KV + graph capture (B2-D):** Step time is already 9ms P50. The risk of increased memory traffic outweighs potential gains.
- **Fused encoder+decoder_init:** Already rejected (init +19%, step +17%). The fp16 handoff issue might change this verdict, but the fused graph was tested with the old harness.
- **Single-session architecture:** The plan explicitly defers this until a design that genuinely avoids the encoder output crossing a session boundary.

---

## Appendix: Diagnostic Instrumentation Added

All diagnostic flags are gated by URL parameters and default to off. They do not affect the production path.

| URL Parameter | Effect | Track |
|---------------|--------|-------|
| `encoderOutputCpu=1` | Forces encoder output to CPU (gpu-buffer → cpu) | A2 |
| `decoderGraphCapture=1` | Enables enableGraphCapture for decoder_step session | B2-C |
| `freeDimOverrides=1` | Enables freeDimensionOverrides {batch:1} for decoder_step | B2-B |

**Files modified:**
- `src/models/whisper-seq2seq/types.ts` — Added diagnostic fields to WhisperSplitGraphArtifactSource
- `src/models/whisper-seq2seq/ort.ts` — Added diagnostic fields to ResolvedWhisperArtifacts, pass-through in resolveSplitGraphArtifacts, freeDimensionOverrides in createWhisperOrtSession
- `src/models/whisper-seq2seq/executor.ts` — encoderOutputCpu guard, encoder sub-timing, decoder graph capture/freeDimOverrides in session creation
- `webgpu-agent-test/src/main.js` — URL parameter parsing, state flags, cache key, result output

**Encoder sub-timing added to metrics:**
- `encoderRunMs` — encoder session.run() time only
- `encoderOutputMs` — maybeCastEncoderHiddenStates processing time
- `encoderOutputLocation` — 'gpu-buffer' or 'cpu'
- `encoderOutputDtype` — 'float16' or 'float32'
