# Milestone G2 Verification Report: Reliability, Streaming Contracts & Soak Qualification

**Date**: 2026-10-31  
**Milestone**: G2 (Reliability, Streaming Contracts & Soak Qualification)  
**Package**: `@asrjs/speech-recognition`  
**Overall Verdict**: **PASS**  

---

## 1. Executive Summary

Milestone G2 verifies the production reliability, streaming contracts, and soak qualification of `@asrjs/speech-recognition`. All four objectives mandated by the ASR.js Product Master Plan have been successfully implemented and verified:

1. **TEN-VAD Worker Cleanup (Issue #254 Remediation)**: All 5 worker cleanup and memory leak vulnerabilities have been remediated across `ten-vad-browser.ts`, `browser-realtime.ts`, `streaming-detector.ts`, and `ten-vad-worker.ts`. Verified with zero lingering Web Worker threads and memory freed on all failure paths.
2. **Streaming Telemetry Counters**: Implemented `StreamingTelemetrySnapshot` and `StreamingTelemetryCollector` in `src/runtime/streaming-telemetry.ts`, tracking sample counters (`inputSampleCount`, `acceptedSampleCount`, `emittedSampleCount`, `droppedSamples`, `duplicatedSamples`), lifecycle counters (`captureStartCount`, `captureStopCount`, `vadStartCount`, `vadEndCount`), and resource handles (`activeWorkerCount`, `activeAudioContextCount`, `disposalDurationMs`).
3. **Automated Synthetic PCM Soak Qualification**: Verified accelerated 44.1 kHz and 48 kHz schedules over multi-frequency tones and Gaussian noise with alternating speech, silence, empty chunks, and burst overflows using jittered chunk feeding (`[1, 7, 64, 128, 256, 511, 1024, 4096]`). Conservation of audio mass was strictly maintained (`droppedSamples === 0`, `unaccountedDroppedSamples === 0`).
4. **Deterministic Fault Injection**: Verified 4 failure modes (network 404/500 asset fetch failure, worker `onerror` panic, abort during in-flight init, and dispose during active streaming), confirming clean graceful degradation, zero leaks, and teardown in under 50 ms (well below the 500 ms ceiling).

---

## 2. Issue #254 Remediation (TEN-VAD Worker Cleanup)

### Root Cause Analysis
During long-running streaming sessions or fallback conditions, failed asset downloads or worker errors failed to terminate the underlying OS/browser Web Worker process, leaving zombie worker threads alive. Furthermore, if `_ten_vad_destroy()` or `_ten_vad_create()` threw during C ABI operations, native heap pointers were leaked in WASM linear memory.

### Implemented Fixes
- **Fix A (`src/runtime/ten-vad-browser.ts`)**: Added private `terminateWorker()` helper that unbinds event handlers and calls `this.worker.terminate()`. Invoked unconditionally on `onerror`, in `init()` error handler, and in `dispose()`.
- **Fix B (`src/runtime/browser-realtime.ts`)**: In `dispose()`, explicitly added `await tenVad?.dispose().catch(() => undefined)` before awaiting detector disposal, preventing orphan workers if handles diverged.
- **Fix C (`src/runtime/streaming-detector.ts`)**: Handled races between in-flight `this.tenVad.init()` and concurrent `dispose()` or cancellation by disposing `tenVad` immediately if `this.startCancelled || this.disposed` is set.
- **Fix D (`src/runtime/ten-vad-worker.ts`)**:
  - In `handleDispose()`, wrapped `_ten_vad_destroy(handlePtr)` in `try ... finally` to ensure pointers `[audioPtr, probPtr, flagPtr, handlePtr]` are always freed via `_free()` even if destruction returns an error.
  - In `handleInit()`, if `_ten_vad_create` returns a non-zero code, all four allocated pointers are immediately freed before throwing.

### Verification
- `tests/ten-vad-worker-abi.test.ts`: 6/6 tests passing (including explicit verification of memory freeing on destroy failure and init failure).
- `tests/browser-ten-vad.test.ts`: 27/27 tests passing (including worker termination on init failure and unhandled worker `onerror`).
- `activeWorkerCount` post-disposal: strictly 0.

---

## 3. Streaming Telemetry Counters & Audio Conservation

The `StreamingTelemetryCollector` implements real-time monotonic counters and invariant checking:

| Counter | Metric | Invariant / Target | Verified Value |
|---|---|---|---|
| `inputSampleCount` | Total raw input audio frames captured/injected | Monotonic | Tracked |
| `acceptedSampleCount` | Total audio frames accepted through resampler | $\text{input} \times \frac{\text{targetRate}}{\text{sourceRate}} \pm \text{lag}$ | Exact match |
| `emittedSampleCount` | Total frames forwarded to detector/transcription | Monotonic | Tracked |
| `droppedSamples` | Unaccounted dropped frames | `=== 0` | **0** |
| `duplicatedSamples` | Replayed or duplicate frames | `=== 0` | **0** |
| `captureStartCount` | Start lifecycle events | Integer | Monotonic |
| `captureStopCount` | Stop lifecycle events | Integer | Monotonic |
| `vadStartCount` | Speech onset events detected | Integer | Monotonic |
| `vadEndCount` | Speech termination events detected | Integer | Monotonic |
| `activeWorkerCount` | Active Web Worker processes | Returns to 0 on dispose | **0** |
| `activeAudioContextCount`| Active Web Audio contexts | Returns to 0 on dispose | **0** |
| `disposalDurationMs` | Time taken to dispose pipeline | `< 500 ms` | **< 50 ms** |

---

## 4. Accelerated Synthetic PCM Soak Test Results

Four synthetic soak schedules were executed across 10 alternating cycles of `[Speech 1.5s -> Silence 0.5s -> Empty chunk 0s -> Burst overflow 8192smp]` with jittered chunk slicing `[1, 7, 64, 128, 256, 511, 1024, 4096]`:

| Schedule | Source Rate | Target Rate | Resampler | Dropped | Duplicated | Unaccounted | Ratio Delta | Status |
|---|---|---|---|---|---|---|---|---|
| `48k-linear` | 48,000 Hz | 16,000 Hz | `StatefulLinearResampler` | 0 | 0 | 0 | < 0.0001 | **PASS** |
| `48k-sinc` | 48,000 Hz | 16,000 Hz | `SincResampler` | 0 | 0 | 0 | < 0.0001 | **PASS** |
| `44.1k-linear` | 44,100 Hz | 16,000 Hz | `StatefulLinearResampler` | 0 | 0 | 0 | < 0.0001 | **PASS** |
| `44.1k-sinc` | 44,100 Hz | 16,000 Hz | `SincResampler` | 0 | 0 | 0 | < 0.0001 | **PASS** |

Key Observations:
- **Empty Chunks (0 samples)**: Processed safely without NaN, infinity, or thrown exceptions.
- **Burst Chunks (8192 samples)**: Absorbed cleanly without buffer overflows or dropped samples.
- **Conservation of Audio Mass**: Confirmed `verifyConservation().ok === true` across all resamplers.

---

## 5. Deterministic Fault Injection Verification

| Scenario | Injected Condition | Expected Behavior | Observed Result | Status |
|---|---|---|---|---|
| **Scenario A** | Asset fetch 404 / network timeout | Terminate worker; degrade to rough energy gate | Worker terminated; `state: degraded`; 0 leaked workers; rough gate operational | **PASS** |
| **Scenario B** | Worker runtime panic (`onerror`) | Terminate worker; fail gracefully | Worker terminated immediately; `activeWorkerCount: 0` | **PASS** |
| **Scenario C** | Abort signal during in-flight `init()` | Terminate worker; return to idle | Worker terminated; `state: idle`; 0 leaked workers | **PASS** |
| **Scenario D** | `dispose()` during active speech streaming | Clean teardown within 500 ms | Teardown completed in < 50 ms; 0 active workers remaining | **PASS** |

---

## 6. Verification Receipts

```pwsh
# 1. Typecheck
npm --prefix speech-recognition run typecheck
# Result: Exit 0 (0 errors)

# 2. Root Import Graph Budget Check
node speech-recognition/tests/smoke/browser-root-import-graph.mjs
# Result: 44 static modules (Ceiling <= 44 budget: PASS)

# 3. Vitest Reliability Suites
npx vitest run tests/streaming-reliability.test.ts tests/ten-vad-worker-abi.test.ts tests/browser-ten-vad.test.ts
# Result: 3 passed test files, 42 passed tests (0 failed)
```

---

## 7. Conclusion & Sign-Off

Milestone G2 has achieved all reliability and soak requirements:
- TEN-VAD worker cleanup defect (Issue #254) is completely resolved and verified.
- Conservation of audio mass holds unconditionally (`droppedSamples === 0`).
- Root static import graph ceiling remains compliant at 44 modules.
- Safety rule respected: `docs/reports/handover-2026-10-30.md` remains untracked.

**Sign-off**: `worker_g2`  
**Verdict**: **READY FOR G3 BENCHMARK PROGRAM**
