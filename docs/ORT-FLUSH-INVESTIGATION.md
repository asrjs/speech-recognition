# ORT WebGPU fp16 Command Buffer Investigation

**Date:** June 18, 2026  
**Agent:** Bev (P520, Windows 11, RTX 5060 Ti)  
**Repo:** `microsoft/onnxruntime` @ `main` (33b389a)  
**Follow-up to:** `docs/EDGE-HUNT-REPORT.md`

---

## 1. Executive Summary

The Edge Hunt proved that ORT WebGPU's fp16 encoder output path incurs a ~178ms
penalty in `decoder_init`. We cloned onnxruntime and traced the full C++/JS
command-submission pipeline to understand **why** this happens and **whether**
the fix belongs in ORT or Dawn.

**Key findings:**

1. **ORT DOES flush properly** — `OnRunEnd` calls `Flush()` → `Finish()` → `Submit()`
2. **The 178ms is GPU async execution time** — `Submit()` is non-blocking; the encoder's
   ~180ms of GPU compute hasn't finished when decoder_init starts. Both sessions share
   the same `device_queue_`, so the decoder's first compute dispatch waits behind the
   encoder's pending work.
3. **fp32 is "fast" only because a CPU cast hides the sync** — `maybeCastEncoderHiddenStates`
   calls `getData(true)` when converting fp32→fp16, which forces the GPU flush. The
   cost is just billed to `encoderOutputMs` instead of `decoderInitMs`. Total time is
   the same or worse.
4. **The fix is either ORT-level (add fence in Flush) or Dawn-level (fix fp16 buffer
   binding pipeline barriers).**

---

## 2. The fp32 vs fp16 Mystery — SOLVED

### The question

| Path | decoderInitMs | Why? |
|------|--------------|------|
| fp16 GPU→GPU | **196ms** | GPU sync wait inside `session.run()` |
| fp32 GPU→fp16 CPU | **23ms** | Sync already happened during cast |
| fp16 CPU→CPU | 19ms | No GPU at all |

### The answer: `maybeCastEncoderHiddenStates` (executor.ts:174-205)

```typescript
// When encoder outputs fp32 but decoder expects fp16:
const f32Data = isGpuBufferTensor(encoderHiddenStates) && encoderHiddenStates.getData
    ? (await encoderHiddenStates.getData(true)) as Float32Array  // ← FORCES GPU FLUSH
    : encoderHiddenStates.data as Float32Array;
// ... fp32→fp16 cast on CPU → new CPU tensor
```

**For fp32 encoder output:**
1. Encoder `session.run()` → GPU output (fp32)
2. `maybeCastEncoderHiddenStates` detects dtype mismatch
3. Calls `encoderHiddenStates.getData(true)` → **downloads GPU→CPU, forces flush** (~193ms)
4. fp32→fp16 conversion on CPU
5. Decoder init gets CPU fp16 tensor → **15ms**

**For fp16 encoder output:**
1. Encoder `session.run()` → GPU output (fp16)
2. `maybeCastEncoderHiddenStates` — type matches, **skips cast entirely**
3. GPU tensor passes straight through
4. Decoder init gets GPU fp16 tensor → **196ms** (178ms GPU wait + 18ms actual)

**The "fp32 is fast" was an illusion.** The ~193ms flush cost is hidden in
`encoderOutputMs` (the inter-session gap). The fp16 path is actually more optimal
(no unnecessary CPU round-trip), but the GPU async penalty surfaces in the next
session's timing.

---

## 3. Full Command Submission Trace

### Encoder session

```
encoderSession.run({ input_features })
  → WebGpuKernel::Compute()                        [webgpu_kernel.cc:18]
    → WebGpuContext::Run()                         [webgpu_context.cc:187]
      → LaunchComputePipeline()                    [webgpu_context.cc:793]
        → compute_pass_encoder.DispatchWorkgroups()
        → num_pending_dispatches_++ = N
      → if N >= 16: Flush()                        [line 508]
  → WebGpuExecutionProvider::OnRunEnd()            [webgpu_execution_provider.cc:822]
    → context_.Flush(BufferManager())              [line 823]
      → EndComputePass()                           [webgpu_context.cc:754]
      → current_command_encoder_.Finish()          [line 784]
      → device_queue_.Submit(1, &command_buffer)   [line 785]  ← NON-BLOCKING
      → buffer_mgr.RefreshPendingBuffers()         [line 787]
      → current_command_encoder_ = nullptr          [line 789]
```

### Inter-session (JS layer)

```
encoderHiddenStates = encoderOutputs[...]  // GPU tensor (fp16)
// maybeCastEncoderHiddenStates: type matches → NO-OP → GPU tensor passes through

feeds = { input_ids, encoder_hidden_states: encoderHiddenStates /*GPU*/ }
decoderInitSession.run(feeds)
  → wasm-core-impl.ts:600 → webgpuRegisterBuffer(gpuBuffer, sessionId)
    → post-webgpu.js:122 → WebGPU.importJsBuffer()  // handle mapping only
  → WebGpuKernel::Compute()
    → WebGpuContext::Run()
      → bind encoder_hidden_states GPUBuffer as input
      → DispatchWorkgroups()  
      → Flush() → Submit()  ← WAITS ~178ms for encoder's GPU work to finish
```

### Timeline (fp16 path)

```
T+0     encoder session.run() starts
T+180   encoder OnRunEnd → Submit()  [GPU starts executing encoder work]
T+180   JS: maybeCastEncoderHiddenStates → NO-OP (both fp16)
T+180   JS: feeds built, decoderInitSession.run() called
T+180   C++: webgpuRegisterBuffer → importJsBuffer (instant)
T+180   C++: Run() → bind buffer → DispatchWorkgroups → Flush() → Submit()
        └─ GPU queue: decoder work WAITS behind encoder work
T+360   GPU: encoder work completes
T+375   GPU: decoder_init compute completes
        └─ decoderInitMs = 196ms (178ms wait + 18ms compute)
```

### Timeline (fp32 path — the "fast" one)

```
T+0     encoder session.run() starts
T+180   encoder OnRunEnd → Submit()
T+180   JS: maybeCastEncoderHiddenStates — dtype mismatch!
T+180   JS: getData(true) → creates copy cmd → Submit() → mapAsync()
        └─ WAITS for GPU to finish encoder + copy (193ms total)
T+373   JS: fp32→fp16 cast on CPU (~1ms)
T+374   JS: decoderInitSession.run(feeds)  [CPU tensor]
T+389   decoder_init completes
        └─ decoderInitMs = 15ms  (no GPU wait needed)
        └─ TOTAL inter-session gap ≈ 193ms (hidden in encoderOutputMs)
```

---

## 4. C++ Code Analysis

### `WebGpuContext::Flush()` (webgpu_context.cc:749-791)

```cpp
void WebGpuContext::Flush(const webgpu::BufferManager& buffer_mgr) {
  if (!current_command_encoder_) {
    return;  // Early return if nothing to flush
  }
  EndComputePass();
  // ... profiling query resolve ...
  auto command_buffer = current_command_encoder_.Finish();
  device_queue_.Submit(1, &command_buffer);     // NON-BLOCKING
  buffer_mgr.RefreshPendingBuffers(...);
  current_command_encoder_ = nullptr;
}
```

**Key observation:** `Submit()` returns immediately. The GPU executes the command
buffer asynchronously. There is no fence, no `OnSubmittedWorkDone`, no
synchronization point. The encoder's GPU work is still in-flight when
`OnRunEnd` returns to JS.

### `WebGpuContext::Run()` (webgpu_context.cc:502-509)

```cpp
if (num_pending_dispatches_ >= max_num_pending_dispatches_ ||  // 16
    (is_profiling_ && query_type_ == TimestampQueryType::AtPasses)) {
  EndComputePass();
}
if (num_pending_dispatches_ >= max_num_pending_dispatches_) {
  Flush(buffer_mgr);
  num_pending_dispatches_ = 0;
}
```

Batches of up to 16 dispatches per command buffer. The final partial batch is
flushed by `OnRunEnd`. No per-dtype logic anywhere.

### `max_num_pending_dispatches_` = 16 (webgpu_context.h:351)

```cpp
const uint32_t max_num_pending_dispatches_ = 16;
```

---

## 5. Why Is This fp16-Specific? (It Isn't)

The penalty is NOT fp16-specific — it's **cross-session GPU tensor pass-through**.
The fp16 path simply happens to exercise this code path more commonly because:

1. fp16 encoder + fp16 decoder → no dtype mismatch → `maybeCastEncoderHiddenStates` no-ops → GPU tensor passes through
2. fp32 encoder + fp16 decoder → dtype mismatch → forced CPU download → implicit flush

If you ran fp32 encoder + fp32 decoder (no dtype mismatch), the same 178ms penalty
would appear in `decoderInitMs`.

---

## 6. Fix Options

### Fix A: Add `OnSubmittedWorkDone` fence in `Flush()` (ORT-level, easy)

```cpp
// After device_queue_.Submit(1, &command_buffer) in Flush():
wgpu::Future fence = device_queue_.OnSubmittedWorkDone(
    wgpu::CallbackMode::WaitAnyOnly,
    [](wgpu::QueueWorkDoneStatus) {});
instance_.WaitAny(fence, UINT64_MAX);
```

**Effect:** Makes `OnRunEnd` block until GPU finishes. Moves the ~178ms from
`decoderInitMs` to the encoder's `OnRunEnd`. Net latency unchanged, but timing
attribution becomes correct.

**Pros:** Trivial change, confirms diagnosis, cleaner metrics.  
**Cons:** Blocks CPU during GPU execution (wastes CPU). Net latency unchanged.

### Fix B: Add `BufferMapAsync` readback fence (application-level, JS)

Instead of modifying ORT, add an explicit `getData()` call on the encoder output
before passing to decoder_init — same as Edge B2 diagnostic but as a production
feature gated behind an option.

**Effect:** Same as Fix A, but in JS.

### Fix C: Dawn upstream fix

If Dawn is creating unnecessary pipeline barriers for fp16 buffer bindings
(internal staging, format conversion, validation), this is a Dawn bug. Report to
`https://bugs.chromium.org/p/dawn/issues/list`.

To verify: run with `DISABLE_ROBUSTNESS` toggle OFF (undoing the
`"disable_robustness"` device toggle) and measure if fp16 decoder_init gets
worse. If robust buffer access checks add pipeline barriers for fp16, disabling
them may help. ORT currently enables `"disable_robustness"` by default (line 535).

### Fix D: Single-session approach (architectural)

If encoder and decoder share an ORT session (single graph), ORT handles all
synchronization internally. The GPU queue would naturally pipeline.

**Pros:** Eliminates cross-session penalty entirely.  
**Cons:** Requires merged ONNX graph. Not always possible (e.g., beam search).

### Fix E: fp32 bridge (already explored as Edge C)

Re-export decoder_init to accept fp32 input. The `maybeCastEncoderHiddenStates`
cast would then be a no-op (dtype matches), but the GPU tensor still passes
through and the penalty persists. Unless you also force a GPU flush.

---

## 7. The `getData()` Downloader Path

For reference, the downloader in `post-webgpu.js:179-208`:

```javascript
return async () => {
  const gpuReadBuffer = device.createBuffer({size, usage: COPY_DST | MAP_READ});
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, gpuReadBuffer, 0, size);
  device.queue.submit([commandEncoder.finish()]);  // Submit + create barrier
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);   // Block until done
  return gpuReadBuffer.getMappedRange().slice(0, originalSize);
};
```

The `mapAsync()` call is what blocks. The `submit()` inserts a pipeline barrier
(COPY from the encoder's output buffer), which forces the GPU to complete the
encoder's compute pass before the copy can proceed.

---

## 8. `webgpuRegisterBuffer` — No Hidden Sync

`post-webgpu.js:122-148`:

```javascript
Module["webgpuRegisterBuffer"] = (buffer, sessionHandle, bufferHandle) => {
    // ...
    const bufferHandle = WebGPU.importJsBuffer(buffer, deviceHandle);
    buffer[gpuBufferMetadataSymbol] = [bufferHandle, 1];
    return bufferHandle;
};
```

`importJsBuffer()` is Emscripten's handle mapping — it creates a WASM-side
`WGPUBuffer` handle for the JS `GPUBuffer`. No buffer content is read, no
commands are submitted to the queue. This is purely metadata.

---

## 9. Source Files Reference

| File | Lines | Role |
|------|-------|------|
| `webgpu_execution_provider.cc` | 822-823 | `OnRunEnd` → calls `Flush` |
| `webgpu_context.cc` | 187-512 | `Run()` — dispatches work, batches flushes |
| `webgpu_context.cc` | 749-791 | `Flush()` — `EndComputePass` + `Finish` + `Submit` |
| `webgpu_context.h` | 186-204 | `GetComputePassEncoder` / `EndComputePass` |
| `webgpu_context.h` | 350-351 | `num_pending_dispatches_`, `max_num_pending_dispatches_=16` |
| `post-webgpu.js` | 122-148 | `webgpuRegisterBuffer` — JS↔WASM buffer handle mapping |
| `post-webgpu.js` | 179-208 | Downloader — `getData()` implementation |
| `buffer_manager.cc` | 548-570 | Buffer creation and caching |
| `buffer_manager.cc` | 611-616 | `RefreshPendingBuffers` — buffer lifecycle |
| `data_transfer.cc` | 10-37 | `CopyTensor` — GPU↔GPU, CPU↔GPU copies |
| `wasm-core-impl.ts` | 600-617 | `gpu-buffer` tensor registration in WASM session |
| `executor.ts` | 174-205 | **`maybeCastEncoderHiddenStates` — THE fp32 sync point** |
| `executor.ts` | 1231-1290 | `runDecoderInit` — timing probes |
| `executor.ts` | 1845-1918 | Encoder run + Edge A/B2 diagnostics |

---

## 10. Conclusions

1. **ORT's flush mechanism is correct** — commands are submitted after every `session.run()`.
2. **The penalty is GPU async execution latency**, not a missing `Submit()`.
3. **fp32 is not "faster"** — it just hides the same ~193ms flush cost in
   `maybeCastEncoderHiddenStates` via `getData()`. The fp16 path is actually
   more efficient (no CPU round-trip) but the timing attribution is misleading.
4. **Fix A (fence in Flush) is the simplest ORT-level intervention** — it moves
   the wait to the correct accounting bucket without changing net latency.
5. **The ultimate fix is likely in Dawn** — if fp16 buffer binding triggers
   unnecessary pipeline barriers, the Dawn team should address it.

---

## Appendix A: Fix A Patch

See `ort-flush-fence.patch` alongside this report.

```diff
--- a/onnxruntime/core/providers/webgpu/webgpu_context.cc
+++ b/onnxruntime/core/providers/webgpu/webgpu_context.cc
@@ -783,6 +783,12 @@ void WebGpuContext::Flush(const webgpu::BufferManager& buffer_mgr) {
   }
   auto command_buffer = current_command_encoder_.Finish();
   device_queue_.Submit(1, &command_buffer);
+
+  // FIX: Block until GPU completes all submitted work.
+  // This prevents the next session from paying the encoder's GPU execution
+  // time as a hidden penalty in its first compute dispatch.
+  Wait(device_queue_.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
+         [](wgpu::QueueWorkDoneStatus){}));
+
   if (graph_capture_state_ != GraphCaptureState::Replaying) {
     buffer_mgr.RefreshPendingBuffers(graph_capture_state_);
   }
```

## Appendix B: Build Commands for Custom ORT Web

Building ORT Web with WebGPU support requires Emscripten and the Dawn native
library. Reference build:

```bash
# From onnxruntime repo root:
# Prerequisites: emsdk, ninja, cmake, Python 3.11+

# Build WebAssembly + WebGPU backend:
python tools/ci_build/build.py \
  --build_dir build/wasm_webgpu \
  --config Release \
  --build_wasm \
  --use_webgpu \
  --skip_tests \
  --parallel

# Output: build/wasm_webgpu/Release/dist/ort.webgpu.min.js + .wasm
```

For Windows native (testing Fix A without WASM):
```bash
# Requires Dawn native library built separately
cmake -S . -B build/win_webgpu ^
  -G "Visual Studio 17 2022" ^
  -D onnxruntime_USE_WEBGPU=ON ^
  -D onnxruntime_BUILD_SHARED_LIB=OFF ^
  -D CMAKE_BUILD_TYPE=Release
cmake --build build/win_webgpu --config Release
```
