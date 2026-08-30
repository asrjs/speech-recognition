# WebGPU Plugin EP 0.3.0 x Parakeet TDT decoder-joint spike (2026-08-30)

Scope: goal work item 3 - bounded compatibility spike against the separate
native ONNX Runtime WebGPU Plugin EP (`onnxruntime-ep-webgpu` 0.3.0,
Python; not an npm/browser package). Question: does its new GRU/LSTM and
integer-op coverage make it a better decoder host than the built-in ORT
Web 1.29 WebGPU EP we ship against, and does it change the browser plan?

## Setup

- `pip install onnxruntime-ep-webgpu==0.3.0` (pulls `onnxruntime` 1.29.0).
- Registration flow that actually works (the EP is NOT usable via the
  `providers=[...]` name list - it silently falls back to CPU):
  `ort.register_execution_provider_library('webgpu', get_library_path())`
  then `SessionOptions().add_provider_for_devices([device], {})`.
- Model: `ysdede/parakeet-tdt-0.6b-v3-onnx` `decoder_joint-model.onnx`
  (fp32) and `decoder_joint-model.int8.onnx`, single-step shapes
  (1 frame, batch 1) with recurrent state carried; 20 warm measurement
  steps; CPU EP as parity oracle.
- Scripts: tools/spikes/parakeet-webgpu-plugin-ep*.py (bench + ORT profiler
  partition probes).

## Results (same session, NVIDIA host GPU)

| Leg | create | first step | steady step | finite | vs CPU parity |
|---|---|---|---|---|---|
| fp32 CPU EP | 108 ms | 2.3 ms | 1.66 ms | yes | oracle |
| fp32 plugin WebGPU | 463 ms | 450 ms | 3.15 ms | yes | max abs 3.7e-3 |
| int8 CPU EP | 45 ms | 1.5 ms | 0.63 ms | yes | oracle |
| int8 plugin WebGPU | 370 ms | 251 ms | 5.49 ms | yes | max abs 0.0 |

Profiler partition (per-run node counts):

- fp32: 26/26 nodes on WebGpuExecutionProvider - both `LSTM` kernels run on
  GPU natively. Confirms 0.3.0's recurrent coverage on the plugin surface.
- int8: 53 nodes on GPU but 9 on CPU, including BOTH `LSTM_quant` kernels
  and the quantized joint MatMul - quantized recurrence still falls off the
  GPU, mirroring our GigaAM RNN-T INT8 finding on the built-in browser EP
  (int8 = size win, not a WebGPU speed win).

## Verdict

1. Compatibility: PASS for the fp32 decoder-joint graph - it loads, runs,
   and matches CPU within fp32 noise with 100% GPU partition.
2. Performance: the plugin EP is NOT faster than our current browser path
   for this workload; on a tiny single-step graph per-call dispatch
   dominates (3.15 ms GPU vs 1.66 ms CPU per step). Our browser decoder
   runs bigger multi-token work per call, but the built-in ORT Web EP in
   1.29 already covers GRU/LSTM (validated in work item 1), so there is no
   graph the plugin can host that the browser EP cannot.
3. Browser plan: NO CHANGE. The plugin EP is Python/.NET-only; do not add
   it as a dependency or a promotion gate. Its remaining value is as a
   diagnostic oracle for kernel-coverage questions (e.g. it independently
   confirms quantized-LSTM GPU support is the missing kernel class).
4. Deferred-dispatch/buffer-cache claims from the 0.3.0 notes were not
   separable in this microbench (create+first step improved vs 0.2.x?
   no baseline run here); treat as untested rather than adopted.

Evidence: scripts committed under tools/spikes/; re-run with
`python tools/spikes/parakeet-webgpu-plugin-ep.py <decoder_joint.onnx>`.
