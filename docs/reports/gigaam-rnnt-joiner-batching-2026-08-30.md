# GigaAM RNN-T speculative batched joiner decode (2026-08-30)

## Summary

Ported the X-ASR speculative batched-joiner decode to the GigaAM v3 E2E
RNN-T executor and improved it with adaptive batch width and predictor-state
caching. The joint graph is row-parallel (leading dim = frame batch), so all
window frames are scored against the current predictor state in one dispatch.
Result on the shared 18.714 s warmed LibriVox fixture, hybrid composition
(encoder WebGPU, decoder WASM, joint WASM), ORT Web 1.29, same-session A/B:

| Browser | Before (sequential) | After (batched) | decodeMs before/after |
|---------|--------------------:|----------------:|----------------------:|
| Chrome  | 27.88x RTFx         | 47.31x RTFx     | 518.7 ms -> 263.8 ms  |
| Edge    | 28.02x RTFx         | 48.75x RTFx     | 509.1 ms -> 250.8 ms  |

Transcripts identical to the placement-A/B baseline runs byte-for-byte
(cross-checked Chrome vs Edge vs the committed baselines). Repeat runs in
the same session stayed in the 40-55x band; the committed numbers are the
median-repeat legs.

## Design

- Blank rows never change the predictor state, so the first non-blank row of
  a batch is exactly what the sequential loop would emit. After an emission
  the predictor advances and the suffix INCLUDING the emitting frame is
  re-batched (GigaAM allows multiple tokens per frame; frame advances to the
  emission frame, not past it). Only a no-emission batch consumes its whole
  window.
- Frames that hit maxTokensPerFrame advance without further scoring,
  matching the sequential loop's clamp.
- Predictor outputs for the current (label, h, c) triple are cached across
  frames: a blank run costs one joint dispatch per frame and zero extra
  LSTM dispatches.
- Adaptive batch width: the first batch is 2 rows; a blank (no-emission)
  batch doubles the window up to 64 rows; any emission resets it to 2. This
  bounds wasted speculative rows when the joint runs on WASM (compute-bound)
  while retaining dispatch amortization. A naive full-suffix batch, as used
  for X-ASR's GPU-resident streaming path, measured 5.4x on this hybrid
  composition - each emission re-batched the entire remaining frame suffix
  against a compute-bound WASM joint and lost more than it saved. Width
  policy is therefore
  adaptive, not suffix-consuming.
- Sticky fallback: a joint graph that throws on batched shapes or returns
  non-row-parallel logits latches batching off permanently for the
  executor; the sequential path is bit-identical. PipelineAbortedError from
  inside a batch run re-throws without latching.

## Evidence

- Unit parity: tests/gigaam-rnnt-joiner-batching.test.ts (6 tests): batched
  vs forced-sequential transcript/token-timing/decodeIterations parity,
  multi-emission frame parity, maxTokensPerFrame clamp on both paths,
  reject and badshape sticky fallback, abort propagation without latching,
  and full tensor-disposal tracking.
- Real artifacts Node: GIGAAM_RNNT_ONNX_SMOKE=1 tests/gigaam-rnnt-onnx-backends.test.ts
  2/2 (WASM official example.wav + WebGPU session class) after the change.
- Browser: post-change legs live in tools/data/results/gigaam/
  gigaam-rnnt-librivox-18s-warmed-joiner-batching-{,edge-}encoder-webgpu-decoder-wasm-joint-wasm.json
  while the pre-change placement-A/B baselines keep their original filenames
  (run via N:\github\asrjs\webgpu-agent-test harness, --oracle=none; correctness by
  transcript identity across all four files).
- Suite: 1043 passed / 18 artifact-gated skips; tsc clean.

## Follow-ups this unlocks

- GigaAM RNN-T is now ~47-49x on the hybrid placement; the encoder GPU leg
  (83-88 ms) is no longer dwarfed by decode. The next bottleneck is the
  per-step joint/decoder WASM dispatch floor (still ~100 emissions x 2 tiny
  runs), i.e. exactly where a GRU-capable decoder-on-GPU path would help if
  the dispatch cost of tiny recurrence graphs improves (built-in EP or
  plugin EP kernel work).
- The same adaptive-width strategy is the measured-rejected path for full
  suffix batching in WASM-jointed RNN-T; record it so future ports (NeMo
  RNN-T) start with the adaptive policy.
