# NeMo RNNT (Parakeet Realtime EOU 120M v1) speculative grid batching (2026-08-30)

Status: implemented, unit- and real-artifact-validated, shipped as OPT-IN
(options.gridBatching === true). Scope:
'src/models/nemo-rnnt/executor.ts' (fused RNNT decoder-joint decode loop),
'tests/nemo-rnnt-grid-batching.test.ts' (new),
'tools/model-debugging/scripts/node-asrjs-parakeet-realtime-parity.mjs'
(--grid-batching=off flag).

## Design

Same speculative template as the TDT port: one '[1, features, width]' joint
run scores a frame window against the current (target, GRU state) pair; blank
rows consume frames (RNNT blank always advances one frame - no duration
head); the first emission row commits its token, transfers the grid output
states, and re-batches from the EMITTING frame (multi-token-per-frame
parity, capped by maxSymbolsPerStep); no-emission windows double the width
(2 -> 32). The 24-column / 70% utilization gate and sticky
malformed-shape latch mirror the TDT port.

Graph truths verified with real artifacts
(N:/models/onnx/nemo/parakeet-realtime-eou-120m-v1-onnx, onnxruntime-node):

- decoder_joint accepts '[1, 512, W]' encoder outputs (the joint consumes
  512-dim ENCODER OUTPUTS, not 128-dim mel) and returns '[1, W, 2, 1027]'
  row-major; per-row trailing 'distributionSize' slice matches the
  sequential '[1,1,2,1027]' read.
- fp32 and fp16 graphs: rows are BIT-EXACT vs single-frame runs across
  targets 0/5/300/1025 and zero AND random non-zero states (slot-exact,
  probe script pattern now captured in the TDT report lesson).
- int8 graph: argmax parity held in all probes, but logits differ (dynamic
  range quantization spans the wider batched input) - transcript parity
  must be verified per quant, never assumed.

## Real-artifact A/B (Node WASM, jfk-short.wav 11 s, encoder fp32)

Exact reference-transcript parity (tokenIds + visibleText + rawText + EOU)
in every cell:

| decoder | grid ON decodeMs | grid OFF decodeMs | verdict |
| --- | --- | --- | --- |
| int8 | 354.7 | 211.0 | regression ~40% |
| fp16 | 637.3 | 635.0 | parity |
| fp32 | 216.2 | 265.0 | win ~18% |

Decision: grid batching ships OPT-IN for RNNT. The measured int8 regression
(dynamic-range requantization scales with the wider '[1, features, width]'
input) outweighs the dispatch win on the likely default preset; the fp32
win justifies keeping the implementation and the option. A quant-aware
default (on for fp32, off for int8/fp16) is the follow-up once more clips
confirm the fp32 win.

## Validation

- 'tests/nemo-rnnt-grid-batching.test.ts': 5 tests (parity incl.
  decodeIterations equality, multi-token re-batch, dense-audio gate bounds,
  sequential-by-default pin, rejection latch).
- The legacy 'tests/nemo-rnnt-executor.test.ts' mock now rejects grid
  requests shape-first WITHOUT consuming scripted steps - it pins the
  sequential path through the executor's real latch behavior.
- Full suite: 1058 passed / 18 artifact-gated skips; tsc clean; build clean.

## Reusable lessons

1. Quantized graphs can break row-independence NUMERICALLY (int8 dynamic
   range) even when argmax parity holds; always A/B transcript + timing per
   quant before shipping speculative batching as default.
2. decodeIterations must count every consumed row (including the emission
   row) or parity assertions against the sequential path will catch the
   undercount - as they did here.
3. On emission, resume AT the emitting frame (blank rows before it are
   already consumed); forgetting the frame advance silently re-decodes
   consumed frames with a wrong target.

Evidence JSONs:
'tools/data/results/nemo-rnnt/parakeet-eou120m-grid-ab-*.json'.

## Next steps

1. Quant-aware default for nemo-rnnt grid batching (fp32 on) after a
   multi-clip confirmation, incl. a blank-heavy clip.
2. Browser demo page for the eou-120m preset in 'webgpu-agent-test'
   (none exists today) to extend the A/B to Chrome headless WebGPU hosts.

