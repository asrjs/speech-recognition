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

## Multi-clip confirmation and quant-aware default (shipped same day)

Second clip A/B (tr-tdk-18s.wav, 18.6 s, Node WASM; ON vs OFF transcripts
compared char-exact):

| decoder | grid ON decodeMs | grid OFF decodeMs | transcripts |
| --- | --- | --- | --- |
| fp32 | 342.5 (19 grid runs) | 408.0 | SAME (win ~16%) |
| int8 | 281.8 (35 grid runs) | 297.5 | **DIFF** (267 vs 269 iterations) |

The int8 transcript DIFF is the decisive finding: dynamic-range
requantization over the wider '[1, features, width]' input breaks row
independence badly enough to flip a near-tie on real audio, violating the
"backend differences must not change output semantics" contract. Combined
with jfk-short (fp32 +18%), the fp32 win is confirmed on two clips and the
int8 risk is proven real.

Shipped default (quant-aware, derived from the loaded decoder filename):
grid batching defaults ON for fp32 decoders; fp16/int8 stay sequential
unless explicitly opted in via 'options.gridBatching'. This also matches
the operating guidance that small models like eou-120m should just run
fp32 - there is no quantization pressure on a 120M model, and fp32 is both
the fastest and the only grid-safe decoder quant measured here.

End-to-end default verification (parity script, no flags): fp32 default run
reports 19 grid runs; int8 default run reports 0 grid runs.

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

1. [Shipped same day] Hard safety net: grid batching is refused for int8
   RNNT decoders even when explicitly opted in (recoverable warning
   'nemo-rnnt.grid-batching-int8-unsupported'), because the tr-tdk-18s A/B
   proved transcript divergence. fp16 stays explicitly opt-in (probe rows
   are bit-exact; decode was parity).
2. Browser demo page for the eou-120m preset in 'webgpu-agent-test'
   (none exists today; 'N:/github/asrjs/streaming-demo' already wires the
   eou-120m preset alongside TDT v2/v3 and is the integration reference)
   to extend the A/B to Chrome headless WebGPU hosts.

## Addendum: gate re-probe ported from TDT; natural-audio decode win doubled (2026-08-30 later)

The RNNT gate had the same once-per-utterance warmup flaw the TDT fix
exposed: the 24-column sampling window accumulated during dense opening
speech and latched batching off before any pause. Ported the fix: the
sequential fallback counts consecutive blank visits and resets the
sampling window after six blanks (~0.5 s at 80 ms/frame), letting the
grid re-probe each silence gap; fluke re-probes stay bounded by the
sampling window. Both the grid and sequential emission paths reset the
blank-run counter.

Real-artifact re-measurement (Node WASM, encoder fp32, char-identical
transcripts and unchanged decodeIterations in every cell):

| Clip | grid runs (before -> after) | decodeMs ON | decodeMs OFF | win |
| --- | --- | --- | --- | --- |
| tr-tdk-18s 18.6 s | 19 -> 70 | 179.5 | 377.4 | 52% (2.1x) |
| jfk-short 11 s | 8 -> 29 | 166.0 | 257.5 | 36% |

The tr-tdk-18s fp32 win more than doubled (was ~16% with the
once-per-utterance gate). jfk-short stays exact-reference-matched
(tokenIds + visibleText + EOU). The re-probe is especially relevant for
the realtime EOU use case, where alternating speech/silence chunks are
the norm. Evidence:
'tools/data/results/nemo-rnnt/parakeet-eou120m-grid-reprobe-*.json'.
