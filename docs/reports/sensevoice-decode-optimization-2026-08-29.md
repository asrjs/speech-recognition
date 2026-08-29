# SenseVoice decode-phase optimization (2026-08-29)

## Target

The SenseVoice decode phase measured 160.9 ms of a 435.0 ms total (37%).
It is JS post-processing: argmaxAndSelectedLogProbs over 187 x 25055
logits, CTC collapse, tokenizer decode, and timing construction.

## Change

argmaxAndSelectedLogProbs in src/ctc/decoder.ts previously performed two
full traversals per row with a per-element ?? Number.NEGATIVE_INFINITY
coalesce and tracked rowMax separately from bestValue, although the two
are bitwise identical (one strict maximum over the same elements). The
optimized version keeps the exact expression shape
(bestValue - (bestValue + log(expSum || 1))) while using one tracked
maximum, hoisted Math.exp/Math.log, and contiguous typed-array access.
A hoisted fast path handles exact-length rows (the only real call shape -
the executor slices to frameLength * vocabSize); the guarded slow path
preserves the original short-row semantics. This is the same
constant-factor/view discipline the Parakeet TDT decode hot path used.

## Measurement

tools/scripts/benchmark-ctc-decode.mjs (187 x 25055, 15 runs, Node v26):

- baseline: p50 70.18 ms / min 68.88 ms
- optimized: p50 54.26 ms / min 53.49 ms (1.29x)
- outputs bit-identical: sample ids 481,0,0,0,0 and log-probs equal to all
  16 printed digits

The remaining cost is the intrinsic 4.68M Math.exp calls of exact
log-softmax normalization. An underflow-threshold skip (delta <= -40 is
provably exact in float64) would help raw-logit graphs but not SenseVoice,
whose graph emits log-probabilities in a narrow range; not pursued.

## Validation

- focused: tests/ctc + tests/sensevoice + tests/gigaam-ctc: 54 passed
- full suite: 1017 passed / 18 artifact-gated skips; build clean
- not yet re-run end-to-end in the browser: the phase sits inside the
  documented +-20% cross-session decode variance (161 ms phase vs ~35 ms
  saved), so a browser A/B cannot resolve it; the Node microbenchmark is
  the authoritative evidence by the same-launch methodology recorded in
  the quantization-matrix report.

