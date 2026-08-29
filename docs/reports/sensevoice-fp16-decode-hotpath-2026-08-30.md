# SenseVoice fp16 CTC decode hot path (2026-08-30)

## Target

The SenseVoice decode phase measured 385.6 ms of a 710 ms transcribe on the
18.7 s librivox clip (316 output frames x 25055 vocab, float16 logits).
The graph returns half-bit patterns, so the runtime first scalar-converted
~7.9M fp16 values to float32 and then ran the two-pass generic argmax +
log-softmax over the copy.

## Change

`argmaxAndSelectedLogProbsFp16` in `src/ctc/decoder.ts` (exported from
`src/ctc/index.ts`) decodes raw fp16 bit buffers without materializing
floats:

- max: strict scan over an integer ordering key of the half bits
  (sign-mapped so larger key == larger float; -0/+0 collapse to one key
  so the first strict maximum wins exactly like the float scan);
- log-sum-exp: Float64Array(65536) lookup table of `Math.exp(fp16ToFloat(code))`,
  accumulated per row. The table is built lazily on first use;
- score: `best - log(sum)` is algebraically identical to the reference
  `best - (best + log(sum(exp(x - best))))` expression;
- parity fallbacks (per row): NaN/infinity codes, maxima outside the
  [-80, +80] exp safe zone, and short/truncated rows all reroute through
  the converting generic pipeline, so raw-logit graphs keep identical
  semantics.

`src/models/sensevoice/executor.ts` routes both the single and batch paths
through `decodeLogitsBlock`, which keeps float16 tensors as `Uint16Array`
subarrays (zero-copy per batch row) and float32 tensors on the generic
pipeline. The batch path no longer copies converted floats per row.

This supersedes the 2026-08-29 conclusion that the remaining cost was
"intrinsic 4.68M Math.exp calls": fp16 codes are 16-bit indices, so exp
becomes a table lookup.

## Measurement

Node microbenchmark (`tools/scripts/benchmark-ctc-decode.mjs`, 9 runs, p50,
same-launch A/B, realistic log-prob bits):

| cell | reference (convert+argmax) | fast (LUT) | speedup |
| --- | --- | --- | --- |
| 187 x 25055 | 149.70 ms | 13.14 ms | 11.4x |
| 316 x 25055 | 260.04 ms | 22.02 ms | 11.8x |

Parity gate in the bench: ids equal, max score diff 0.00e+0 at 187 x 25055.

Chrome headless WebGPU harness (NVIDIA Blackwell adapter, warmup + 3 runs,
18.7 s librivox clip, `sensevoice-small-webgpu` preset):

| metric | before (2026-08-29) | after |
| --- | --- | --- |
| decodeMs | 385.6 | 163.8-178.8 |
| transcribeMs (median run) | 710.0 | 384.2-436.3 |
| RTFx (median run) | 26.42 | 48.71-45.29 |
| transcript | identical (byte-equal, 59 tokens) | identical |

Evidence:

- `tools/data/results/sensevoice/sensevoice-small-librivox-18s-warmed-webgpu-chrome.json` (before)
- `tools/data/results/sensevoice/sensevoice-small-librivox-18s-warmed-fp16-lut-webgpu-chrome.json` (after)

Note: an intermediate rerun with `oracle=fixed` shows status `fail` only
because the page still carries the JFK expected string for this non-JFK
clip; transcript parity is the correctness gate here.

## Validation

- new parity suite: `tests/ctc-decoder-fp16.test.ts` (5 tests: realistic
  rows, denormals/signed zeros, out-of-zone raw logits, NaN/inf rows,
  truncated buffers; all ids equal, score diffs <= 1e-5)
- adversarial Node parity probe: 6 shapes incl. -60000..60000 wide range,
  edge-of-safe-zone tails with exp underflow, all-zeros rows, denormal
  soup, truncated mid-row: ids equal, maxDiff 0.00e+0
- focused: tests/ctc* + tests/sensevoice*: green
- full suite: 1033 passed / 18 artifact-gated skips; typecheck + build clean
- browser single-run cell: SenseVoice WebGPU run above (decodeMs 385.6 -> 163.8)
- browser batch cell: `--batch` jfk-short rerun passes the parity gates
  (`batchFirstMatch`, `batchSecondNonEmpty`) with per-row zero-copy fp16
  decode (`decodeMs` 114.4 ms over both rows); evidence refreshed in
  `tools/data/results/sensevoice/sensevoice-small-jfk-short-batch-webgpu-chrome.json`
  (the previous committed copy carried no phase metrics or oracle)
