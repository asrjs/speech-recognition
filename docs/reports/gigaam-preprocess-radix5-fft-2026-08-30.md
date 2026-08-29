# GigaAM shared preprocessing FFT optimization (2026-08-30)

## Target

GigaAM preprocessing measured 117-131 ms of ~449 ms total in the browser
(26-29%). Stage attribution with a Node microbenchmark
(tools/scripts/benchmark-gigaam-preprocess.mjs, 11.29 s at 16 kHz, 64 mels,
nFft 320) found the 320-point FFT alone consumed 64.22 of 69.49 ms p50:
92% of the stage.

## Root cause

The shared MedAsrJsPreprocessor (used by GigaAM CTC, GigaAM RNN-T, and
lasr-ctc families) falls back to Bluestein's chirp-z algorithm for
non-power-of-two nFft. Bluestein computes a 320-point DFT as three
1024-point power-of-two FFTs per frame - 1128 frames meant ~3400 large
FFT calls per preprocess pass.

## Change

Added RadixFivePowerOfTwoFft in src/models/lasr-ctc/mel.ts: a direct
Cooley-Tukey decomposition for N = 5 * 2^m (GigaAM's 320 = 5 x 64 and
160 = 5 x 32). It runs one N/5-point power-of-two FFT per strided
subsequence plus a 5-point direct DFT with fully precomputed twiddle and
5-point factor tables - the same mathematical DFT up to float rounding.
MedAsrJsPreprocessor now selects: power-of-two twiddles, then radix-5,
then Bluestein fallback.

During development the first version computed the wrong DFT (max relative
error ~1.6e+2): the precomputed N^(n1*k2/N) twiddle table stored the wrong
sine sign. Verified against a naive DFT oracle and Bluestein after the fix.

## Measurement (Node v26, same-launch)

| Metric | Baseline | Radix-5 |
|---|---|---|
| FFT-only, 1128 frames | 64.22 ms | 17.29 ms (3.7x) |
| process() p50 | 69.49 ms | 24.51 ms (2.84x) |
| feature checksum | -893853.284 | -893853.284 |

Numerical agreement vs Bluestein: max relative error 2.659e-12 across 20
random trials. Browser end-to-end with exact oracles:

| Family | Before | After | Parity |
|---|---|---|---|
| GigaAM RNN-T (enc WebGPU, dec/joint WASM) | 449.3 ms / 25.1x | 342.8 ms / 32.9x | exact |
| GigaAM CTC fp16 WebGPU | (281 ms / 39.8x on the old matrix) | 180.3 ms / 61.0x | exact |

RNN-T preprocess phase dropped 130.8 -> 31.7 ms in the browser.

## Validation

- tests/composite-fft.test.ts: RadixFivePowerOfTwoFft matches a naive DFT
  for sizes 5, 10, 40, 80, 160, 320; agrees with Bluestein at GigaAM
  scale; rejects non-5 x 2^m sizes (8 tests pass)
- focused suites (gigaam-ctc, gigaam-rnnt, sensevoice, lasr-ctc): green,
  including the golden feature-parity tests
- full suite: 1017 passed / 18 artifact-gated skips

## Artifacts

tools/data/results/gigaam/v3-rnnt-encgpu-decwasm-1t-radix5-librivox.json,
tools/data/results/gigaam/ctc-fp16-webgpu-radix5-jfk.json

## Reproduction

node tools/scripts/benchmark-gigaam-preprocess.mjs
node scripts/run-gigaam-rnnt-webgpu.mjs --warmup=1 --repeat=3
node scripts/run-gigaam-webgpu.mjs --warmup=1 --runs=3

