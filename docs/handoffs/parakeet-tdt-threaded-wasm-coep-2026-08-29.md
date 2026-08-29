# Parakeet TDT threaded WASM and confidence-gate handoff

Date: 2026-08-29

## Threaded WASM root cause

The sibling Chrome harness (`N:\github\asrjs\webgpu-agent-test`) reported
`WASM_THREADS_UNAVAILABLE` for `cpuThreads > 1`. `crossOriginIsolated` was
already true and `SharedArrayBuffer` was available, but every
`/ort-dist/*` response only carried `Cross-Origin-Resource-Policy:
same-origin`. ORT Web 1.29's threaded build spawns module workers that fetch
sibling `.mjs` and `.wasm` subresources; under the page's
`Cross-Origin-Embedder-Policy: require-corp`, those subresources need their own
compatible embedder-policy header. Chrome blocked them with
`coep-frame-resource-needs-coep-header`, so worker setup never completed.

The harness static asset route now also sends
`Cross-Origin-Embedder-Policy: require-corp` for served ONNX/WASM assets.

## Measurement

Same fixture, fp16 WebGPU encoder + fp32 WASM decoder + ONNX preprocessor,
18.714 s `librivox.org.wav`, exact 91-token parity:

| cpuThreads | Warmed median | Native RTFx | Decode (ms) |
| ---------- | ------------: | ----------: | ----------: |
| 1          |     ~1,065 ms |       17.7x |        ~720 |
| 4          |     ~1,325 ms |       14.2x |        ~950 |

Threaded WASM is now available and correct, but this one-frame GRU decoder is
slower with threads than single-threaded WASM. Keep the Parakeet production
default single-threaded; use `--cpu-threads=N` only as a larger-workload
diagnostic. The harness now omits `cpuThreads` by default so it follows the
library `navigator.hardwareConcurrency` value.

## Confidence-gate hot path

`NemoTdtTranscriptionOptions` now exposes `returnConfidence` (defaults to
true). When false, the decoder still performs the token and TDT-duration
argmax but skips the full-vocabulary softmax/entropy pass and leaves the
native confidence summary undefined. This matches the `wav2vec2` convention
and the gated behavior in the faster `parakeet.js` reference, and gives
throughput-only callers an allocation-free transcript path.

Validation: full suite 1017 passed / 18 artifact-gated skips; typecheck clean.
