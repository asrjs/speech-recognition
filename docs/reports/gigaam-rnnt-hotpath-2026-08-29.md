# GigaAM RNN-T hot-path profile (2026-08-29)

## Scope

This report profiles the official GigaAM v3 E2E RNN-T ONNX artifacts through
the library executor, not a mock graph:

- encoder: `v3_e2e_rnnt_encoder.onnx`
- decoder: `v3_e2e_rnnt_decoder.onnx`
- joiner: `v3_e2e_rnnt_joint.onnx`
- audio: `tools/data/results/gigaam/v3-e2e-rnnt-example-reference.json`
- runner: `tools/model-debugging/scripts/node-gigaam-rnnt-benchmark.mjs`
- backend: ONNX Runtime Web WASM, one CPU thread

The reference transcript contains 78 tokens and is the exact parity oracle.
The benchmark records load time, total latency, RTFx, per-phase timing, token
count, transcript equality, and process memory.

## Baseline phase attribution

The baseline JSON is
`gigaam-rnnt-hotpath-phase-profile-2026-08-29.json` (one warm-up and three
measured runs). All runs produced the exact reference text.

| phase | observed range | share of total |
| --- | ---: | ---: |
| preprocessing | 77–145 ms | about 1–3% |
| encoder | 5,251–5,611 ms | about 93% |
| decoder + joiner | 192–256 ms | about 3–5% |
| total | 5,525–6,013 ms | — |

The encoder is therefore the only plausible source of a large end-to-end
speedup. Decoder allocation cleanup can improve a small tail, but cannot turn
the current WASM path into a major throughput win.

## Rejected allocation candidate

Two local candidates were measured against the pre-change control
`gigaam-rnnt-hotpath-baseline-2026-08-29.json`:

1. Keeping the extracted encoder frame tensor alive across all emitted symbols
   moved the three-run median to 6,586 ms and was rejected.
2. Hoisting only frame extraction while retaining the original per-joint
   tensor lifetime produced `gigaam-rnnt-hotpath-frame-hoist-2026-08-29.json`:
   median 6,105 ms versus 6,158 ms control, but p90 6,593 ms versus 6,171 ms.

The small median movement is within run-to-run variance and the tail regressed,
so neither candidate is treated as a production optimization. This is a
useful negative result: do not spend more decoder-loop effort while the
encoder dominates.

## Provider-placement capability

GigaAM RNN-T now accepts optional `encoderBackend`, `decoderBackend`, and
`jointBackend` fields on direct or Hugging Face artifact sources. Omitting them
preserves the historical all-one-backend behavior. Explicit values enable a
controlled browser comparison such as:

```ts
source: {
  kind: 'direct',
  encoderBackend: 'webgpu',
  decoderBackend: 'wasm',
  jointBackend: 'wasm',
  artifacts,
}
```

The executor initializes the runtime with WebGPU when any component requests
it, then creates each ORT session with its selected provider. Transcript
metrics expose the resolved component providers. This is a capability and a
measurement surface, not a default change. The existing Chrome real-WebGPU
harness was used to compare hybrid, all-WebGPU, and WASM controls with exact
text parity.

The first corrected Chrome matrix is captured in
`tools/data/results/gigaam/gigaam-rnnt-browser-component-matrix-2026-08-29.json`
on a real NVIDIA Blackwell adapter with ORT Web 1.29.0. All three compositions
were exact. The hybrid path (WebGPU encoder + WASM decoder/joiner) measured
1,910 ms / 5.92x RTFx, versus 4,638 ms / 2.44x for all-WebGPU and 5,948 ms /
1.90x for all-WASM. This is a strong placement signal, but it is one browser
session per composition; repeat runs and cross-browser confirmation are still
required before changing a preset default. Same-session repeats expose the
startup/steady-state split: hybrid runs were 744, 455, 481, 387, and 366 ms
(median 455 ms / 24.87x RTFx), while all-WebGPU repeats were 4,271, 4,370, and
4,177 ms (median 2.64x) and all-WASM repeats were 5,703, 5,744, and 5,117 ms
(median 1.98x). Every repeat remained exact. The first hybrid run is still a
cold dispatch/shader point; do not use it as the steady-state throughput
number.

## Reproduction

```powershell
npm run build
node tools/model-debugging/scripts/node-gigaam-rnnt-benchmark.mjs `
  --runs 3 --warmup 1 `
  --output docs/reports/gigaam-rnnt-hotpath-phase-profile-2026-08-29.json
```

The benchmark is artifact-gated and fails clearly when the official local
weights or reference capture are unavailable.
