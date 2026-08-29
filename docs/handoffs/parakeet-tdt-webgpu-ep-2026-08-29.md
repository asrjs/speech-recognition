# Parakeet TDT v3 WebGPU EP probe — 2026-08-29

## Scope

This handoff records the controlled browser probe for the Parakeet TDT v3
`decoder_joint-model.onnx` GRU graph after upgrading `onnxruntime-web` from
the 1.27 nightly to stable 1.29.0. The probe used the existing
`N:\github\asrjs\webgpu-agent-test` Chrome headless harness, a real NVIDIA
Blackwell adapter, and the local model artifact at
`N:\models\onnx\nemo\parakeet-tdt-0.6b-v3-onnx`.

The graph emits `[1, 1, 1, 8198]`: 8193 tokenizer-vocabulary logits followed
by five TDT duration logits. Token selection must slice the first 8193 values;
the raw argmax is not a valid token ID for the next decoder step.

## Corrected A/B result

The first harness version had two evidence defects: spike mode imported the
full library before the direct ORT probe, and the spike logger called an
undefined `push` function. It also compared raw logits without excluding the
duration head. Those defects were fixed before accepting the measurements;
the runner now recognizes the two per-EP records emitted by spike mode.

Three corrected runs per ORT version/backend completed with finite outputs and
the same vocabulary-sliced token sequence (`8192` for all five steps):

| ORT Web      | EP     | Session-load median | Warm-step mean range | Result   |
| ------------ | ------ | ------------------: | -------------------: | -------- |
| 1.27 nightly | WebGPU |              1.41 s |         14.0–32.2 ms | 3/3 pass |
| 1.27 nightly | WASM   |              0.63 s |           5.5–5.8 ms | 3/3 pass |
| 1.29.0       | WebGPU |              1.46 s |          9.9–14.9 ms | 3/3 pass |
| 1.29.0       | WASM   |              0.68 s |           5.5–6.5 ms | 3/3 pass |

The earlier “1.29 session-create hang” was therefore a false classification;
the corrected graph loads and runs on both browser execution providers.
The production default remains encoder-WebGPU/decoder-WASM because the full
model has not yet established a transcript-quality and lifecycle win for a
WebGPU decoder.

## GPU-state experiment

An opt-in direct ORT run set
`preferredOutputLocation: { outputs: 'cpu', output_states_1: 'gpu-buffer',
output_states_2: 'gpu-buffer' }`. ORT 1.29 reported both recurrent state
outputs as `gpu-buffer` on every step, and the vocabulary-sliced sequence
remained identical. Three WebGPU repetitions produced warm-step means of
9.01, 8.26, and 7.78 ms. Comparable CPU-state runs measured 12.73 and
13.14 ms, so the decoder-only microbenchmark suggests roughly 29–41% lower
warm-step time.

This is a hypothesis, not a promoted library path. The experiment does not
include the FastConformer encoder, real transcript quality, repeated model
transcriptions, or safe disposal of recurrent GPU tensors. In particular,
the initial version that disposed GPU state handles surfaced ORT's `null
function` failure; the spike was rerun without per-step disposal only to
measure execution. A production implementation must prove ownership and
disposal (or a bounded fallback) before enabling GPU-state retention.

## Decode hot-path optimization

The library decoder consumed each float32 logits tensor synchronously before
disposing the ORT output, but still allocated a full `Float32Array` copy for
every step. The TDT path now borrows the typed-array view and only normalizes
non-float32 views. In the same change, `confidenceFromLogits` no longer runs
the entropy traversal that belongs to `tokenQualityFromLogits`; it retains the
same max/softmax/log-probability arithmetic while doing only the work its
callers request. TDT model/session disposal also removes redundant async
wrappers.

The reproducible `benchmark:hot-paths` harness now includes an 8,193-entry
TDT-vocabulary scenario. On Node v26.2.0 / win32-x64 with 10,000 iterations,
the confidence-only path measured `0.0849 ms` p50 versus `0.3970 ms` p50 for
the full entropy path (about 79% lower). Borrowing the logits view measured
`0.0928 ms` p50 versus `0.1114 ms` p50 for the copy control (about 17% lower).
These are post-processing microbenchmarks, not end-to-end RTFx claims; the
existing executor tests and full suite remain the correctness gate.

## Remaining work

1. Compare warm-cache full-model fp16 against the available fp32 and int8
   encoder artifacts; the 1.24 GB fp16 FastConformer graph previously spent
   more than ten minutes in headless Chrome session creation.
2. Add an opt-in, lifecycle-safe TDT GPU-state path only after end-to-end
   transcript parity, repeated-transcription memory checks, and disposal
   behavior are measured.
3. Keep backend placement workload-specific: the direct spike is faster on
   WASM for this one-frame GRU workload, while GPU-state retention is a
   promising optimization for larger decode loops.

## Reproduction

Start Vite in `N:\github\asrjs\webgpu-agent-test` with the repository's ORT
package aliased by `vite.config.js`, then run:

```text
npm run dev -- --force
node scripts/run-parakeet-tdt-webgpu.mjs --mode=spike
node scripts/run-parakeet-tdt-webgpu.mjs --mode=spike --gpu-state
```

The browser runner uses `--enable-unsafe-webgpu`, Vulkan/D3D11 ANGLE flags,
and writes JSON evidence to `webgpu-agent-test/_results/`.
