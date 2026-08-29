# ORT WebGPU Entrypoint Playbook

Use this playbook when a browser ONNX Runtime WebGPU session creates
successfully but produces unstable, corrupted, or unexpectedly slow model
outputs.

## Why this boundary matters

`onnxruntime-web` and `onnxruntime-web/webgpu` are distinct package entry
points. A bundler alias that maps both imports to `ort.all.bundle.min.mjs` can
cross the provider-registration boundary. In the Qwen3-ASR explicit-KV graph,
that substitution created a deterministic 13-token corrupted transcript on
ORT Web 1.29.0 while session creation still reported success. This was not a
model, tokenizer, or ONNX artifact failure.

## Required alias shape

Keep the imports explicit in browser harnesses and application bundlers:

```js
{
  find: 'onnxruntime-web/webgpu',
  replacement: ortWebgpuBundle,
},
{
  find: 'onnxruntime-web',
  replacement: ortAllBundle,
},
```

The WebGPU subpath must resolve to the WebGPU bundle. Use the all bundle only
for the plain import when a page also exercises WASM. Do not rely on a regex
alias that treats the subpath and package root as interchangeable.

## Minimal investigation

1. Record the exact ORT version, package entry URL, browser, adapter, model
   hashes, and fixture hash.
2. Run the failing alias and the separated-alias control in fresh browser
   profiles. Capture token IDs and final text, not only a pass/fail label.
3. Confirm that session creation, first inference, warm inference, and tensor
   disposal are all measured separately.
4. If the separated alias restores parity, preserve the harness correction and
   stop changing model code. Re-run the model-specific performance benchmark
   only after the correctness boundary is green.
5. Repeat on another browser or adapter before changing a public preset.

## Qwen3-ASR evidence

On 2026-08-29, ORT Web 1.29.0 with the official dynamic encoder and explicit-KV
decoder on NVIDIA Blackwell produced exact 30-token output with the separated
aliases. Three same-session GPU-KV runs had a 1,854.6 ms median (`5.93x` RTFx);
the CPU-KV control had a 3,880.65 ms median (`2.83x` RTFx). The machine-readable
result is
`docs/reports/qwen3-asr-webgpu-bundle-boundary-2026-08-29.json`.

This playbook captures a runtime-entry correction and a placement measurement;
it does not imply that every ORT bundle or every model has the same failure.
