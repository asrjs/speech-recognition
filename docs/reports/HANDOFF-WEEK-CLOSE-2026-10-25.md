# Week close — 2026-10-25

The library owns model loading, audio decoding/downmix/resampling, model-aware
long-audio windowing/merge, capture, cancellation and cleanup. Framework bindings
stay thin and outside core. This checkpoint preserves the week's runtime,
model-tooling, demo and documentation work; historical failures remain evidence.

## Verified scope

- Core: 2,050 tests passed, 36 skipped; build and source/example type checks passed.
- Root import graph: 44 modules, with the existing 45-module ceiling unchanged.
- Full lint: zero errors, 16 warnings. API docs: zero errors, 42 warnings.
- Four demos: 93 tests and four production builds passed.
- React/Svelte/Vue workspace: 10 tests, adapter/example type checks and builds passed.
- Seven apps served at http://127.0.0.1:4190/ in the existing local suite.
- Current demo SDK: vendor/asrjs-speech-recognition-0.2.0-microphone-context-20261025.tgz;
  SHA-256 4520cd899fa8d4d1695f947be37f1cde6dfc9213728d938576bebdc0e6f920a8.

See [microphone receipts](microphone-composition-and-reference-review-2026-10-25.md)
for the final real-weight Parakeet TDT v3 file/manual/TEN-VAD runs. The controlled
Chrome microphone uses prerecorded audio. It does not certify physical microphone
hardware, every catalog model, or numerical parity of every model stage.
All model/backend combinations without receipts remain unverified. CI must be
checked separately after pushing; local checks do not establish remote CI success.

## Resume

Framework examples are published in asrjs/demos on branch framework-examples,
as a standalone sibling checkout named frameworks. Keep speech-recognition,
demos and frameworks beside one another. Follow the workspace README commands.
Do not restart model downloads or return to non-Turbo Whisper Large/EOU120.

Next implementation gates are TDT tail rescan and decoder-state carry, with
reference tests before behavior changes. Acoustic context is already retained
by the shared microphone composition; that does not establish causal encoder
streaming. Review existing receipts before claiming a new model acceptance.

## Next loading milestone — approved artifact profiles

User direction: offer ysdede-hosted HF bundles for our approved exports and let
core select the model's execution profile automatically. Demo source choices
currently follow loading.supportsHubSource; false does not establish that no
upstream HF repository exists. Qwen, for example, has a catalog repoId while
its registered loading path remains local-only. No HF upload was performed.

Existing descriptors already provide per-component backend/precision defaults,
and the shared demo policy consumes them. Parakeet TDT defaults to FP16 encoder
on WebGPU and INT8 decoder/joint on WASM. These declarations are capability
metadata, not evidence that every cross-product combination passed real weights.

Implement next with an explicit, versioned artifact manifest: repo + immutable
revision, graph roles/paths and external data, tokenizer/config/processor files,
precision per graph, graph I/O contract, compatible SDK/adapter version and
validated execution profiles linked to receipts/artifact hashes. Config and
ONNX metadata may identify the graph; backend quality/support still requires
runtime probes and real-weight acceptance. Avoid inferring approval from a
filename or dtype. Do not assume every family has only encoder and decoder.

Core should intersect approved profiles with device/runtime capabilities,
select the preferred profile, then reuse the existing loader/cache. Keep an
explicit advanced override, with unverified combinations marked as such; never
silently switch precision or claim an unavailable fallback. Apps should consume
this resolved plan rather than reproduce policy. Prepare a per-model publication
inventory and review exact approved files, licenses and destination before HF
publication. This checkpoint records the design; it does not claim the manifest
resolver or HF bundles have been shipped.

## Final merge receipts — 2026-10-25

- Core PR #239 merged: 12dd390adb76d2aaa802020c3b2285cae6f5ff07.
  Exact-head CI run 40222763700 completed successfully before merge.
- Demos PR #2 merged: 5cbdcb4543b86faa734db4b1d48c101f5957ce39.
  Initial demo head had no workflows/check runs: UNVERIFIED, not PASS.
  Added CI in 3d0a211e849f11109df4d52261d97ae5d621b85d; run
  40223392075 passed npm ci, all 93 tests and the four-app production build.
- Framework branch demos/framework-examples remains independent, at 008d11a.
- Main branches now contain the checkpoint implementation; framework setup may
  use main for core/demos. Historical checkpoint branches remain reproducible.
- These are CI/build/test results, not blanket model accuracy acceptance.
  Real-weight and physical microphone limitations above remain unchanged.

PRs: https://github.com/asrjs/speech-recognition/pull/239 and
https://github.com/asrjs/demos/pull/2.
