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
