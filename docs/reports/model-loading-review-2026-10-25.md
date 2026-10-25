# Model loading review — 2026-10-25

The library owns artifact selection, component backends, caching, fallback,
progress and cancellation. Applications select a model and expose user controls.
Framework bindings subscribe to the headless controller.

## Verified source paths

- The original TDT v2 demo and benchmark use an FP16 WebGPU encoder and INT8
  WASM decoder/joiner. The current catalog and executor retain this combination.
- All four maintained demos use the shared `demo-runtime` Hub loader, which
  delegates request construction and loading to the library. The realtime worker
  also uses the library builder. Local folder inspection delegates to library
  adapters. Playground's legacy Parakeet compatibility example intentionally
  demonstrates its older API.
- React, Svelte and Vue examples use `buildSpeechModelLoadOptions` and the shared
  controller. They do not independently choose artifact filenames or precision.
- Regression tests connect the public catalog builder to the actual TDT artifact
  resolver: hybrid selects FP16/INT8; WASM selects INT8/INT8; explicit precision
  and component backend overrides remain effective. An explicitly configured
  WebGPU encoder is preserved even if the runtime's primary backend is WASM.

## Changes

The TDT executor previously recorded successful precision fallback warnings only
in transcription output. Encoder and decoder fallback attempts now emit
`model-load:fallback` before another artifact is requested. The event includes
the attempted filename, backend, warning code and initialization error message.
Existing fallback policy and explicit precision behavior are unchanged.

`RuntimeProgressEvent.warning` is optional. The controller retains warnings in
`loadWarnings`, independently of transient download progress. A new load clears
them; superseded loads cannot publish into the current snapshot. This new
snapshot field is optional to preserve existing structural consumers. The three
framework examples display it without model-specific logic. Benchmark now also
accepts progress messages; its old file-only handler hid explanatory messages.

A warning describes an attempt, not a successful load. Only the final controller
status establishes readiness. Transcription still retains the existing warning
contract for a successful fallback.

## Validation

- Core build and source/example typecheck: passed.
- Full core tests: 2020 passed, 36 skipped; 237 passed files, 7 skipped files.
- Focused catalog/executor/controller tests: 29 passed.
- Root static import graph: 44 modules; requested limit remains below 45.
- Changed core files: lint has zero errors and one existing executor length warning.
- React/Svelte/Vue typecheck and bindings/suite tests: passed, 10 tests.
- Installed demo SDK: all 1711 dist hashes match the new archive's provenance.
- Four-demo suite/config tests: 22 passed; application tests: 71 passed.
- Four-demo and three-framework production builds: passed. Existing chunk-size
  and service-worker bundling warnings remain; no thresholds were relaxed.
- Chrome: all seven pages served HTTP 200 and rendered their main landmark.
  Six isolated framework/backend checks requested the expected encoder:
  FP16 for WebGPU hybrid and INT8 for WASM. Model requests were deliberately
  aborted before weights could download; the resulting fetch errors are expected.

Logs and browser receipts: [evidence](evidence/model-loading-2026-10-25/README.md).

Package: `asrjs-speech-recognition-0.2.0-load-warnings-20261025.tgz`.
SHA-256: `f3cd318879b1f56dbbe27228cbdd7f63e274a0f3bd5c7a11ba00350c558921d1`.
Previous archives are retained. See the adjacent provenance file in `demos/vendor`.

No new real-weight transcription, microphone, streaming, accuracy or performance
certification is claimed. The exact FP16 initialization error in the user's
earlier browser session remains unverified; the screenshot alone cannot identify
its cause. New events expose that cause for subsequent attempts.
