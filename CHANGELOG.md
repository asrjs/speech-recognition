# Changelog

## 0.3.0

First npm release.

### Added

- Approved model profiles: a versioned manifest registry and planner that pick
  component backends and precision from probed runtime capabilities.
  Parakeet TDT 0.6B v3 is the first approved profile (FP16 encoder on WebGPU,
  INT8 decoder on WASM, pinned artifact revision). New root APIs:
  `listApprovedSpeechModelManifests`, `resolveSpeechModelExecutionPlan` and
  `probeSpeechModelExecutionPlan`.
- Fallback warnings: when a device cannot run the approved profile, the plan
  lists the departure and the loader emits a `model-load:fallback` event, which
  `createSpeechController()` exposes in `loadWarnings`.
- Pinned Hugging Face bundles for Qwen3-ASR 0.6B and Moonshine Streaming Tiny
  (experimental).
- `setSpeechDiagnosticsLogger()` in `/io` routes or silences recoverable
  library warnings.
- `resampleSinc` and `SincResampler` in `/browser` and `/browser/media`.

### Fixed

- Microphone capture decimated from the device rate with linear interpolation
  and no anti-aliasing filter. Capture and model-rate conversion now use a
  band-limited resampler.
- Large model files no longer fail with `ERR_BLOB_OUT_OF_MEMORY` when
  materialized as blob URLs.
- Cached NeMo models load without network access; an unreachable optional
  external-data sidecar is treated as absent.
- An explicit `backend` no longer drops the pinned model revision.
- The Whisper production pipeline read the wrong word fields and threw on real
  executor output.

### Breaking

- `package.json` exports list model-family and preset entry points
  explicitly. Deep imports of internal files such as
  `/models/nemo-tdt/executor` or `/presets/parakeet/manifest` now fail with
  `ERR_PACKAGE_PATH_NOT_EXPORTED`. Import from the family entry, the preset
  entry or `/presets`.
