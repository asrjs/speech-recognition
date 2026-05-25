# Workspace Context

This document is a short memory map for the local ASR workspace around `@asrjs/speech-recognition`. Use it when returning to the project after a break or when an agent needs to understand which sibling repo/app is the right place to inspect.

## Core Project

### `speech-recognition`

This is the new `asr.js` core library: a single-package, speech-first TypeScript runtime for browser and local Node.js inference.

The source of truth is here. Keep it headless, framework-neutral, ESM-first, and runtime-oriented. The public API is exported from `src/index.ts`; internal boundaries are preserved through `src/` folders such as `runtime`, `processors`, `inference`, `models`, `presets`, `tokenizers`, and `types`.

The core package may own reusable browser/runtime primitives such as capture, resampling, ring buffers, chunking, VAD, rough gate, detector/controller logic, waveform renderers, canvas utilities, and snapshot-style monitor APIs. React/Vue/Svelte/Solid wrappers should stay in thin sibling packages if they are needed.

## Implementation Apps

These apps exist because browser ASR behavior crosses many layers: microphone capture, chunking, model artifact loading, VAD, visualization, benchmarking, and realtime transcription. The library should remain reusable; the apps provide focused integration surfaces.

### `streaming-demo`

Current main realtime integration app.

Use this to test microphone streaming, waveform canvas rendering, rough gate behavior, TEN-VAD/FireRed VAD diagnostics, streaming ASR flows, local model folders, and Hugging Face model loading.

Current resume point:

- waveform canvas, VAD markers, and energy triggering are the active realtime surface
- rough-gate-first segmentation is the stable fallback path
- TEN-VAD can still be used as the lighter VAD path
- FireRed VAD is experimental and may be diagnostics-only depending on the branch/app wiring
- TEN-VAD historically used roughly 16 ms chunks, while FireRed VAD expects 10 ms frames; when visualizing or comparing them, align results to the shared timeline instead of assuming identical frame sizes

### `browser-demo`

General browser ASR demo and earlier integration surface.

Use it for upload/sample-file workflows, model loading UX, local-vs-remote artifact behavior, and regression checks that do not need the full realtime streaming UI.

### `benchmark-demo`

Performance and benchmark surface.

Use it for comparing model/runtime variants, measuring latency or throughput, and checking whether changes to loaders, executors, or processors affect runtime cost.

### `playground`

Scratch app for API and UI experiments.

Use it when exploring ideas before deciding whether they belong in the core library, a demo, or a thin framework adapter.

### `vad-demo`

VAD-focused implementation app.

Use it for isolated VAD behavior, timeline rendering, segment boundaries, and comparisons between rough gate, TEN-VAD, and FireRed-style signals.

### `firered-vad-web`

Experimental dedicated FireRed VAD browser port/library.

Use it to inspect FireRed-specific model loading, frame sizing, inference behavior, and diagnostics before moving stable, reusable primitives into `speech-recognition`.

## Reference Repos

These repos are useful for archaeology and comparison. They are not the architecture to copy directly into `speech-recognition`.

### `parakeet.js`

Older focused Parakeet TDT runtime.

Use it as a working reference for ONNX artifact loading, browser demo patterns, JavaScript audio preprocessing, decoder loops, and model-specific Parakeet behavior. It is narrower than `speech-recognition`, so copy lessons carefully rather than copying its structure wholesale.

### `transformers-v4-parakeet-demo`

Transformers.js v4 Parakeet TDT fork/demo.

Use it as a reference for Transformers.js-compatible model formatting, Hugging Face model management, and how Parakeet TDT can be represented inside a Transformers.js-style stack. Do not bring the generic Transformers.js architecture or public API philosophy into this repo.

### `medasr.js`

Older/sibling MedASR experiments.

Use it for MedASR-specific history, model artifact clues, and comparison against the newer preset/model-family organization in this library.

### `NeMo`

Upstream or research reference for NVIDIA NeMo model behavior.

Use it when checking architecture details, export assumptions, tokenizer behavior, or TDT model semantics.

### `onnx-asr`

ONNX ASR reference/conversion work.

Use it when checking ONNX file layout, external data conventions, browser-compatible exports, and older artifact naming patterns.

### `keet`

Experimental or older Parakeet-related scratch work.

Treat it as reference material until its current contents are rechecked.

## Current Direction

`speech-recognition` is meant to be the platform and model agnostic layer above the earlier focused experiments. The old projects are valuable because they contain working solutions and failure history, but this repo should stay:

- speech-first instead of model-zoo-first
- single-package instead of workspace-package-heavy
- headless and framework-neutral
- model-family based under `src/models`
- branded only at the preset layer under `src/presets`
- careful about keeping backend differences from changing transcript semantics

## Where To Look First

- Runtime/library bug: start in `speech-recognition`.
- Realtime microphone, waveform, VAD, and streaming behavior: start in `streaming-demo`.
- Remote Hugging Face or local-folder model loading behavior: compare `speech-recognition`, `streaming-demo`, `browser-demo`, `parakeet.js`, and `transformers-v4-parakeet-demo`.
- VAD frame-size or timeline alignment issue: compare `streaming-demo`, `vad-demo`, and `firered-vad-web`.
- ONNX external data, artifact naming, or Parakeet TDT loader issue: compare `speech-recognition`, `parakeet.js`, `onnx-asr`, and `transformers-v4-parakeet-demo`.

Remote model loading resilience is documented in [HUGGINGFACE_DOWNLOAD_RESILIENCE.md](./HUGGINGFACE_DOWNLOAD_RESILIENCE.md). FireRed diagnostic fallback behavior is documented in [FIRERED_VAD_DEGRADED_TROUBLESHOOTING.md](./FIRERED_VAD_DEGRADED_TROUBLESHOOTING.md).
