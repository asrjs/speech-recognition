# `@asrjs/speech-recognition` Architecture

## System Overview

`@asrjs/speech-recognition` is a single-package speech runtime library with one build output and one release flow, but a layered public API.

It is designed around:

- explicit runtime composition
- architecture-based model families
- thin branded presets
- canonical transcript contracts
- injectable asset loading and caching
- browser and Node.js support without environment leakage

The library is intentionally speech-first. It is not a generic multimodal framework and it does not mirror generic task-pipeline architecture.

## Internal Layout

```text
src/
  audio/
  inference/
  io/
  models/
  presets/
  runtime/
  tokenizers/
  types/
```

## Dependency Direction

The intended dependency direction is:

`types -> audio / tokenizers / io / runtime -> inference -> models -> presets`

Rules:

- `src/types` defines stable contracts and should stay environment-agnostic.
- `src/runtime` owns orchestration, registration, lifecycle, backend selection, and transcript normalization.
- `src/io` owns asset resolution and caching, not model-specific execution.
- `src/inference` owns shared descriptors, generic math, and shared streaming primitives only.
- `src/models/*` owns family-specific execution, decode loops, tokenizer behavior, timestamp reconstruction, and native output shaping.
- `src/presets/*` resolves branded presets into technical family load requests without owning execution logic.

## Folder Responsibilities

### `src/types`

Owns stable contracts:

- transcript and streaming result types
- backend capability types
- runtime, model, session, and preset interfaces
- model classification and architecture metadata
- IO contracts such as `AssetRequest`, `AssetProvider`, and `ResolvedAssetHandle`

### `src/audio`

Owns reusable audio and feature-preprocessing primitives:

- PCM normalization
- mono/channel handling
- sample-rate helpers
- chunking and framing
- audio containers such as `PcmAudioBuffer`

### `src/io`

Owns the environment boundary for model assets.

This layer exists so model families do not hardcode:

- `fetch()`
- `URL.createObjectURL()`
- direct filesystem access
- IndexedDB access

Core contracts:

- `AssetRequest`
- `ResolvedAssetHandle`
- `AssetProvider`
- `AssetCache`

`ResolvedAssetHandle` is stream-first:

- `openStream()` is the primary large-asset path
- `readBytes()` is convenience for eager consumers
- `readText()` and `readJson()` build on the same handle
- `getLocator('url' | 'path')` supports runtimes like ORT that still need a concrete locator
- `dispose()` cleans up temporary URLs, paths, or cache-owned resources

Built-in providers and caches currently cover:

- browser fetch
- Hugging Face hub resolution
- browser local file/directory handles
- IndexedDB caching
- Node filesystem paths

### `src/runtime`

Owns orchestration, not model intelligence.

Responsibilities:

- backend registration
- model-family registration
- preset registration
- model loading
- backend capability probing and selection
- runtime/model/session lifecycle
- errors, logging, hooks, and transcript normalization

The runtime surface now distinguishes technical model families from branded presets:

- `registerModelFamily(...)`
- `registerPreset(...)`
- `listModelFamilies()`
- `listPresets()`

Load requests are intentionally discriminated:

- family path:
  - `{ family, modelId, ... }`
- preset path:
  - `{ preset, modelId?, ... }`

Exactly one of `family` or `preset` should be present.

### `src/inference`

Owns only shared inference building blocks that are truly model-agnostic:

- architecture descriptors in `src/inference/descriptors.ts`
- generic math in `src/inference/math.ts`
- shared streaming orchestration in `src/inference/streaming/*`
- backend packages and probes in `src/inference/backends/*`

This folder must not become a catch-all for:

- backend policy
- runtime orchestration
- model-family quirks
- tokenizer-specific behavior
- topology-specific decode loops

If shared descriptors become too broad later, the next split should be within `src/inference`, not by pushing model-owned logic upward prematurely.

### `src/models`

Owns architecture-based implementation families such as:

- `nemo-tdt`
- `hf-ctc-common`
- `whisper-seq2seq`
- `firered-llm`

Family packages own:

- config parsing
- topology-specific preprocessing glue
- tensor/session wiring
- decode loops
- timestamp and confidence reconstruction
- native output shaping

Shared helpers move upward only after reuse is proven across multiple families.

### `src/presets`

Owns branded presets such as:

- `parakeet`
- `medasr`
- `whisper`

Presets are intentionally thin:

- they resolve branded names into technical family requests
- they may inject default config, model ids, classification metadata, and asset hints
- they may apply branded runtime or decoding defaults

Presets must not own:

- execution code
- backend implementations
- model-family decode loops

## Model Families vs Presets

This distinction is a core design rule.

Examples:

- technical family:
  - `nemo-tdt`
- branded preset:
  - `parakeet`

When a model is loaded through a preset:

- `model.info.family` stays technical
- `model.info.preset` records the branded alias

This lets sibling branded families reuse the same implementation family without duplicating execution code.

## Runtime Lifecycle and Disposal

`dispose()` is the primary lifecycle contract.

Required rules:

- `SpeechSession.dispose()` is required and idempotent
- `SpeechModel.dispose()` is required and idempotent
- `SpeechRuntime.dispose()` is required and idempotent
- `ResolvedAssetHandle.dispose()` is required
- backend execution resources must be released through the same lifecycle chain

Ownership rules:

- models track and dispose sessions they create
- runtimes track and dispose models they load
- asset handles clean up temporary locators and owned resources

`Symbol.asyncDispose` may be implemented as optional sugar, but it is not the primary or required API.

## Canonical Transcript Boundary

`@asrjs/speech-recognition` keeps transcript semantics stable across model families and backends.

Canonical contracts:

- `TranscriptResult`
- `PartialTranscript`
- `TranscriptionEnvelope<TNative>`

Rules:

- canonical transcript objects must remain structured-clone-safe POJOs
- canonical outputs must not contain classes, methods, `Map`, `Set`, browser handles, backend references, or session references
- native outputs are not guaranteed worker-safe unless documented by the family

Streaming state currently uses:

- `revision`
- `committedText`
- `previewText`

This supports partial/final updates and retroactive correction workflows without locking the library into a more opinionated cursor protocol too early.

## Public API Shape

`@asrjs/speech-recognition` uses one npm package with intentional subpath exports.

### Root: `@asrjs/speech-recognition`

Keep the root entry narrow and runtime-critical:

- transcript/runtime/backend/audio contracts
- `createSpeechRuntime`
- canonical transcript normalizers
- backend factories
- `PcmAudioBuffer`

### Secondary subpaths

- `@asrjs/speech-recognition/builtins`
  - built-in family/preset registration and convenience runtime composition
- `@asrjs/speech-recognition/io`
  - asset providers, caches, and model-asset helpers
- `@asrjs/speech-recognition/inference`
  - descriptors, generic math, and shared streaming primitives
- `@asrjs/speech-recognition/browser`
  - browser-only helpers such as microphone capture and audio decoding
- `@asrjs/speech-recognition/realtime`
  - app-facing live transcription helpers
- `@asrjs/speech-recognition/bench`
  - benchmark and evaluation utilities
- `@asrjs/speech-recognition/datasets`
  - dataset helpers
- `@asrjs/speech-recognition/models/*`
  - technical model-family entry points
- `@asrjs/speech-recognition/presets/*`
  - branded preset entry points

Subpath separation is the primary bundling boundary. Lazy imports inside built-in helpers may be added later, but they are secondary to clean public entrypoint separation.

## Realtime and Browser Helpers

`@asrjs/speech-recognition` includes library-grade helpers extracted from real apps:

- `AudioRingBuffer`
- `LayeredAudioBuffer`
- `VoiceActivityTimeline`
- `StreamingWindowBuilder`
- `UtteranceTranscriptMerger`
- `RealtimeTranscriptionController`
- `startMicrophoneCapture`
- `decodeAudioSourceToMonoPcm`

These are intentionally separated from the root API because they are higher-level application helpers, not the minimal runtime core.

## Built-In Composition vs Explicit Composition

`createBuiltInSpeechRuntime()` is a convenience entrypoint.

The stricter, more explicit architecture path is:

1. create a runtime
2. register the backends you want
3. register the model families you want
4. register the presets you want

This keeps tree-shaking and ownership clearer for serious applications, while still allowing a batteries-included convenience path.

## Why This Architecture Is Speech-First

This architecture is intentionally optimized for speech workloads:

- it is a speech runtime, not a general task-pipeline framework
- it keeps canonical transcript contracts as the stable app-facing boundary
- it keeps branded families downstream as presets over technical implementations
- it separates asset loading from model execution through an explicit IO layer
- it keeps runtime orchestration distinct from model-family execution
- it keeps backend choice from changing transcript semantics

## Architecture Rules To Keep

1. Runtime is orchestration, not intelligence.
2. Model families own execution and decoding.
3. Presets stay thin and branded.
4. Canonical transcripts are the stable app-facing contract.
5. Browser-only code must stay off the root import path.
6. Shared model logic moves upward only after reuse is proven.
7. `src/inference` must stay descriptive and generic, not policy-heavy.
