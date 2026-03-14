# `asr.js` Architecture

## System Overview

`asr.js` is a single-package speech runtime library built around one public API, one build output, and one release flow.

The internal codebase preserves strong separation, but those boundaries live inside `src/` folders rather than workspace packages. The library is organized around a speech-native inference stack and architecture-based model integrations, while branded families remain presets layered on top.

The dependency direction is:
`types -> processors / runtime / tokenizers -> inference -> models -> presets`

This keeps the repo simple to build and publish while preserving technical boundaries that matter for maintainability and future model support.

## Internal Layout

```text
src/
  inference/
  models/
  presets/
  processors/
  runtime/
  tokenizers/
  types/
```

### Folder responsibilities

- `src/types`
  - Canonical transcript types, backend capability types, runtime/model/session contracts, and classification metadata.
- `src/runtime`
  - Runtime registration, lifecycle, backend selection, errors, logging hooks, model loading, browser-audio helpers, transcript normalization helpers, and app-facing realtime helpers such as ring buffers, window builders, and utterance mergers.
- `src/processors`
  - PCM normalization, chunking, channel handling, and processor descriptors.
- `src/tokenizers`
  - Tokenization contracts, tokenizer stubs, and text reconstruction helpers.
- `src/inference`
  - Backend probes, streaming orchestration, and shared inference graph descriptors for encoders, decoder heads, and decoding strategies.
- `src/models`
  - Architecture-based model integrations such as NeMo TDT, HF CTC, Whisper seq2seq, and FireRed speech-LLM.
- `src/presets`
  - Branded model-family manifests and thin preset factories such as Parakeet, MedASR, and Whisper.

## Model Classification

Models are classified by implementation-relevant traits rather than marketing names.

```ts
interface ModelClassification {
  readonly ecosystem: string;
  readonly processor?: string;
  readonly encoder?: string;
  readonly decoder?: string;
  readonly topology?: string;
  readonly family?: string;
  readonly task: string;
}
```

Implementation code lives under `src/models`, while branded aliases live under `src/presets`. This keeps reusable architecture logic separate from family naming.

## Shared vs Model-Owned Logic

- `src/inference/graph.ts`
  - Owns shared classification for processor, encoder, decoder-head, and decoding-strategy families.
- `src/models/*`
  - Owns concrete execution logic, family-specific tensor wiring, and any decoder loop that is not yet genuinely reusable.
- `src/tokenizers/*`
  - Owns tokenizer contracts and implementations directly. Tokenizers are not nested under `inference`.

## Realtime App Helpers

`asr.js` intentionally includes a small helper layer for real apps, not just raw model loading.

- `AudioRingBuffer`
  - Fixed-size PCM ring buffer with absolute frame addressing and overwrite protection.
- `VoiceActivityTimeline`
  - Ring-buffered speech/silence timeline for lightweight VAD-aware windowing and silence-tail checks.
- `StreamingWindowBuilder`
  - Cursor-aware window construction for long-running and low-latency transcription loops.
- `UtteranceTranscriptMerger`
  - Mature/pending transcript management for direct model outputs with punctuation-based finalization.
- `RealtimeTranscriptionController`
  - Thin coordinator for audio ring buffering, optional VAD tracking, window building, and transcript merging.
- `startMicrophoneCapture`
  - Browser microphone helper that emits PCM chunks without taking over UI state or worker orchestration.
- `decodeAudioSourceToMonoPcm`
  - Shared browser helper for fetching and decoding audio sources into mono PCM before inference.

These helpers are library-grade building blocks extracted from real application needs, but they stop short of pulling UI, worker orchestration, or app-specific state management into the core library.

This split is deliberate: `asr.js` shares semantics and descriptors early, but only extracts executable math after multiple model families prove the code is truly shared.

## Public API

`asr.js` exposes one root public API from `src/index.ts`.

Consumers import from:

```ts
import { createSpeechRuntime, createWasmBackend } from 'asr.js';
```

The library does not publish segmented workspace-style packages like `@asrjs/core` or `@asrjs/model-nemo-tdt`.

## Transcript Output Model

`asr.js` keeps transcript semantics stable across model families and backends.

- `TranscriptResult`
  - canonical app-facing JSON with `text`, `warnings`, `meta`, and optional `segments`, `words`, and `tokens`
- `PartialTranscript`
  - streaming update shape for partial/final transcript state
- `TranscriptionEnvelope<TNative>`
  - canonical transcript plus attached model-native output for power users
- `responseFlavor: 'canonical' | 'canonical+native' | 'native'`
  - lets apps choose whether they want stable canonical output, model-native output, or both

`src/runtime/transcripts.ts` provides normalizers so consumers can route different native outputs such as NeMo TDT, HF CTC, Whisper, or legacy Parakeet JSON into the same canonical transcript workflow.

## Metadata and Stability Rules

- Canonical transcript semantics remain stable across all backends.
- Backend choice may affect performance and availability, but not transcript meaning.
- `ModelArchitectureDescriptor.module` and `sharedModule` identify internal subsystems such as `processors` and `inference`, not publishable package names.
- Decoder internals remain model-specific until real shared behavior is proven.

## Why This Differs From `transformers.js`

This architecture is intentionally different from `transformers.js`:

- it is a speech runtime, not a general model framework
- it preserves architecture-based model boundaries instead of centering everything on generic task pipelines
- it keeps one coherent library API instead of multiple pseudo-packages
- it treats branded families as presets over technical implementations
- it keeps backend policy capability-based without letting backend choice leak into output semantics
