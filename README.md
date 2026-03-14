# asr.js

`asr.js` is a single-package, speech-first TypeScript runtime library.

It keeps strong internal separation for processors, runtime orchestration, inference, model integrations, and presets, but publishes one library package with one root public API.

## Structure

```text
src/
  inference/
  models/
  presets/
  processors/
  runtime/
  tokenizers/
  types/
docs/
examples/
tests/
```

`src/inference/graph.ts` is the shared descriptor layer for encoder families, decoder-head families, and decoding strategies. Real execution code stays in `src/models/*` until it is proven reusable across multiple families.

## Usage

```ts
import { createSpeechRuntime, createWasmBackend } from 'asr.js';
```

## Realtime Helpers

`asr.js` also exposes small app-facing helpers for realtime and long-form speech apps:

- `AudioRingBuffer`
- `VoiceActivityTimeline`
- `StreamingWindowBuilder`
- `UtteranceTranscriptMerger`
- `RealtimeTranscriptionController`
- `startMicrophoneCapture`
- `AudioChunker`
- `LayeredAudioBuffer`
- `AudioFeatureCache`

It also includes lightweight helpers for benchmark and evaluation apps:

- `summarizeNumericSeries`
- `benchmarkRunRecordsToCsv`
- `fetchDatasetSplits`
- `fetchSequentialRows`
- `fetchRandomRows`
- `decodeAudioSourceToMonoPcm`

For transcript handling across different model families, `asr.js` supports:

- canonical transcript results via `TranscriptResult`
- streaming updates via `PartialTranscript`
- native-only responses via `responseFlavor: 'native'`
- dual canonical + native envelopes via `TranscriptionEnvelope<TNative>`
- model-output normalizers such as `normalizeNemoTdtTranscript`, `normalizeWhisperTranscript`, and `normalizeLegacyParakeetTranscript`

## Commands

```bash
npm install
npm run build
npm run typecheck
npm test
```

## Status

- canonical transcript contracts: implemented
- runtime and backend registration: implemented
- processor scaffolding: implemented
- inference scaffolding: implemented
- tokenizer abstractions and stubs: implemented
- architecture-based model scaffolds: implemented
- branded preset layers: implemented
- current implementation families: NeMo TDT, HF CTC, Whisper seq2seq, FireRed speech-LLM
- production inference executors: not implemented yet
