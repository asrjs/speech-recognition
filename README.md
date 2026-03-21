# @asrjs/speech-recognition

`@asrjs/speech-recognition` is a speech-first TypeScript runtime for browser and local Node.js inference.

It ships as one package, but with intentional subpath entry points so the root API stays small and environment-safe. The library is built around:

- explicit runtime composition
- architecture-based model families
- thin branded presets
- canonical transcript contracts
- reusable realtime/browser helpers
- injectable asset loading and caching

This is intentionally not a generic task-pipeline model framework.

## Install

```bash
npm install @asrjs/speech-recognition
```

## Development

Useful library-level workflows:

```bash
npm run typecheck
npm run lint
npm run format:check
npm test
npm run build
npm run docs:build
```

`docs:build` generates API docs from the library JSDoc into `.typedoc-site/`.

Source-structure guardrails:

- the ESLint config warns when a source or test module grows past `800` lines
- preset-specific logic should stay behind preset/model adapters instead of leaking into runtime-common modules

## Public API

### `@asrjs/speech-recognition`

Use the root entry for runtime-critical surfaces only:

- transcript/runtime/backend/audio contracts
- built-in model discovery helpers
- preset-aware load/transcription option builders
- `createSpeechRuntime`
- `listSpeechModels`
- `getSpeechModelDescriptor`
- `buildSpeechModelLoadOptions`
- `buildSpeechTranscriptionOptions`
- `loadSpeechModel`
- `transcribeSpeech`
- `transcribeSpeechFromMonoPcm`
- `createSpeechPipeline`
- backend factories
- canonical transcript normalizers
- `PcmAudioBuffer`

```ts
import {
  createSpeechRuntime,
  createWasmBackend,
  listSpeechModels,
  buildSpeechModelLoadOptions,
  loadSpeechModel,
  transcribeSpeech,
  transcribeSpeechFromMonoPcm,
  createSpeechPipeline,
  PcmAudioBuffer,
  getCanonicalTranscript,
} from '@asrjs/speech-recognition';
```

#### Few-line quick start

If you want one obvious import path for discovery, loading, and inference, start here:

```ts
import {
  buildSpeechModelLoadOptions,
  listSpeechModels,
  buildSpeechTranscriptionOptions,
  loadSpeechModel,
} from '@asrjs/speech-recognition';

const modelId = listSpeechModels().find((model) => model.preset === 'parakeet')?.modelId;
if (!modelId) {
  throw new Error('No built-in Parakeet model is available.');
}

const loaded = await loadSpeechModel(buildSpeechModelLoadOptions({
  modelId,
  backend: 'webgpu-hybrid',
}));

const result = await loaded.transcribeMonoPcm(pcm, 16000, {
  responseFlavor: 'canonical',
  ...buildSpeechTranscriptionOptions(modelId, {
    timestamps: true,
  }),
});

console.log(result.text);
await loaded.dispose();
```

#### High-level model loading

If you want the library to handle built-in runtime creation, preset/family
resolution, model loading, and ready-session creation for you, start from
`loadSpeechModel`:

```ts
import { loadSpeechModel } from '@asrjs/speech-recognition';

const loaded = await loadSpeechModel({
  modelId: 'parakeet-tdt-0.6b-v2',
  onProgress(event) {
    console.log(event.phase, event.loaded, event.total);
  },
});

const result = await loaded.transcribeMonoPcm(pcm, 16000, {
  responseFlavor: 'canonical+native',
  onProgress(event) {
    console.log(event.stage, event.progress, event.remainingMs, event.metrics);
  },
});

await loaded.dispose();
```

#### One-shot automatic transcription

If you want a single call that handles model loading, transcription, and
cleanup automatically, use `transcribeSpeechFromMonoPcm`:

```ts
import { transcribeSpeechFromMonoPcm } from '@asrjs/speech-recognition';

const canonical = await transcribeSpeechFromMonoPcm(pcm, 16000, {
  modelId: 'google/medasr',
  backend: 'wasm',
  transcribeOptions: {
    responseFlavor: 'canonical',
    detail: 'detailed',
  },
});

console.log(canonical.text);
```

`transcribeSpeech` still accepts `AudioInputLike` directly when you already have
an `AudioBufferLike` or want the default 16 kHz mono PCM shorthand.

#### Cached model-agnostic pipeline

If you need repeated transcriptions across one or more model families, use
`createSpeechPipeline` to reuse loaded models automatically:

```ts
import { createSpeechPipeline } from '@asrjs/speech-recognition';

const pipeline = createSpeechPipeline({
  cacheModels: true,
});

const med = await pipeline.transcribeMonoPcm(pcm, 16000, {
  modelId: 'google/medasr',
  backend: 'wasm',
});

const parakeet = await pipeline.transcribeMonoPcm(pcm, 16000, {
  modelId: 'parakeet-tdt-0.6b-v3',
  backend: 'webgpu-hybrid',
});

console.log(med.text, parakeet.text);
await pipeline.dispose();
```

### `@asrjs/speech-recognition/builtins`

Use the builtins entry for convenience composition:

```ts
import { createBuiltInSpeechRuntime, loadBuiltInSpeechModel } from '@asrjs/speech-recognition/builtins';
```

Use `loadBuiltInSpeechModel` when you want the same convenience path but prefer
to stay explicit about the built-ins namespace:

```ts
const loaded = await loadBuiltInSpeechModel({
  modelId: 'parakeet-tdt-0.6b-v2',
  onProgress(event) {
    console.log(event.phase, event.loaded, event.total);
  },
});

const result = await loaded.transcribe(audioBuffer, {
  responseFlavor: 'canonical+native',
  onProgress(event) {
    console.log(event.stage, event.progress, event.remainingMs, event.metrics);
  },
});

await loaded.dispose();
```

For built-in Parakeet TDT presets, the library now chooses optimized weights by
default:

- WebGPU: `fp16` encoder, `int8` decoder
- automatic encoder fallback to `fp32` when `fp16` cannot initialize
- WASM: `int8` encoder, `int8` decoder

Advanced callers can still override these defaults explicitly:

- preset helper path: pass `encoderQuant` / `decoderQuant`
- direct runtime path: pass `options.source.encoderQuant` / `options.source.decoderQuant`
- direct artifact path: create the model with your own chosen files or URLs

### `@asrjs/speech-recognition/models/*`

Use model-family entry points for technical implementations:

```ts
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';
import { createLasrCtcModelFamily } from '@asrjs/speech-recognition/models/lasr-ctc';
import { createWhisperSeq2SeqModelFamily } from '@asrjs/speech-recognition/models/whisper-seq2seq';
```

### `@asrjs/speech-recognition/presets/*`

Use preset entry points for branded model families and convenience loaders:

```ts
import { createParakeetPresetFactory } from '@asrjs/speech-recognition/presets/parakeet';
import { createMedAsrPresetFactory } from '@asrjs/speech-recognition/presets/medasr';
import { createWhisperPresetFactory } from '@asrjs/speech-recognition/presets/whisper';
```

Most app code no longer needs this entry just to discover models or build
preset-aware load/transcription options. Those common helpers are now available
from the root `@asrjs/speech-recognition` import.

Parakeet-specific helper loaders also live under its preset entry:

```ts
import {
  getParakeetModel,
  loadParakeetModelWithFallback,
  loadParakeetModelFromLocalEntries,
} from '@asrjs/speech-recognition/presets/parakeet';
```

### `@asrjs/speech-recognition/io`

Use the IO entry for asset providers, cache implementations, and Hugging Face/file helpers:

```ts
import {
  createDefaultAssetProvider,
  createHuggingFaceAssetProvider,
  fetchModelFiles,
  getModelFile,
  pickPreferredQuant,
} from '@asrjs/speech-recognition/io';
```

### `@asrjs/speech-recognition/inference`

Use the inference entry for shared descriptors, generic math, and shared streaming primitives:

```ts
import { FASTCONFORMER_ENCODER, argmax, DefaultStreamingTranscriber } from '@asrjs/speech-recognition/inference';
```

### Other subpaths

```ts
import { startMicrophoneCapture } from '@asrjs/speech-recognition/browser';
import { RealtimeTranscriptionController } from '@asrjs/speech-recognition/realtime';
import { benchmarkRunRecordsToCsv } from '@asrjs/speech-recognition/bench';
import { fetchDatasetSplits } from '@asrjs/speech-recognition/datasets';
```

The browser entry also owns browser-only local model helpers such as:

```ts
import {
  createSpeechModelLocalEntries,
  collectSpeechModelLocalEntries,
  inspectSpeechModelLocalEntries,
  loadSpeechModelFromLocalEntries,
  createBrowserRealtimeStarter,
  TenVadAdapter,
  encodeMonoPcmToWavBlob,
} from '@asrjs/speech-recognition/browser';
```

`createBrowserRealtimeStarter` is the browser-side convenience starter that
bundles `TenVadAdapter`, `VoiceActivityProbabilityBuffer`, and
`RealtimeTranscriptionController` into one setup flow for realtime demos.
Waveform history is derived from `ringBufferDurationMs` and the realtime
timeline chunk size; there is no separate waveform point-count knob for the
primary chunk-aligned view.

## Quick Start

### Explicit runtime composition

This is the preferred path when you want clear ownership and good bundle boundaries.

```ts
import { createSpeechRuntime, createWasmBackend, PcmAudioBuffer } from '@asrjs/speech-recognition';
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';
import { createParakeetPresetFactory } from '@asrjs/speech-recognition/presets/parakeet';

const runtime = createSpeechRuntime({
  backends: [createWasmBackend()],
  modelFamilies: [createNemoTdtModelFamily()],
  presets: [createParakeetPresetFactory()],
});

const model = await runtime.loadModel({
  preset: 'parakeet',
  modelId: 'parakeet-tdt-0.6b-v3',
});

const session = await model.createSession();

try {
  const result = await session.transcribe(PcmAudioBuffer.fromMono(pcm, 16000), {
    detail: 'words',
    responseFlavor: 'canonical',
  });

  console.log(result.text);
  console.log(result.words);
} finally {
  await session.dispose();
  await model.dispose();
  await runtime.dispose();
}
```

### Built-in convenience runtime

```ts
import { PcmAudioBuffer, getCanonicalTranscript } from '@asrjs/speech-recognition';
import { createBuiltInSpeechRuntime } from '@asrjs/speech-recognition/builtins';

const runtime = createBuiltInSpeechRuntime();
const model = await runtime.loadModel({
  preset: 'whisper',
  modelId: 'openai/whisper-base',
});
const session = await model.createSession();

try {
  const response = await session.transcribe(PcmAudioBuffer.fromMono(pcm, 16000), {
    detail: 'detailed',
    responseFlavor: 'canonical+native',
  });

  const canonical = getCanonicalTranscript(response);
  console.log(canonical.text);
} finally {
  await session.dispose();
  await model.dispose();
  await runtime.dispose();
}
```

## Model Families vs Presets

`@asrjs/speech-recognition` treats technical implementations and branded families as different concepts.

- model families
  - `nemo-tdt`
  - `lasr-ctc`
  - `whisper-seq2seq`
- presets
  - `parakeet`
  - `medasr`
  - `whisper`

Rules:

- model families own execution and decoding
- presets resolve into family load requests
- presets may inject config defaults, artifact hints, and branded classification metadata
- presets do not own backend logic or execution loops

When a model is loaded through a preset, `model.info.family` remains technical and `model.info.preset` carries the branded alias.

## Transcript Contracts

`@asrjs/speech-recognition` is built around a stable canonical transcript shape.

### Canonical output

`TranscriptResult` includes:

- `text`
- `warnings`
- `meta`
- optional `segments`
- optional `words`
- optional `tokens`

### Streaming output

`PartialTranscript` includes:

- `kind`
- `revision`
- `text`
- `committedText`
- `previewText`
- `warnings`
- `meta`

### Transcription progress

Session transcription options can include `onProgress(event)` for stage-aware
progress updates during inference.

`TranscriptionProgressEvent` includes:

- `stage`
- `progress`
- `elapsedMs`
- `remainingMs`
- optional `completedUnits` / `totalUnits`
- optional `metrics`

Current real progress coverage is strongest on the NeMo TDT execution path,
including preprocess, encode, decode, postprocess, and complete stages.

### Detailed metrics

Transcript results can now carry richer timing metadata in `meta.metrics`, including:

- phase timings: `preprocessMs`, `encodeMs`, `decodeMs`, `tokenizeMs`, `postprocessMs`
- end-to-end timings: `totalMs`, `wallMs`
- throughput: `rtf`, `rtfx`
- model/runtime details: `preprocessorBackend`, `encoderFrameCount`, `decodeIterations`, `emittedTokenCount`, `emittedWordCount`
- optional audio-prep details when available: `decodeAudioMs`, `downmixMs`, `resampleMs`, `audioPreparationMs`, `inputSampleRate`, `outputSampleRate`, `resampler`

Browser audio decoding helpers also expose a preparation profile:

```ts
import { decodeAudioSourceToMonoPcm } from '@asrjs/speech-recognition/browser';

const decoded = await decodeAudioSourceToMonoPcm(file);
console.log(decoded.metrics);
```

### Response flavors

- `'canonical'`
- `'canonical+native'`
- `'native'`

### Worker safety

Canonical transcript objects are intentionally structured-clone-safe POJOs. That means they can cross Web Worker boundaries safely.

Native outputs are not guaranteed worker-safe unless a family documents that explicitly.

## IO and Asset Loading

`@asrjs/speech-recognition` has a first-class IO layer under `@asrjs/speech-recognition/io`.

Core contracts:

- `AssetRequest`
- `ResolvedAssetHandle`
- `AssetProvider`
- `AssetCache`

`ResolvedAssetHandle` is stream-first:

- `openStream()` is the large-asset path
- `readBytes()` is convenience
- `readText()` and `readJson()` build on the same handle
- `getLocator('url' | 'path')` exists for runtimes like ORT that still need a concrete URL or path
- `dispose()` cleans up temporary locators and other owned resources

Built-in providers/caches include:

- browser fetch
- Hugging Face
- browser local file/directory handles
- IndexedDB cache
- Node filesystem

## Lifecycle

`dispose()` is the primary lifecycle contract.

- `SpeechSession.dispose()` is required and idempotent
- `SpeechModel.dispose()` is required and idempotent
- `SpeechRuntime.dispose()` is required and idempotent
- asset handles also own disposal

Ownership rules:

- models dispose sessions they created
- runtimes dispose models they loaded
- backend execution resources must be released through the same lifecycle chain

`Symbol.asyncDispose` may exist as optional sugar in implementations, but it is not the required or primary API.

## Current Family Coverage

Architecture-based implementation families currently present in the repo:

- `nemo-tdt`
- `lasr-ctc`
- `whisper-seq2seq`
- `firered-llm`

Built-in runtime registration currently includes:

- `nemo-tdt`
- `lasr-ctc`
- `whisper-seq2seq`

Current branded presets include:

- `parakeet`
- `medasr`
- `whisper`

## Extending @asrjs/speech-recognition

`@asrjs/speech-recognition` should not become a bottleneck for new model support.

Bring your own model family:

```ts
import { createSpeechRuntime, createWasmBackend } from '@asrjs/speech-recognition';
import { createMySpeechModelFamily } from './my-family';

const runtime = createSpeechRuntime({
  backends: [createWasmBackend()],
  modelFamilies: [createMySpeechModelFamily()],
});
```

Bring your own branded preset:

```ts
runtime.registerPreset(createMyPresetFactory());
```

## Design Rules

- runtime is orchestration, not model intelligence
- model families own execution and decoding
- presets stay thin and branded
- canonical transcript contracts are the stable app-facing boundary
- browser-only APIs stay off the root path
- root exports stay narrow
- shared model logic moves up only after real reuse is proven
