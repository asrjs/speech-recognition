# @asrjs/speech-recognition

A TypeScript library for running speech recognition locally: transcribe audio files, build microphone applications, and consume streaming transcripts through one speech-focused API. WebGPU and WASM are the browser inference backends; Node.js and Bun are also product targets, with explicit environment-specific integration.

The package owns model loading, audio preparation, inference, canonical transcripts, and resource lifecycle. Browser capture, TEN-VAD/FireRed VAD, waveform monitoring, and realtime orchestration are reusable library components. Applications choose their UI and framework.

Audio decoding, mono downmix, resampling and model-aware long-audio
windowing/merge are shared implementations. File and microphone convenience
APIs compose those implementations with model ownership and cancellation.

## Install and transcribe

```sh
npm install @asrjs/speech-recognition
```

In a browser, pass an audio `File`, `Blob` or URL. The library loads the model,
decodes and resamples the audio, transcribes it and releases the model resources.
This example uses Parakeet TDT with a WebGPU encoder and WASM decoder/joint:

```ts
import { transcribeSpeechFromAudioSource } from '@asrjs/speech-recognition';

export function transcribe(file: File) {
  return transcribeSpeechFromAudioSource(file, {
    modelId: 'parakeet-tdt-0.6b-v2',
    backend: 'webgpu-hybrid',
  });
}
```

For repeated files, use `createSpeechController()` and call
`controller.transcribeSource(file)` after loading once. For decoded PCM in a
browser or Node.js/Bun, use `transcribeSpeechFromMonoPcm(pcm, sampleRate, options)`.
Model assets are separate from the npm package. Choose an available model/backend
combination with `listSpeechModels()` and `buildSpeechModelLoadOptions()`; WASM
availability is model-dependent. Browser file decoding uses Web Audio by default;
importing the root package does not initialize browser APIs.

The built-in loading recipe resolves the preset's configured model source and may fetch weights on first use. For application-supplied files, use the [local-model loading recipe](https://github.com/asrjs/speech-recognition/blob/main/docs/INTEGRATION_GUIDE.md#load-a-local-model-in-the-browser). A catalog entry describes capabilities; compatible model artifacts must still be available.

See [browser-file-transcription.ts](https://github.com/asrjs/speech-recognition/blob/main/examples/browser-file-transcription.ts)
for the file API. [examples/README.md](https://github.com/asrjs/speech-recognition/blob/main/examples/README.md)
maps file, microphone, native streaming and Node.js integrations.

## Choose your integration

| You want to                                              | Start here                                                                                                                                                                                                                                                                                         |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Transcribe files or PCM, inspect timestamps and progress | [Public API guide](https://github.com/asrjs/speech-recognition/blob/main/docs/PUBLIC_API_GUIDE.md#high-level-model-loading)                                                                                                                                                                        |
| Add microphone capture, VAD and live transcripts         | [Streaming and deployment guide](https://github.com/asrjs/speech-recognition/blob/main/docs/INTEGRATION_GUIDE.md)                                                                                                                                                                                  |
| Run local Node.js/Bun inference                          | [Native integration](https://github.com/asrjs/speech-recognition/blob/main/docs/PUBLIC_API_GUIDE.md#local-native-nodebun-experimental) and [executable example](https://github.com/asrjs/speech-recognition/blob/main/examples/nemo-tdt-native-local.mjs)                                          |
| Use cached/local weights without network access          | [Offline deployment](https://github.com/asrjs/speech-recognition/blob/main/docs/INTEGRATION_GUIDE.md#offline-deployment)                                                                                                                                                                           |
| Explore models, waveform/VAD signals and performance     | [Companion demos](https://github.com/asrjs/demos)                                                                                                                                                                                                                                                  |
| Add a model or optimize its ONNX graphs                  | [Architecture](https://github.com/asrjs/speech-recognition/blob/main/docs/architecture.md), [model tools](https://github.com/asrjs/speech-recognition/blob/main/tools/README.md), [Whisper export workflow](https://github.com/asrjs/speech-recognition/blob/main/docs/whisper-export-workflow.md) |

## React, Svelte and Vue

[Framework bindings](https://github.com/asrjs/speech-recognition/blob/main/docs/FRAMEWORKS.md) connect one controller to React hooks, Svelte stores and Vue composables. The core owns loading, progress, transcription and streaming state. The companion packages and examples are currently local development builds.

## Offline and streaming

Model files use the library's [browser cache](https://github.com/asrjs/speech-recognition/blob/main/docs/BROWSER_CACHE.md). Applications on the same origin can reuse stored artifacts across framework and SDK instances. Different sites or ports have separate browser storage.

Offline transcription processes a complete recording. Offline deployment means all required model and engine assets are available locally; the two are independent.

Streaming has three distinct forms: native model streaming, bounded windowed inference over an offline model, and VAD-driven microphone transcription. The library exposes capability checks and shared capture/orchestration so applications do not have to implement model-specific branches. [The integration guide](https://github.com/asrjs/speech-recognition/blob/main/docs/INTEGRATION_GUIDE.md) explains state, context and cleanup for each form.

Browser WebGPU/WASM and native Node/Bun are distinct execution environments. Native CPU integration is currently explicit and experimental; it is not a claim that every family supports every environment. A model appearing in the catalog is discovery metadata, not a universal compatibility guarantee.

## Established foundations

Parakeet TDT optimization began in `parakeet.js`, continued through the Transformers.js fork, and informed this library's model-family implementations and hybrid execution. Whisper has its own optimized ONNX export workflow, including the four-graph path. These are foundations to preserve and build upon as new model families arrive.

TEN-VAD and FireRed VAD complement ASR with speech detection and endpointing. Waveform, probability and event timelines make capture behavior inspectable. The companion suite uses these shared components in four applications:

- **Showcase:** try transcription and inspect results.
- **Playground:** explore loading, options, output and streaming capabilities.
- **Realtime:** microphone, waveform/VAD diagnostics and live transcripts.
- **Benchmark:** compare measured model/backend workloads.

Historical ports, measured results and current acceptance are recorded separately. See the [documentation index](https://github.com/asrjs/speech-recognition/blob/main/docs/README.md) for product design and dated engineering evidence.

## Project structure

One ESM-first, framework-neutral package, with narrow public subpaths:

| Location                                | Responsibility                                          |
| --------------------------------------- | ------------------------------------------------------- |
| `src/types`                             | Audio, transcript, model and streaming contracts        |
| `src/runtime`                           | Loading, sessions, lifecycle, capture and orchestration |
| `src/audio`, `src/io`, `src/tokenizers` | Audio preparation, assets/cache and text decoding       |
| `src/inference`                         | Shared inference descriptors and primitives             |
| `src/models`                            | Family-specific graph execution and decoding            |
| `src/presets`                           | Branded configurations and asset locators               |
| `examples`                              | Integration recipes using public APIs                   |
| `tools`                                 | Export, profiling, parity and model-port tooling        |
| `docs`                                  | Product/API guides, architecture and dated reports      |

Model-specific logic stays in families; reusable speech behavior stays in the library; application state and framework components stay in the companion apps. [Architecture and dependency rules](https://github.com/asrjs/speech-recognition/blob/main/docs/architecture.md).

## Development

```sh
npm run typecheck
npm run lint
npm test
npm run build
npm run docs:build
```

`docs:build` generates API documentation from JSDoc into `.typedoc-site/`. Start with the [project charter](https://github.com/asrjs/speech-recognition/blob/main/docs/PROJECT_CHARTER.md) for product decisions and the [contributor entrypoint](https://github.com/asrjs/speech-recognition/blob/main/docs/AGENT_ENTRYPOINT.md) for scoped engineering work.

Runnable examples: [React](https://github.com/asrjs/demos/tree/framework-examples/examples/react), [Svelte](https://github.com/asrjs/demos/tree/framework-examples/examples/svelte), [Vue](https://github.com/asrjs/demos/tree/framework-examples/examples/vue). The [framework workspace](https://github.com/asrjs/demos/tree/framework-examples) includes the thin adapters, build commands and local setup.
