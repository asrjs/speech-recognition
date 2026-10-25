# Streaming and deployment

Use the [README](../README.md) for the first transcription and the [API guide](PUBLIC_API_GUIDE.md) for loading, transcript formats and lifecycle details.

## Choose a streaming contract

| Path                                | Input and state                                                                        | Application contract                                                         |
| ----------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Native streaming                    | Ordered chunks; model-owned encoder/decoder state where supported                      | Check `loaded.supportsStreaming`; create one transcriber for the stream      |
| Windowed offline inference          | Bounded overlapping audio windows decoded by an offline session                        | Use explicit window/overlap policy; do not label it native streaming         |
| VAD-driven microphone transcription | Captured PCM, speech probabilities, utterance boundaries and optional acoustic context | Keep capture and inference lifecycles separate; revise the active hypothesis |

VAD answers when speech occurs. It does not make an offline model stateful. Parakeet/Whisper-style offline inference can participate in a realtime application through shared windowing and endpointing; models with native streaming keep their own state contracts.

### Native streaming lifecycle

After loading a compatible model, check `loaded.supportsStreaming`, then call `loaded.createStreamingTranscriber()`. Await each `pushAudio(input)` in order. `flush()` requests a current result; `finalize()` ends the stream. Use `reset()` to begin another stream on reusable resources. Dispose the loaded handle when finished; it tracks transcribers it creates. Do not interleave unrelated audio streams on one transcriber.

Partial results are hypotheses with revisions. Render the current hypothesis rather than blindly concatenating every partial. Finalization, timestamps and canonical output have defined contracts in the [API guide](PUBLIC_API_GUIDE.md#streaming-output).

The [native streaming recipe](../examples/streaming-native.ts) takes the selected model's load options, an ordered async iterable of PCM chunks, their sample rate and a revision callback. It checks support before creating a transcriber and releases its loaded handle on completion or failure. It does not resample input. Cancellation is checked between awaited operations; streaming calls do not currently accept a per-push abort signal, and the audio producer must handle interruption of a pending read itself.

For an explicit offline-window adapter, see [examples/streaming.ts](../examples/streaming.ts); this example intentionally does not claim native model streaming.

## Browser capture and VAD

The `/browser` entry exposes browser capture and realtime helpers; `/realtime` exposes orchestration. Keep browser-only dependencies out of server imports. The shared capture layer owns PCM preparation, resampling and buffering. TEN-VAD and FireRed VAD provide neural speech detection; rough-gate and neural modes are separate choices.

The Realtime companion app combines these primitives with waveform, VAD probability and event timelines. It is the maintained integration surface, not a separate inference engine. Framework UI state belongs in the app; reusable capture and inference behavior belongs in the library.

The shared application controller can reuse its loaded model for microphone
transcription:

```ts
import { createSpeechController } from '@asrjs/speech-recognition/controller';

const speech = createSpeechController();
await speech.load({ modelId: 'parakeet-tdt-0.6b-v2', backend: 'webgpu-hybrid' });
speech.subscribe(() => console.log(speech.getSnapshot().microphone?.transcriptText));
// Start from a user gesture.
await speech.startMicrophone({
  config: { gateMode: 'vad-only', vadBackend: 'ten-vad', tenVadEnabled: true },
});
// Later:
await speech.stopMicrophone();
await speech.dispose();
```

This reuses the existing capture, VAD, acoustic-context and realtime transcript
controllers. `stopMicrophone()` flushes and waits for queued transcription,
preserves the last capture text, releases capture resources, and keeps the model
loaded. `dispose()` also releases the model. A file operation, native stream or
model change cannot run concurrently with the active microphone session.

If an application already owns a `LoadedSpeechModel`, pass it as `model` to
`createBrowserRealtimeMicrophoneController`. The [microphone recipe](../examples/browser-microphone.ts)
shows this borrowed-model path. `onTranscript` receives a canonical realtime
update with capture-relative timestamps and context id/revision;
`getState().transcriptText` combines revisions without appending them twice.
Call `stop({ flush: true })` to finish capture and await queued inference.
Call microphone disposal before disposing the borrowed model. Failures are
reported in `micError` and rejected stop/cleanup promises; keep handling both.

The opt-in `transcriptionContext` browser-controller option retains actual captured audio across nearby VAD utterances. Its browser defaults are 600 ms preroll, a 30-second context bound and a 3-second gap limit. Successive snapshots share a context id and increasing revision. Consumers replace the corresponding transcript rather than append repeated context. A successful flushing stop can emit one final captured-tail revision. This is acoustic context re-decoding, not persistent native decoder-cache support for every model.

Forward cancellation signals to inference and stop capture before disposing resources. Resetting a controller must not allow stale work to overwrite a new recording. The companion applications demonstrate ownership and cleanup around model loading, microphone permission and asynchronous transcription.

## Offline deployment

Keep two meanings explicit: processing a complete audio file, and running without network access. For network-independent operation:

1. Supply all model graphs, external-data sidecars, tokenizer/config files and any VAD model assets locally or through an application-owned cache.
2. Serve the matching ONNX Runtime Web engine modules and WASM files locally. Browser local-model helpers accept `wasmPaths` separately from the model directory; use a URL prefix ending in `/`.
3. Choose a consistent engine-asset location before initializing sessions: ORT Web configuration is module-wide. Omitted paths may retain CDN defaults.
4. Configure the deployment's worker URLs and cross-origin isolation for paths requiring shared memory. Model weights alone do not establish offline readiness.
5. Verify a cold deployment with network access disabled, including loading, capture and inference. A warm browser cache can conceal missing assets.

The demo suite copies engine assets during its build. Consumer applications must configure their own deployment; the library does not implicitly package every model or engine asset.

### Load a local model in the browser

Supply files from a directory input (including graph sidecars, configuration and tokenizer files) with paths preserved. The same loader selects the model-family adapter; applications do not need a switch for each model:

```ts
import {
  createSpeechModelLocalEntries,
  loadSpeechModelFromLocalEntries,
} from '@asrjs/speech-recognition/browser';

export async function loadLocalModel(files: FileList, modelId: string) {
  return loadSpeechModelFromLocalEntries({
    modelId,
    entries: createSpeechModelLocalEntries(Array.from(files)),
    backend: 'wasm',
    wasmPaths: '/ort/',
  });
}
```

Set a directory input's `webkitdirectory` attribute when supported; normal multi-file selection may lose the relative paths required by a model layout. Choose a descriptor whose `loading.supportsLocalSource` is true and supply its expected exported layout. The WASM example still requires that model's WASM-compatible graphs. Serve `/ort/` from the application with engine files matching its installed ONNX Runtime Web version. The returned handle has the same transcription and disposal contract as `loadSpeechModel()`; dispose it when finished. Local inspection can describe missing artifacts but does not certify the graphs' numerical correctness.

## Node.js and Bun

The product goal includes local JavaScript runtimes alongside browsers. The current [native NeMo TDT recipe](../examples/nemo-tdt-native-local.mjs) explicitly composes the native backend, family and filesystem provider. Native ORT is an optional dependency and this path remains experimental. The recipe targets NeMo TDT; running it with Bun is not evidence of equivalent coverage for every runtime/model pair. Use [native setup instructions](PUBLIC_API_GUIDE.md#local-native-nodebun-experimental); do not infer all-family native or server WebGPU support from browser compatibility.

## Extending the product

Keep application code model-agnostic through descriptors, capability checks and load-option builders. Graph shapes, caches, decoding and model-specific audio requirements stay within model families. Presets select assets and configurations. A new model should reuse the public lifecycle and transcript contract, rather than require new loading branches in all four demos.

Established optimized Parakeet and Whisper paths are regression anchors. Custom ONNX exports are part of the engineering toolkit, not an exceptional fallback. See the [Whisper workflow](whisper-export-workflow.md) and [tool map](../tools/README.md) before creating another exporter.
