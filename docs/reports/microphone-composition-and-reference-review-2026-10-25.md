# Existing audio services and microphone integration — 2026-10-25

Decode, mono conversion, resampling, long-audio windowing and transcript merge
were already library services. The previous audio-source change composed them
with model loading, cancellation and cleanup; it did not add those DSP services.

## Source audit

| Existing service | Current implementation |
| --- | --- |
| File/Blob/URL decode and channel averaging | `src/runtime/media.ts`: `decodeAudioSourceToMonoPcm`, `mixAudioBufferChannelsToMono` |
| Resampling | Web Audio target-rate decode; existing `resampleLinear` native-rate path; capture's worklet/fixed-chunk resampler |
| Long audio | Loaded handle in `src/runtime/load.ts` delegates canonical requests to `planWindowedTranscription` / `transcribeWithWindowing` |
| Window assembly | `src/pipeline/long-audio-windowing.ts`, model inference limits and merge policy; timestamps/words are reconciled there |
| Microphone/VAD/context | `src/runtime/browser-controller.ts`, `browser-realtime.ts`, `SpeechContextBuffer`, `RealtimeTranscriptionController` |
| Native streaming | Model-specific streaming capability; distinct from VAD context re-decoding |

The inspected local references were `N:/github/ysdede/parakeet.js`,
`N:/github/ysdede/transformers.js`, the TDT WebGPU and Parakeet reference demos,
and the sibling browser/streaming/Parakeet UI demos. Legacy apps that decode and
take channel 0 are not a replacement for the current multichannel downmix.
The TDT audio-preparation lab has selectable parity/benchmark resamplers; it
does not establish that common audio preparation is absent from this library.

Parakeet.js `src/long_audio.js` and the fork's
`packages/transformers/src/models/nemo_conformer_tdt/pipeline_nemo_conformer_tdt.js`
share the main 180-second threshold / 90-second window / 10-second overlap
lineage already reflected in the core TDT policy. We did not transplant another
windowing or merge engine.

Two reference refinements require separate model-owned work: Parakeet.js's
audible-tail re-scan and its `previousDecoderState` continuation between calls.
The current TDT executor exposes a state snapshot but does not accept that prior
state as a new-call input. Decoder-state continuation alone also does not prove
causal/native encoder streaming. No parity or regression claim is made for these
differences without a discriminating real-model fixture.

## Changes

- The existing browser microphone controller accepts a borrowed loaded `model`.
  It creates one transcription path, reuses existing resampling for differing
  model rates, and rejects simultaneous model/callback/native-transcriber paths.
- `createSpeechController()` adds `startMicrophone` / `stopMicrophone` through a
  lazily imported browser controller. It reuses its loaded model. Existing base
  controller contracts remain usable by custom consumers and framework bindings.
- The Realtime app's established capture-context text collector moved into
  `runtime/realtime.ts`; its app module re-exports the library implementation.
  Revisions replace the same context, stale revisions are ignored, and repeated
  words in distinct turns remain intact. This is separate from window merging.
- Stop now drains offline utterance inference, not just native streaming work.
  Failures reject after cleanup; arbitrary rejection values remain observable.
  Disposal cancels/drains work before the borrowed model can be released.
  Cleanup failure ordering remains compatible with the existing tests.
- React/Svelte/Vue examples have microphone controls and reuse the same loaded
  controller. The framework build uses ES module workers, as the main demos do.

Read capture text from `snapshot.microphone?.transcriptText`; canonical results
retain their own time/context semantics. Stop releases capture but retains the
model; controller disposal also releases the model. Capture errors remain in
`micError` and in rejected stop/cleanup promises.

## Verification

- Core build/typecheck: passed. Full tests: **2050 passed, 36 skipped**.
- Browser root static graph: **44 modules**; no budget relaxation.
- Framework typecheck and adapter/suite tests: passed, **10 tests**.
- Changed core lint: zero errors, one existing browser-controller length warning.
- Generated API documentation: zero errors, 42 existing warnings.
- Four main demos: 22 suite/config tests and 71 app tests passed; all four rebuilt.
- Three framework examples rebuilt; seven-page smoke and six backend request
  checks passed. Microphone controls exist and remain disabled before model load.
- Parent verified every installed SDK dist hash: 1711 matched; source manifest
  also matches. The Realtime app's original three capture-text tests pass through
  the shared library export.

Real Chrome used the existing local Parakeet TDT v3 FP16 encoder on WebGPU and
INT8 decoder on WASM. The 11-second JFK fixture matched its known text for file
input, controlled manual microphone input, TEN-VAD-only input and a subsequent
file on the same model. TEN-VAD was ready; its five context updates produced one
final transcript without repetition. Both capture tracks ended, graph requests
occurred once per graph, and disposal completed. Parent checks independently
verified the receipt and actual artifact hashes.

This acceptance is controlled Chromium audio, not a physical microphone test,
and is one model/backend/fixture. It is not all-model accuracy, numerical stage
parity, full-WebGPU operator coverage or a performance comparison.

See [evidence](evidence/microphone-composition-2026-10-25/README.md). Failed setup
receipts and the initial cleanup-order regression log are retained. The former
failed before model inference: Python MIME/transport and Vite dev public-module
handling were avoided by reusing the existing production suite server.

SDK archive: `asrjs-speech-recognition-0.2.0-microphone-context-20261025.tgz`.
SHA-256: `4520cd899fa8d4d1695f947be37f1cde6dfc9213728d938576bebdc0e6f920a8`.
