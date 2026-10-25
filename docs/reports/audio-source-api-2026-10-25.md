# Audio source integration — 2026-10-25

Applications can pass browser audio files to the library without choosing a
sample rate or managing a separate decode-to-PCM step.

This was an API composition change. Decode, mono downmix, resampling and the
model-aware long-audio window/merge implementation already existed. The new
entry points reuse `runtime/media.ts` and the loaded handle's established
transcription path; they did not introduce new DSP or a replacement decoder.

## Public workflow

```ts
import { transcribeSpeechFromAudioSource } from '@asrjs/speech-recognition';

const result = await transcribeSpeechFromAudioSource(file, {
  modelId: 'parakeet-tdt-0.6b-v2',
  backend: 'webgpu-hybrid',
});
```

The one-shot API owns model loading, decode/downmix/resampling to the loaded
model's sample rate, transcription and awaited disposal. Existing PCM and
one-shot contracts remain available. The existing one-shot API and new helper
share cleanup handling, including simultaneous operation and cleanup failures.

For repeated files, load the controller once and call
`controller.transcribeSource(file)` for each file. Decode is part of its active
operation: overlapping loads/transcriptions are rejected; disposal cancels and
waits before releasing the model. Cancellation is checked before preparation
and inference. The original `SpeechController` interface remains valid for
custom consumers; the factory returns its extended audio-source interface.

React, Svelte and Vue file examples use that method. Their bindings retain the
concrete controller type, including custom capabilities, without framework
logic entering the core. The React chunk-streaming example intentionally keeps
its separate PCM recipe; this change does not add file-to-stream orchestration.

Browser decoding is lazy. The root package does not initialize Web Audio or
import the media module until needed. PCM remains the portable Node.js/Bun
integration path; arbitrary Node file decoding is not claimed. Low-level media
cancellation uses `AssetLoadAbortedError` with a stage identifier. This change
does not alter model precision, graph execution, cache identity or streaming
decoding policy.

## Verification

- Core build, source/example typecheck: passed.
- Full regression: 2031 passed, 36 skipped; 238 files passed, 7 skipped.
- Focused source/controller/media/root compatibility tests: 37 passed.
- Root static import graph: 44; requested limit stays below 45.
- Framework typecheck, including compile-time capability assertions: passed.
- Framework adapter/suite tests: 10 passed; three binding packages built.
- API documentation build: zero errors, 42 pre-existing warnings.
- Changed core files lint: zero errors, one existing load-file length warning.
- Four-demo suite/config tests: 22 passed; application tests: 71 passed.
- All four demos and three framework examples rebuilt with the updated SDK.
- All seven pages served HTTP 200 and rendered their main landmark in Chrome.
  Six isolated framework/backend request checks still selected FP16 encoder
  for WebGPU hybrid and INT8 encoder for WASM; model downloads were aborted.
- Parent verification: all 1711 installed SDK dist hashes match the archive;
  source hash manifest also matches the current source files.

Real Chrome decoded File and URL WAV inputs and resampled 4000 samples at
16 kHz to 2000 samples at the model's 8 kHz rate. Reusing a controller for two
files did not reload its model. Invalid audio was rejected. Across the fixture,
all four acquired model handles and all five AudioContexts were closed.
Model loading/inference was deliberately mocked at its module boundary: this
proves browser audio preparation and lifecycle, not model accuracy or performance.

The initial full run failed because the frozen root export test did not include
the new public function. The historical export fixture was preserved, and the
test now expects precisely that baseline plus the named new export. Missing
historical names and unexpected additions still fail. Both initial and final
test logs are retained in [evidence](evidence/audio-source-2026-10-25/README.md).

SDK archive: `asrjs-speech-recognition-0.2.0-audio-source-20261025.tgz`.
SHA-256: `f02ac32a521d934e44b6acc66682c29d84d3bb4665a9ee3c83bcdfde04d5112f`.
The archive has an adjacent source/dist hash manifest in `demos/vendor`.

This checkpoint does not certify every catalog model, real-weight browser
inference, microphones, VAD, native streaming or all Node.js/Bun environments.
Those remain distinct acceptance gates in the product tasks and integration
guide. The next consumer-workflow gap is bringing microphone/VAD setup and
cleanup to the same simplicity using the existing core controllers.
