# Framework bindings

The core library stays framework-neutral. Thin companion packages connect its
controller to React, Svelte and Vue:

| Framework | Package                            | Binding                                    |
| --------- | ---------------------------------- | ------------------------------------------ |
| React     | `@asrjs/speech-recognition-react`  | `useSpeechRecognition(controller)`         |
| Svelte    | `@asrjs/speech-recognition-svelte` | `createSpeechRecognitionStore(controller)` |
| Vue       | `@asrjs/speech-recognition-vue`    | `useSpeechRecognition(controller)`         |

The local companion workspace is `../frameworks` beside this repository. Its
README contains build commands and an executable example for each framework.
The packages are development builds, not yet published releases. They require
this checkout's `/controller` export.

## Shared controller

```ts
import { createSpeechController } from '@asrjs/speech-recognition/controller';

const controller = createSpeechController();
const unsubscribe = controller.subscribe(() => {
  console.log(controller.getSnapshot());
});

// Load using the same options accepted by loadSpeechModel().
// await controller.load(loadOptions);

unsubscribe();
await controller.dispose();
```

`load`, `transcribeSource`, `transcribe`, `startStreaming`, `pushAudio` and `finalizeStreaming`
perform the work. `getSnapshot` and `subscribe` expose immutable state to an
application. Model-specific decoding stays in its model family. No framework
runtime is imported by the core controller.

For browser files, call `controller.transcribeSource(file)` after loading the
model. This also accepts a `Blob` or audio URL. The core resolves the model's
sample rate and owns decoding, mono conversion and cancellation. Decoding is part
of the active operation, so disposal waits for it before releasing the model.
The file capability extends the original PCM controller contract as
`AudioSourceSpeechController`. Existing custom controllers remain valid for bindings;
the bindings preserve the concrete controller type supplied to them.

The same factory returns a `MicrophoneSpeechController`, adding
`startMicrophone(options)` and `stopMicrophone()`. It composes the existing
browser capture/VAD controller lazily and reuses the loaded model. Read
`snapshot.microphone?.transcriptText` for the accumulated capture text and
`snapshot.microphone?.micError` for capture/inference errors. Stop awaits pending
utterances and releases capture resources; the model remains available for
another recording or file. Controller disposal also releases its model.

Model loading, artifact selection, caching and fallback belong to the core
library. Examples pass model choices through `buildSpeechModelLoadOptions`;
bindings only subscribe to the controller. `RuntimeProgressEvent.warning`
reports recoverable loading decisions, including the reason for a precision
fallback before another artifact is requested. The controller retains these
in `loadWarnings` after download progress and readiness updates, and clears
them when a new load begins. A warning describes an attempted fallback;
`status` still determines whether loading succeeded.

Each binding borrows the controller. Component teardown removes its listener;
the application that created the controller disposes it. This permits several
components to read one model without loading it repeatedly or disposing it
when just one view closes. For SSR, create state per request and start browser
inference from client actions.

A model's streaming capability determines whether native streaming is available.
Microphone capture and VAD remain separate browser primitives; see
[Streaming and deployment](INTEGRATION_GUIDE.md). Node/Bun integrations use the
same headless core and do not need framework packages.
