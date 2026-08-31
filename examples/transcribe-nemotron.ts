import { createSpeechRuntime, createWasmBackend } from '@asrjs/speech-recognition';
import { createNemotronRnntModelFamily } from '@asrjs/speech-recognition/models/nemotron-rnnt';
import { createNemotronPresetFactory } from '@asrjs/speech-recognition/presets/nemotron';

export async function transcribeNemotronExample() {
  // Explicitly register the nemotron family and preset (the built-in
  // runtime path is reserved for the public catalog; see README's
  // Few-line quick start for the high-level helper that lists the
  // model). To pull weights from the upstream INT4 repo, set the
  // `useManifestSource` option on the preset factory.
  const runtime = createSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createNemotronRnntModelFamily()],
    presets: [createNemotronPresetFactory({ useManifestSource: true })],
  });

  const model = await runtime.loadModel({
    preset: 'nemotron',
    modelId: 'nemotron-3.5-asr-streaming-0.6b',
  });
  const session = await model.createSession();

  return session.transcribe(new Float32Array(16000), {
    detail: 'words',
    responseFlavor: 'canonical+native',
  });
}
