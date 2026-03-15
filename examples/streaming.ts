import { createSpeechRuntime } from '@asrjs/speech-recognition';
import { createWasmBackend } from '@asrjs/speech-recognition';
import { DefaultStreamingTranscriber } from '@asrjs/speech-recognition/inference';
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';
import { createParakeetPresetFactory } from '@asrjs/speech-recognition/presets/parakeet';

export async function streamingExample() {
  const runtime = createSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createNemoTdtModelFamily()],
    presets: [createParakeetPresetFactory({ useManifestSource: true })]
  });

  const model = await runtime.loadModel({
    preset: 'parakeet',
    modelId: 'parakeet-tdt-0.6b-v3'
  });
  const session = await model.createSession();
  const streaming = new DefaultStreamingTranscriber(session, {
    detail: 'words',
    overlapMs: 500,
    maxWindowMs: 5000
  });

  const partial = await streaming.pushAudio(new Float32Array(8000));
  const final = await streaming.finalize();

  return { partial, final };
}
