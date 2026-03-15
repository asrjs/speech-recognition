import { createWasmBackend } from '@asrjs/speech-recognition';
import { createSpeechRuntime } from '@asrjs/speech-recognition';
import { createLasrCtcModelFamily } from '@asrjs/speech-recognition/models/lasr-ctc';
import { createMedAsrPresetFactory } from '@asrjs/speech-recognition/presets/medasr';

export async function transcribeMedAsrExample() {
  const runtime = createSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createLasrCtcModelFamily()],
    presets: [createMedAsrPresetFactory()],
  });

  const model = await runtime.loadModel({
    preset: 'medasr',
    modelId: 'google/medasr',
  });
  const session = await model.createSession();

  return session.transcribe(new Float32Array(16000), {
    detail: 'words',
    responseFlavor: 'canonical+native',
  });
}
