import { createWasmBackend } from '@asrjs/speech-recognition';
import { createSpeechRuntime } from '@asrjs/speech-recognition';
import { createHfCtcModelFamily } from '@asrjs/speech-recognition/models/hf-ctc-common';
import { createMedAsrPresetFactory } from '@asrjs/speech-recognition/presets/medasr';

export async function transcribeMedAsrExample() {
  const runtime = createSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createHfCtcModelFamily()],
    presets: [createMedAsrPresetFactory()]
  });

  const model = await runtime.loadModel({
    preset: 'medasr',
    modelId: 'google/medasr'
  });
  const session = await model.createSession();

  return session.transcribe(new Float32Array(16000), {
    detail: 'words',
    responseFlavor: 'canonical+native'
  });
}
