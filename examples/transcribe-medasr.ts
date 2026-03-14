import { createWasmBackend } from 'asr.js';
import { createSpeechRuntime } from 'asr.js';
import { createMedAsrPresetFactory } from 'asr.js';

export async function transcribeMedAsrExample() {
  const runtime = createSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createMedAsrPresetFactory()]
  });

  const model = await runtime.loadModel({
    family: 'medasr',
    modelId: 'google/medasr'
  });
  const session = await model.createSession();

  return session.transcribe(new Float32Array(16000), {
    detail: 'words',
    responseFlavor: 'canonical+native'
  });
}
