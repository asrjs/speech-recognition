import { createSpeechRuntime } from '@asrjs/speech-recognition';
import { createWasmBackend } from '@asrjs/speech-recognition';
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';
import { createParakeetPresetFactory } from '@asrjs/speech-recognition/presets/parakeet';

export async function detailedAndNativeExample() {
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

  return session.transcribe(new Float32Array(32000), {
    detail: 'detailed',
    responseFlavor: 'canonical+native',
    returnFrameIndices: true,
    returnLogProbs: true,
    returnTdtSteps: true
  });
}
