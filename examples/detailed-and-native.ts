import { createSpeechRuntime } from 'asr.js';
import { createWasmBackend } from 'asr.js';
import { createParakeetPresetFactory } from 'asr.js';

export async function detailedAndNativeExample() {
  const runtime = createSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createParakeetPresetFactory({ useManifestSource: true })]
  });

  const model = await runtime.loadModel({
    family: 'parakeet',
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
