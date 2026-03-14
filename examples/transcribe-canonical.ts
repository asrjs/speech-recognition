import { createSpeechRuntime } from 'asr.js';
import { createWasmBackend } from 'asr.js';
import { createNemoTdtModelFamily } from 'asr.js';

export async function transcribeCanonicalExample() {
  const runtime = createSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createNemoTdtModelFamily()]
  });

  const model = await runtime.loadModel({
    modelId: 'nemo-fastconformer-tdt-scaffold',
    classification: {
      ecosystem: 'nemo',
      encoder: 'fastconformer',
      decoder: 'tdt',
      task: 'asr'
    }
  });
  const session = await model.createSession();

  return session.transcribe(new Float32Array(16000), {
    detail: 'segments',
    responseFlavor: 'canonical'
  });
}
