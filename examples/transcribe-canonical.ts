import { createSpeechRuntime } from '@asrjs/speech-recognition';
import { createWasmBackend } from '@asrjs/speech-recognition';
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';

export async function transcribeCanonicalExample() {
  const runtime = createSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createNemoTdtModelFamily()]
  });

  const model = await runtime.loadModel({
    family: 'nemo-tdt',
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
