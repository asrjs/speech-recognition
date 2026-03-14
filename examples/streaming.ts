import { createSpeechRuntime } from 'asr.js';
import { createWasmBackend } from 'asr.js';
import { createParakeetPresetFactory } from 'asr.js';
import { DefaultStreamingTranscriber } from 'asr.js';

export async function streamingExample() {
  const runtime = createSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createParakeetPresetFactory({ useManifestSource: true })]
  });

  const model = await runtime.loadModel({
    family: 'parakeet',
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
