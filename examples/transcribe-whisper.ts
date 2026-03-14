import { createWasmBackend } from 'asr.js';
import { createSpeechRuntime } from 'asr.js';
import { createWhisperPresetFactory } from 'asr.js';

export async function transcribeWhisperExample() {
  const runtime = createSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createWhisperPresetFactory()]
  });

  const model = await runtime.loadModel({
    family: 'whisper',
    modelId: 'openai/whisper-base'
  });
  const session = await model.createSession();

  return session.transcribe(new Float32Array(16000), {
    detail: 'detailed',
    task: 'transcribe',
    responseFlavor: 'canonical+native'
  });
}
