import { createWasmBackend } from '@asrjs/speech-recognition';
import { createSpeechRuntime } from '@asrjs/speech-recognition';
import { createWhisperSeq2SeqModelFamily } from '@asrjs/speech-recognition/models/whisper-seq2seq';
import { createWhisperPresetFactory } from '@asrjs/speech-recognition/presets/whisper';

export async function transcribeWhisperExample() {
  const runtime = createSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createWhisperSeq2SeqModelFamily()],
    presets: [createWhisperPresetFactory()]
  });

  const model = await runtime.loadModel({
    preset: 'whisper',
    modelId: 'openai/whisper-base'
  });
  const session = await model.createSession();

  return session.transcribe(new Float32Array(16000), {
    detail: 'detailed',
    task: 'transcribe',
    responseFlavor: 'canonical+native'
  });
}
