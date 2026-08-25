import { loadSpeechModel } from '../../../src/runtime/load.ts';
import { decodeAudioSourceToMonoPcm } from '../../../src/runtime/media.ts';

const status = document.querySelector('#status');
const result = document.querySelector('#result');
const MODEL_ROOT = '/@fs/N:/models/onnx/nemo/canary-180m-flash-smoke';
const SAMPLE_AUDIO_URL = '/@fs/N:/github/asrjs/speech-recognition/tools/data/fixtures/audio/jfk-short.wav';
const expected =
  'And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.';

function report(value) {
  result.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

async function run() {
  if (!navigator.gpu) throw new Error('WebGPU is unavailable.');
  status.textContent = 'Loading Canary FP16 graphs…';
  const model = await loadSpeechModel({
    family: 'nemo-aed',
    modelId: 'nvidia/canary-180m-flash',
    backend: 'webgpu',
    options: {
      source: {
        kind: 'direct',
        preprocessorBackend: 'js',
        encoderBackend: 'webgpu',
        decoderBackend: 'webgpu',
        artifacts: {
          encoderUrl: `${MODEL_ROOT}/encoder-model.fp16.onnx`,
          decoderUrl: `${MODEL_ROOT}/decoder-model.fp16.onnx`,
          tokenizerUrl: `${MODEL_ROOT}/tokenizer.json`,
          configUrl: `${MODEL_ROOT}/config.json`,
        },
      },
    },
  });

  try {
    status.textContent = 'Transcribing JFK sample…';
    const audio = await decodeAudioSourceToMonoPcm(SAMPLE_AUDIO_URL, { targetSampleRate: 16000 });
    const response = await model.transcribeMonoPcm(audio.pcm, audio.sampleRate, {
      sourceLanguage: 'en',
      targetLanguage: 'en',
      maxNewTokens: 128,
      responseFlavor: 'canonical+native',
    });
    const transcript = response.canonical?.text || response.native?.utteranceText || '';
    report({
      ok: transcript === expected,
      transcript,
      expected,
      metrics: response.canonical?.meta?.metrics,
      warnings: response.canonical?.warnings,
      native: response.native,
      crossOriginIsolated: globalThis.crossOriginIsolated,
    });
    status.textContent = transcript === expected ? 'PASS' : 'TEXT MISMATCH';
  } finally {
    await model.dispose();
  }
}

run().catch((error) => {
  status.textContent = 'FAIL';
  report({ error: error instanceof Error ? error.stack || error.message : String(error) });
});
