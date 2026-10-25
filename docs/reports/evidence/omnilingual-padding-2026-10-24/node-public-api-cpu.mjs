import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { loadSpeechModel, PcmAudioBuffer } from '../../dist/index.js';
import { createBuiltInSpeechRuntime } from '../../dist/builtins.js';
import { decodeWavToMonoPcm } from '../../dist/audio/wav.js';
import { normalizeWav2Vec2Waveform } from '../../dist/models/wav2vec2/executor.js';

const root = process.cwd();
const wavPath = process.argv[2] ?? `${root}/tests/fixtures/jfk2.en.wav`;
const modelPath = `${root}/models/onnx/omnilingual-ctc-300m/model.int8.onnx`;
const tokensPath = `${root}/models/onnx/omnilingual-ctc-300m/tokens.txt`;
const wavBytes = readFileSync(wavPath);
const decoded = decodeWavToMonoPcm(wavBytes);
const normalized = normalizeWav2Vec2Waveform(decoded.pcm);
const tokensText = readFileSync(tokensPath, 'utf8');
const runtime = createBuiltInSpeechRuntime();
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const stats = (values) => {
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    sum += value;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  const mean = sum / values.length;
  let squares = 0;
  for (const value of values) squares += (value - mean) ** 2;
  return { mean, std: Math.sqrt(squares / values.length), min, max };
};
const model = await loadSpeechModel({
  modelId: 'OpenVoiceOS/omnilingual-asr-ctc-300m-onnx',
  family: 'omnilingual-ctc',
  runtime,
  backend: 'wasm',
  options: {
    modelUrl: modelPath,
    tokensUrl: `data:text/plain,${encodeURIComponent(tokensText)}`,
    sampleRate: 16000,
    backendId: 'wasm',
  },
});
let result;
try {
  const response = await model.transcribe(PcmAudioBuffer.fromMono(decoded.pcm, decoded.sampleRate), {
    responseFlavor: 'canonical+native',
    detail: 'detailed',
  });
  result = {
    status: 'PUBLIC_LIBRARY_CPU_TRACE_NOT_ACCEPTANCE',
    backend: model.model?.info?.backendId ?? 'native-cpu',
    input: { path: wavPath, wavSha256: sha256(wavBytes), sampleRate: decoded.sampleRate, channels: decoded.numberOfChannels, samples: decoded.pcm.length },
    pcm: { sha256Float32le: sha256(Buffer.from(decoded.pcm.buffer, decoded.pcm.byteOffset, decoded.pcm.byteLength)), ...stats(decoded.pcm) },
    normalizedPcm: { sha256Float32le: sha256(Buffer.from(normalized.buffer, normalized.byteOffset, normalized.byteLength)), ...stats(normalized) },
    artifacts: { modelSha256: sha256(readFileSync(modelPath)), tokensSha256: sha256(readFileSync(tokensPath)) },
    transcript: response.canonical.text,
    nativeText: response.native.utteranceText,
    collapsedIds: response.native.ctc?.collapsedIds ?? [],
  };
} finally {
  await model.dispose();
}
writeFileSync(`${root}/tmp/omnilingual-diagnosis/node-public-api-cpu-${wavPath.split(/[\\/]/).pop().replace(/\.[^.]+$/, '')}.json`, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify(result, null, 2));
