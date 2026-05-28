// @ts-nocheck
import { WhisperOnnxExecutor } from '../src/models/whisper-seq2seq/executor.js';

const config = {
  melBins: 80,
  maxSourcePositions: 3000,
  maxTargetPositions: 448,
  languages: ['tr', 'en'],
  processorArchitecture: 'whisper-mel',
  encoderArchitecture: 'whisper-transformer',
  decoderArchitecture: 'transformer-decoder',
  tokenizer: { kind: 'tiktoken', vocabSize: 51865 },
  windowing: { kind: 'disabled' },
};

const source = {
  kind: 'direct',
  artifacts: {
    encoderUrl: '/tmp/whisper-tiny-onnx/encoder_model.onnx',
    decoderUrl: '/tmp/whisper-tiny-onnx/decoder_model_merged.onnx',
    tokenizerUrl: 'file:///tmp/whisper-tiny-onnx/tokenizer.json',
  },
};

const executor = new WhisperOnnxExecutor(
  'whisper-tiny',
  { family: 'whisper-seq2seq', task: 'transcribe', samplingRate: 16000 },
  config,
  'wasm',
  { source }
);

// 1 second of 440 Hz sine at 16 kHz
const sampleRate = 16000;
const samples = new Float32Array(sampleRate);
for (let i = 0; i < sampleRate; i++) {
  samples[i] = Math.sin((2 * Math.PI * 440 * i) / sampleRate);
}

const audio = {
  sampleRate,
  durationSeconds: 1,
  channels: [samples],
  numberOfChannels: 1,
  numberOfFrames: sampleRate,
};

console.log('Loading model...');
await executor.ready();
console.log('Model loaded. Running inference...');

const result = await executor.transcribe(
  audio,
  { language: 'tr', noTimestamps: true, maxNewTokens: 50 },
  { modelId: 'whisper-tiny', config }
);

console.log('Result:', JSON.stringify(result, null, 2));

await executor.dispose();
