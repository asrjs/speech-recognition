import * as fs from 'fs';
import { WhisperOnnxExecutor } from '../src/models/whisper-seq2seq/executor.js';
import type {
  WhisperArtifactSource,
  WhisperSeq2SeqModelConfig,
} from '../src/models/whisper-seq2seq/types.js';

function readWavToFloat32(path: string): { samples: Float32Array; sampleRate: number } {
  const buf = fs.readFileSync(path);
  // RIFF/WAVE header parsing
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const sampleRate = view.getUint32(24, true);
  const bitsPerSample = view.getUint16(34, true);
  const dataOffset = 44; // standard offset
  const dataSize = buf.length - dataOffset;

  const numSamples = Math.floor(dataSize / (bitsPerSample / 8));
  const samples = new Float32Array(numSamples);

  if (bitsPerSample === 16) {
    for (let i = 0; i < numSamples; i++) {
      const val = view.getInt16(dataOffset + i * 2, true);
      samples[i] = val / 32768.0;
    }
  } else if (bitsPerSample === 32) {
    for (let i = 0; i < numSamples; i++) {
      const val = view.getInt32(dataOffset + i * 4, true);
      samples[i] = val / 2147483648.0;
    }
  } else {
    throw new Error(`Unsupported bits per sample: ${bitsPerSample}`);
  }

  return { samples, sampleRate };
}

const config: WhisperSeq2SeqModelConfig = {
  ecosystem: 'openai',
  architecture: 'whisper-seq2seq',
  melBins: 80,
  sampleRate: 16000,
  maxSourcePositions: 3000,
  maxTargetPositions: 448,
  vocabularySize: 51865,
  languages: ['tr', 'en'],
  processorArchitecture: 'whisper-mel',
  encoderArchitecture: 'whisper-transformer',
  decoderArchitecture: 'transformer-decoder',
  tokenizer: { kind: 'tiktoken', vocabSize: 51865 },
};

const source: WhisperArtifactSource = {
  kind: 'direct',
  artifacts: {
    encoderUrl: 'file:///tmp/whisper-tiny-onnx/encoder_model.onnx',
    decoderUrl: 'file:///tmp/whisper-tiny-onnx/decoder_model_merged.onnx',
    tokenizerUrl: 'file:///tmp/whisper-tiny-onnx/tokenizer.json',
  },
};

const executor = new WhisperOnnxExecutor(
  'whisper-tiny',
  { ecosystem: 'openai', family: 'whisper-seq2seq', task: 'transcribe' },
  config,
  'wasm',
  { source }
);

const { samples, sampleRate } = readWavToFloat32('/tmp/librivox_16k.wav');
const durationSeconds = samples.length / sampleRate;

console.log(`Audio: ${sampleRate}Hz, ${samples.length} samples, ${durationSeconds.toFixed(2)}s`);

const audio = {
  sampleRate,
  durationSeconds,
  channels: [samples],
  numberOfChannels: 1,
  numberOfFrames: samples.length,
};

console.log('Loading model...');
await executor.ready();
console.log('Model loaded. Running inference...');

const result = await executor.transcribe(
  audio,
  { language: 'en', noTimestamps: false, maxNewTokens: 200 },
  { modelId: 'whisper-tiny', config }
);

console.log('\n=== Result ===');
console.log('Language:', result.language);
console.log('Text:', result.utteranceText);
console.log('Segments:', JSON.stringify(result.segments, null, 2));
console.log('Token count:', result.tokens?.length ?? 0);

await executor.dispose();
