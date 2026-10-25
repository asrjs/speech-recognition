import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import * as ort from 'onnxruntime-node';
import { decodeWavToMonoPcm } from '../../dist/audio/wav.js';
import { normalizeWav2Vec2Waveform } from '../../dist/models/wav2vec2/executor.js';
import { OmnilingualCtcTokenizer } from '../../dist/models/omnilingual-ctc/tokenizer.js';
import { CtcDecoder } from '../../dist/ctc/index.js';

const root = process.cwd();
const wavPath = process.argv[2] ?? `${root}/tests/fixtures/jfk2.en.wav`;
const modelPath = `${root}/models/onnx/omnilingual-ctc-300m/model.int8.onnx`;
const tokensPath = `${root}/models/onnx/omnilingual-ctc-300m/tokens.txt`;
const wavBytes = readFileSync(wavPath);
const decodedWav = decodeWavToMonoPcm(wavBytes);
const pcm = decodedWav.pcm;
const normalized = normalizeWav2Vec2Waveform(pcm);
const f32le = (values) => Buffer.from(values.buffer, values.byteOffset, values.byteLength);
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const stats = (values) => {
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) sum += value;
  const mean = sum / values.length;
  let squares = 0;
  for (const value of values) {
    squares += (value - mean) ** 2;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return { mean, std: Math.sqrt(squares / values.length), min, max };
};
const tokenizer = OmnilingualCtcTokenizer.fromText(readFileSync(tokensPath, 'utf8'));
const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });
const outputs = await session.run({ x: new ort.Tensor('float32', normalized, [1, normalized.length]) });
const logits = outputs.logits;
const frameCount = Number(logits.dims[1]);
const vocabSize = Number(logits.dims[2]);
const data = logits.data instanceof Float32Array ? logits.data : Float32Array.from(logits.data);
const decoder = new CtcDecoder({ blankId: tokenizer.blankId, vocabSize, tokenizer });
const decoded = decoder.decodeFromLogits(data, frameCount, { outFrames: frameCount });
const firstFrames = Array.from({ length: Math.min(6, frameCount) }, (_, frame) => {
  const row = data.subarray(frame * vocabSize, (frame + 1) * vocabSize);
  const top = [];
  for (let index = 0; index < row.length; index++) {
    const value = row[index];
    let position = top.findIndex((item) => value > item.value);
    if (position < 0) position = top.length;
    if (position < 5) {
      top.splice(position, 0, { id: index, value });
      if (top.length > 5) top.pop();
    }
  }
  return { argmax: top[0]?.id, top };
});
const result = {
  status: 'CPU_TRACE_NOT_ACCEPTANCE',
  runtime: { node: process.version, onnxruntime: ort.version, providers: ['cpu'] },
  input: { path: wavPath, wavSha256: sha256(wavBytes), sampleRate: decodedWav.sampleRate, channels: decodedWav.numberOfChannels, samples: pcm.length },
  pcm: { sha256Float32le: sha256(f32le(pcm)), ...stats(pcm) },
  normalizedPcm: { sha256Float32le: sha256(f32le(normalized)), ...stats(normalized) },
  artifacts: { modelPath, modelSha256: sha256(readFileSync(modelPath)), tokensSha256: sha256(readFileSync(tokensPath)) },
  graph: { input: session.inputNames, output: session.outputNames, frameCount, vocabSize },
  tokenizer: { vocabSize: tokenizer.vocabSize, blankId: tokenizer.blankId },
  firstFrames,
  collapsedIds: decoded.collapsedIds,
  text: decoded.text,
};
writeFileSync(`${root}/tmp/omnilingual-diagnosis/node-cpu-trace-${wavPath.split(/[\\/]/).pop().replace(/\.[^.]+$/, '')}.json`, `${JSON.stringify(result, null, 2)}\n`);
await session.release();
console.log(JSON.stringify({ ...result, collapsedIds: result.collapsedIds.slice(0, 100) }, null, 2));
