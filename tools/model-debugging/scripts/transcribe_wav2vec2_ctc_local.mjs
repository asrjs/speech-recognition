/** Transcribes a local wav2vec2 CTC folder with the library decode path. */
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-node';

import { decodeWavToMonoPcm } from '../../../dist/audio/wav.js';
import { CtcDecoder } from '../../../dist/ctc/index.js';
import { normalizeWav2Vec2Waveform } from '../../../dist/models/wav2vec2/executor.js';

function argument(name, fallback) {
  const prefix = `--${name}=`;
  const found = process.argv.find((value) => value.startsWith(prefix));
  return found ? found.slice(prefix.length) : fallback;
}

const artifactDir = argument('model-dir');
const audioPath = argument('audio');
if (!artifactDir || !audioPath) {
  throw new Error('Required: --model-dir=<local folder> --audio=<16 kHz mono WAV>');
}

const vocab = JSON.parse(readFileSync(`${artifactDir}/vocab.json`, 'utf8'));
const idToToken = new Array(Object.keys(vocab).length).fill('');
for (const [token, id] of Object.entries(vocab)) idToToken[id] = token;
const blankId = idToToken.indexOf('[PAD]');
const tokenizer = {
  decode(ids) {
    let text = '';
    for (const id of ids) {
      const token = idToToken[id];
      if (!token || token === '[PAD]' || token === '<s>' || token === '</s>' || token === '[UNK]') continue;
      text += token === '|' ? ' ' : token;
    }
    return text;
  },
  decodeTokenPiece(id) {
    return tokenizer.decode([id]);
  },
};

const pcm = decodeWavToMonoPcm(readFileSync(audioPath)).pcm;
const session = await ort.InferenceSession.create(`${artifactDir}/model.onnx`, {
  executionProviders: ['cpu'],
});
const started = performance.now();
const outputs = await session.run({
  input_values: new ort.Tensor('float32', normalizeWav2Vec2Waveform(pcm), [1, pcm.length]),
});
const elapsed = performance.now() - started;
const logits = outputs.logits;
const frames = Number(logits.dims[1]);
const vocabSize = idToToken.length;
const data = logits.data instanceof Float32Array ? logits.data : Float32Array.from(logits.data);
const decoded = new CtcDecoder({ blankId, vocabSize, tokenizer }).decodeFromLogits(
  data.subarray(0, frames * vocabSize),
  frames,
  { outFrames: frames },
);

console.log(
  JSON.stringify({
    status: 'RUN_COMPLETED_NOT_MODEL_ACCEPTANCE',
    host: { runtime: 'node', version: process.versions.node },
    backend: { id: 'native-cpu', engine: 'onnxruntime-node', provider: 'cpu' },
    input: { path: audioPath, sampleRate: 16000, samples: pcm.length },
    transcript: {
      canonical: {
        text: decoded.text.trim(),
        tokens: decoded.collapsedIds.map((id) => ({ id })),
        warnings: [],
        meta: {
          metrics: {
            totalMs: Math.round(elapsed * 1000) / 1000,
            encoderFrameCount: frames,
            emittedTokenCount: decoded.collapsedIds.length,
          },
        },
      },
    },
  }),
);
