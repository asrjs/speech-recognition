/**
 * Transcribes a local ParakeetForCTC folder with the public runtime path.
 *
 * Prints the CLI-shaped JSON that tools/model-debugging/scripts/port_manifest.py
 * compares against a saved oracle. Nothing is downloaded.
 */
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-node';

import { decodeWavToMonoPcm } from '../../../dist/audio/wav.js';
import { CtcDecoder } from '../../../dist/ctc/index.js';
import { ParakeetCtcJsPreprocessor } from '../../../dist/models/parakeet-ctc/frontend.js';
import { ParakeetCtcTokenizer } from '../../../dist/models/parakeet-ctc/tokenizer.js';

function argument(name, fallback) {
  const prefix = `--${name}=`;
  const found = process.argv.find((value) => value.startsWith(prefix));
  return found ? found.slice(prefix.length) : fallback;
}

const artifactDir = argument('model-dir');
const audioPath = argument('audio');
const vocabSize = Number(argument('vocab-size', '1025'));
const blankId = Number(argument('blank-id', '1024'));
if (!artifactDir || !audioPath) {
  throw new Error('Required: --model-dir=<local folder> --audio=<16 kHz mono WAV>');
}

const pcm = decodeWavToMonoPcm(readFileSync(audioPath)).pcm;
const tokenizer = ParakeetCtcTokenizer.fromTokenizerJsonText(
  readFileSync(`${artifactDir}/tokenizer.json`, 'utf8'),
);
const batch = new ParakeetCtcJsPreprocessor().process(pcm);
const frames = batch.frameCount;
const mels = batch.featureSize;

// The shared frontend is mel-major; the graph wants [batch, frames, mels].
const frameMajor = new Float32Array(mels * frames);
for (let mel = 0; mel < mels; mel += 1) {
  for (let frame = 0; frame < frames; frame += 1) {
    frameMajor[frame * mels + mel] = batch.features[mel * frames + frame] ?? 0;
  }
}

const session = await ort.InferenceSession.create(`${artifactDir}/model.onnx`, {
  executionProviders: ['cpu'],
});
const started = performance.now();
const outputs = await session.run({
  input_features: new ort.Tensor('float32', frameMajor, [1, frames, mels]),
  attention_mask: new ort.Tensor('int64', new BigInt64Array(frames).fill(1n), [1, frames]),
});
const elapsed = performance.now() - started;
const logits = outputs.logits;
const outFrames = Number(logits.dims[1]);
const data = logits.data instanceof Float32Array ? logits.data : Float32Array.from(logits.data);
const decoded = new CtcDecoder({ blankId, vocabSize, tokenizer }).decodeFromLogits(
  data.subarray(0, outFrames * vocabSize),
  outFrames,
  { outFrames },
);

console.log(
  JSON.stringify({
    status: 'RUN_COMPLETED_NOT_MODEL_ACCEPTANCE',
    host: { runtime: 'node', version: process.versions.node },
    backend: { id: 'wasm', engine: 'onnxruntime-node', provider: 'cpu' },
    input: { path: audioPath, sampleRate: 16000, samples: pcm.length },
    transcript: {
      canonical: {
        text: decoded.text.trim(),
        tokens: decoded.collapsedIds.map((id) => ({ id })),
        warnings: [],
        meta: {
          metrics: {
            totalMs: Math.round(elapsed * 1000) / 1000,
            encoderFrameCount: outFrames,
            emittedTokenCount: decoded.collapsedIds.length,
          },
        },
      },
    },
  }),
);
