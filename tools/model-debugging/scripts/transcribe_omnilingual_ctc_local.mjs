/**
 * Transcribes a local omnilingual CTC folder with the library path.
 *
 * Prints the CLI-shaped JSON that tools/model-debugging/scripts/port_manifest.py
 * compares against a recorded baseline. Nothing is downloaded.
 */
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-node';

import { decodeWavToMonoPcm } from '../../../dist/audio/wav.js';
import { OmnilingualCtcTokenizer } from '../../../dist/models/omnilingual-ctc/tokenizer.js';
import { CtcDecoder } from '../../../dist/ctc/index.js';
import { normalizeWav2Vec2Waveform } from '../../../dist/models/wav2vec2/executor.js';

function argument(name, fallback) {
  const prefix = `--${name}=`;
  const found = process.argv.find((value) => value.startsWith(prefix));
  return found ? found.slice(prefix.length) : fallback;
}

const artifactDir = argument('model-dir');
const audioPath = argument('audio');
const graph = argument('graph', 'model.int8.onnx');
if (!artifactDir || !audioPath) {
  throw new Error('Required: --model-dir=<local folder> --audio=<16 kHz mono WAV>');
}

const pcm = decodeWavToMonoPcm(readFileSync(audioPath)).pcm;
const tokenizer = OmnilingualCtcTokenizer.fromText(
  readFileSync(`${artifactDir}/tokens.txt`, 'utf8'),
);
const vocabSize = tokenizer.vocabSize;
const session = await ort.InferenceSession.create(`${artifactDir}/${graph}`, {
  executionProviders: ['cpu'],
});
const started = performance.now();
const outputs = await session.run({
  x: new ort.Tensor('float32', normalizeWav2Vec2Waveform(pcm), [1, pcm.length]),
});
const elapsed = performance.now() - started;
const logits = outputs.logits;
const frames = Number(logits.dims[1]);
const data = logits.data instanceof Float32Array ? logits.data : Float32Array.from(logits.data);
const decoded = new CtcDecoder({
  blankId: tokenizer.blankId,
  vocabSize,
  tokenizer,
}).decodeFromLogits(data.subarray(0, frames * vocabSize), frames, { outFrames: frames });

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
            encoderFrameCount: frames,
            emittedTokenCount: decoded.collapsedIds.length,
          },
        },
      },
    },
  }),
);
