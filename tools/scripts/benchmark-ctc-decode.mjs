#!/usr/bin/env node

// Measures argmaxAndSelectedLogProbs at SenseVoiceSmall scale (187 output
// frames x 25055 vocabulary, the 11.3 s jfk-short fixture shape). Run after
// `npm run build` on the base and candidate commits with the same Node
// version before accepting a change.
import { performance } from 'node:perf_hooks';

const { argmaxAndSelectedLogProbs } = await import('../../dist/ctc/index.js');

const FRAME_COUNT = 187;
const VOCAB_SIZE = 25055;
const RUNS = 15;

const logits = new Float32Array(FRAME_COUNT * VOCAB_SIZE);
for (let index = 0; index < logits.length; index += 1) {
  logits[index] =
    Math.log(Math.abs(Math.sin(index * 0.0007)) + 1e-9) * 0.5 -
    (index % VOCAB_SIZE) * 0.001;
}

const times = [];
let reference = null;
for (let run = 0; run < RUNS; run += 1) {
  const started = performance.now();
  const result = argmaxAndSelectedLogProbs(logits, FRAME_COUNT, VOCAB_SIZE);
  times.push(performance.now() - started);
  if (run === 0) reference = result;
}
times.sort((a, b) => a - b);
console.log(`argmax+logsoftmax p50=${times[Math.floor(times.length / 2)].toFixed(2)}ms min=${times[0].toFixed(2)}ms runs=${RUNS}`);
console.log(`sample ids=${reference.frameIds.slice(0, 5).join(',')}`);
console.log(`sample logProbs=${reference.selectedLogProbs.slice(0, 3).map((v) => v.toFixed(9)).join(',')}`);

