#!/usr/bin/env node

// Measures GigaAmJsPreprocessor.process (delegating to MedAsrJsPreprocessor)
// at the GigaAM example.wav scale: 11.29 s at 16 kHz, 64 mel bins, nFft 320,
// hop 160. Run after `npm run build` on the base and candidate commits with
// the same Node version before accepting a change.
import { performance } from 'node:perf_hooks';

const { GigaAmJsPreprocessor } = await import('../../dist/models/gigaam-ctc/frontend.js');

const SAMPLE_RATE = 16000;
const DURATION_SEC = 11.29;
const RUNS = 20;

const sampleCount = Math.floor(SAMPLE_RATE * DURATION_SEC);
const audio = new Float32Array(sampleCount);
for (let index = 0; index < sampleCount; index += 1) {
  audio[index] =
    0.3 * Math.sin((2 * Math.PI * 220 * index) / SAMPLE_RATE) +
    0.1 * Math.sin((2 * Math.PI * 1700 * index) / SAMPLE_RATE);
}

const preprocessor = new GigaAmJsPreprocessor();
const warm = preprocessor.process(audio);
console.log(`frames=${warm.frameCount} mels=${warm.featureSize} sample0=${warm.features[0]?.toFixed(6)}`);

const times = [];
let reference = null;
for (let run = 0; run < RUNS; run += 1) {
  const started = performance.now();
  const output = preprocessor.process(audio);
  times.push(performance.now() - started);
  if (run === 0) reference = output;
}
times.sort((a, b) => a - b);
let checksum = 0;
for (let index = 0; index < reference.features.length; index += 1) checksum += reference.features[index];
console.log(`process p50=${times[Math.floor(times.length / 2)].toFixed(2)}ms min=${times[0].toFixed(2)}ms runs=${RUNS} checksum=${checksum.toFixed(3)}`);

