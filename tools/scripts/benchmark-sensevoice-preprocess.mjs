#!/usr/bin/env node

// Measures SenseVoiceJsPreprocessor.processOfficial at the jfk-short scale:
// 11.29 s at 16 kHz, 80 mel bins, nFft 512, hop 160, hamming window with DC
// removal and frame preemphasis, followed by LFR (7:6) and CMVN. Run after
// `npm run build` on the base and candidate commits with the same Node
// version before accepting a change.
import { performance } from 'node:perf_hooks';
import fs from 'node:fs';

const { SenseVoiceJsPreprocessor } = await import('../../dist/models/sensevoice/frontend.js');

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

// Minimal synthetic CMVN matching the official shape (560 means/scales).
const cmvnDim = 560;
const means = new Float32Array(cmvnDim);
const scales = new Float32Array(cmvnDim).fill(1);
const cmvn = { means, scales };

const preprocessor = new SenseVoiceJsPreprocessor();
const warm = preprocessor.processOfficial(audio, cmvn);
console.log(`frames=${warm.frameCount} featureSize=${warm.featureSize} sample0=${warm.features[0]?.toFixed(6)}`);

const times = [];
let reference = null;
for (let run = 0; run < RUNS; run += 1) {
  const started = performance.now();
  const output = preprocessor.processOfficial(audio, cmvn);
  times.push(performance.now() - started);
  if (run === 0) reference = output;
}
times.sort((a, b) => a - b);
let checksum = 0;
for (let index = 0; index < reference.features.length; index += 1) checksum += reference.features[index];
console.log(`processOfficial p50=${times[Math.floor(times.length / 2)].toFixed(2)}ms min=${times[0].toFixed(2)}ms runs=${RUNS} checksum=${checksum.toFixed(3)}`);
