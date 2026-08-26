#!/usr/bin/env node

import { GigaAmJsPreprocessor } from '../../dist/models/gigaam-ctc/frontend.js';

const SAMPLE_RATE = 16000;
const DEFAULT_DURATIONS = [1, 10, 30];
const DEFAULT_RUNS = 5;

function readNumberArg(name, fallback) {
  const prefix = `--${name}=`;
  const arg = process.argv.find((value) => value.startsWith(prefix));
  if (!arg) return fallback;
  const parsed = Number(arg.slice(prefix.length));
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function readDurations() {
  const arg = process.argv.find((value) => value.startsWith('--durations='));
  if (!arg) return DEFAULT_DURATIONS;
  const durations = arg
    .slice('--durations='.length)
    .split(',')
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isFinite(value) && value > 0);
  return durations.length > 0 ? durations : DEFAULT_DURATIONS;
}

function createSignal(durationSec) {
  const sampleCount = Math.round(durationSec * SAMPLE_RATE);
  const audio = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) {
    audio[i] =
      0.2 * Math.sin((2 * Math.PI * 440 * i) / SAMPLE_RATE) +
      0.05 * Math.sin((2 * Math.PI * 3000 * i) / SAMPLE_RATE);
  }
  return audio;
}

function summarize(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    avg: values.reduce((sum, value) => sum + value, 0) / values.length,
    median: sorted[Math.floor(sorted.length / 2)],
    min: sorted[0],
    max: sorted[sorted.length - 1],
  };
}

const durations = readDurations();
const runs = Math.round(readNumberArg('runs', DEFAULT_RUNS));
const processor = new GigaAmJsPreprocessor();

console.log('GigaAM frontend (n_fft=320 Bluestein) benchmark');
console.log(`sample_rate=${SAMPLE_RATE} runs=${runs}`);
console.log('');

for (const durationSec of durations) {
  const audio = createSignal(durationSec);
  const expectedFrames = Math.floor((audio.length - 320) / 160) + 1;
  processor.process(audio);

  const times = [];
  for (let run = 0; run < runs; run++) {
    const started = performance.now();
    const result = processor.process(audio);
    const elapsed = performance.now() - started;
    if (result.frameCount !== expectedFrames) {
      throw new Error(`Unexpected frame count for ${durationSec}s audio: ${result.frameCount}`);
    }
    times.push(elapsed);
  }

  const { avg, median, min, max } = summarize(times);
  const rtf = avg / (durationSec * 1000);
  console.log(
    `  ${durationSec.toFixed(1)}s audio: avg=${avg.toFixed(1)}ms median=${median.toFixed(1)}ms min=${min.toFixed(
      1,
    )}ms max=${max.toFixed(1)}ms rtf=${rtf.toFixed(4)} rtfx=${(1 / rtf).toFixed(0)}`,
  );
}
