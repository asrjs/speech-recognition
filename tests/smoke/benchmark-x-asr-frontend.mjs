#!/usr/bin/env node

import assert from 'node:assert/strict';
import { XAsrJsFrontend } from '../../dist/models/x-asr/frontend.js';

const SAMPLE_RATE = 16000;
const DEFAULT_RUNS = 5;
const DEFAULT_DURATIONS = [2, 10];
const CHUNK_SAMPLES = Math.round(SAMPLE_RATE * 0.2);

function readRuns() {
  const argument = process.argv.find((value) => value.startsWith('--runs='));
  if (!argument) return DEFAULT_RUNS;
  const parsed = Number(argument.slice('--runs='.length));
  return Number.isInteger(parsed) && parsed > 0 ? parsed : DEFAULT_RUNS;
}

function readDurations() {
  const argument = process.argv.find((value) => value.startsWith('--durations='));
  if (!argument) return DEFAULT_DURATIONS;
  const values = argument
    .slice('--durations='.length)
    .split(',')
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isFinite(value) && value > 0);
  return values.length > 0 ? values : DEFAULT_DURATIONS;
}

function createSignal(durationSec) {
  const audio = new Float32Array(Math.round(durationSec * SAMPLE_RATE));
  for (let index = 0; index < audio.length; index += 1) {
    audio[index] = 0.2 * Math.sin(index / 17) + 0.03 * Math.cos(index / 43);
  }
  return audio;
}

function append(previous, chunk) {
  const combined = new Float32Array(previous.length + chunk.length);
  combined.set(previous);
  combined.set(chunk, previous.length);
  return combined;
}

function appendAmortized(previous, buffer, chunk) {
  const required = previous.length + chunk.length;
  if (required === 0) return { audio: previous, buffer };
  const capacity = buffer ? Math.floor(buffer.byteLength / Float32Array.BYTES_PER_ELEMENT) : 0;
  if (buffer && capacity >= required) {
    const view = new Float32Array(buffer, 0, required);
    view.set(chunk, previous.length);
    return { audio: view, buffer };
  }
  let nextCapacity = Math.max(required, capacity > 0 ? Math.ceil(capacity * 1.5) : required);
  while (nextCapacity < required) nextCapacity = Math.max(required, Math.ceil(nextCapacity * 1.5));
  const nextBuffer = new ArrayBuffer(nextCapacity * Float32Array.BYTES_PER_ELEMENT);
  const view = new Float32Array(nextBuffer, 0, required);
  view.set(previous);
  view.set(chunk, previous.length);
  return { audio: view, buffer: nextBuffer };
}

function summarize(values) {
  const sorted = [...values].sort((left, right) => left - right);
  return {
    medianMs: sorted[Math.floor(sorted.length / 2)],
    minMs: sorted[0],
    maxMs: sorted[sorted.length - 1],
  };
}

function runBaseline(frontend, audio) {
  let accumulated = new Float32Array(0);
  const started = performance.now();
  for (let offset = 0; offset < audio.length; offset += CHUNK_SAMPLES) {
    accumulated = append(accumulated, audio.subarray(offset, Math.min(audio.length, offset + CHUNK_SAMPLES)));
    frontend.process(accumulated);
  }
  return performance.now() - started;
}

function runIncremental(frontend, audio, amortizedAudio = true) {
  let tail = new Float32Array(0);
  let accumulated = new Float32Array(0);
  let audioBuffer;
  let sampleCount = 0;
  let frameCount = 0;
  const pieces = [];
  const expected = frontend.process(audio);
  const started = performance.now();
  for (let offset = 0; offset < audio.length; offset += CHUNK_SAMPLES) {
    const chunk = audio.subarray(offset, Math.min(audio.length, offset + CHUNK_SAMPLES));
    const appended = amortizedAudio
      ? appendAmortized(accumulated, audioBuffer, chunk)
      : { audio: append(accumulated, chunk), buffer: undefined };
    accumulated = appended.audio;
    audioBuffer = appended.buffer;
    const result = frontend.processIncremental(tail, sampleCount, chunk, frameCount);
    pieces.push(result.features);
    tail = result.tail;
    sampleCount += chunk.length;
    frameCount += result.frameCount;
  }
  const final = frontend.processIncremental(tail, sampleCount, new Float32Array(0), frameCount, true);
  pieces.push(final.features);
  const streamed = new Float32Array(pieces.reduce((sum, piece) => sum + piece.length, 0));
  let outputOffset = 0;
  for (const piece of pieces) {
    streamed.set(piece, outputOffset);
    outputOffset += piece.length;
  }
  const elapsedMs = performance.now() - started;
  assert.equal(streamed.length, expected.length, 'incremental feature length mismatch');
  let maxAbs = 0;
  for (let index = 0; index < expected.length; index += 1) {
    maxAbs = Math.max(maxAbs, Math.abs((streamed[index] ?? 0) - (expected[index] ?? 0)));
  }
  assert.ok(maxAbs < 1e-6, `incremental feature parity mismatch: maxAbs=${maxAbs}`);
  return { elapsedMs, maxAbs };
}

const runs = readRuns();
const durations = readDurations();
const report = {
  schema: 'asrjs.x-asr.frontend-streaming-benchmark.v1',
  sampleRate: SAMPLE_RATE,
  chunkSamples: CHUNK_SAMPLES,
  runs,
  scenarios: [],
};

for (const durationSec of durations) {
  const audio = createSignal(durationSec);
  const frontend = new XAsrJsFrontend();
  runBaseline(frontend, audio);
  runIncremental(frontend, audio, false);
  runIncremental(frontend, audio, true);
  const baseline = [];
  const frontendOnly = [];
  const incremental = [];
  let maxAbs = 0;
  for (let run = 0; run < runs; run += 1) {
    baseline.push(runBaseline(frontend, audio));
    const frontendOnlyResult = runIncremental(frontend, audio, false);
    frontendOnly.push(frontendOnlyResult.elapsedMs);
    const result = runIncremental(frontend, audio, true);
    incremental.push(result.elapsedMs);
    maxAbs = Math.max(maxAbs, result.maxAbs);
  }
  const baselineSummary = summarize(baseline);
  const frontendOnlySummary = summarize(frontendOnly);
  const incrementalSummary = summarize(incremental);
  report.scenarios.push({
    durationSec,
    chunkCount: Math.ceil(audio.length / CHUNK_SAMPLES),
    baseline: baselineSummary,
    frontendOnly: frontendOnlySummary,
    incremental: incrementalSummary,
    frontendOnlySpeedup: baselineSummary.medianMs / frontendOnlySummary.medianMs,
    speedup: baselineSummary.medianMs / incrementalSummary.medianMs,
    maxAbs,
  });
}

if (process.argv.includes('--json')) {
  console.log(JSON.stringify(report, null, 2));
} else {
  console.log(`X-ASR streaming frontend benchmark (chunk=${CHUNK_SAMPLES} samples, runs=${runs})`);
  for (const scenario of report.scenarios) {
    console.log(
      `  ${scenario.durationSec}s / ${scenario.chunkCount} chunks: ` +
        `baseline median=${scenario.baseline.medianMs.toFixed(2)}ms, ` +
        `frontend-only median=${scenario.frontendOnly.medianMs.toFixed(2)}ms, ` +
        `incremental median=${scenario.incremental.medianMs.toFixed(2)}ms, ` +
        `frontend speedup=${scenario.frontendOnlySpeedup.toFixed(2)}x, ` +
        `combined speedup=${scenario.speedup.toFixed(2)}x, maxAbs=${scenario.maxAbs}`,
    );
  }
}
