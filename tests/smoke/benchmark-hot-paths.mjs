#!/usr/bin/env node

import assert from 'node:assert/strict';
import os from 'node:os';
import { performance } from 'node:perf_hooks';

const DEFAULT_RUNS = 20;
const SAMPLE_RATE = 16000;

function printHelp() {
  console.log(`Hot-path benchmark and correctness harness

Usage: npm run benchmark:hot-paths -- [--runs=20] [--json]

Options:
  --runs=N  Timed iterations per scenario (default: ${DEFAULT_RUNS})
  --json    Emit machine-readable JSON instead of the short report
  --help    Show this help

The command measures the current checkout only. Run it on the base and
candidate commits with the same Node version and hardware before accepting a
hot-path optimization.`);
}

function readRuns() {
  const argument = process.argv.find((value) => value.startsWith('--runs='));
  if (!argument) {
    return DEFAULT_RUNS;
  }
  const parsed = Number(argument.slice('--runs='.length));
  return Number.isInteger(parsed) && parsed > 0 ? parsed : DEFAULT_RUNS;
}

function createTranscript(index) {
  const tokens = Array.from({ length: 48 }, (_, tokenIndex) => ({
    index: tokenIndex,
    id: index * 100 + tokenIndex,
    text: `token-${index}-${tokenIndex}`,
    startTime: tokenIndex * 0.01,
    endTime: tokenIndex * 0.01 + 0.01,
  }));
  const words = Array.from({ length: 12 }, (_, wordIndex) => ({
    index: wordIndex,
    text: `word-${index}-${wordIndex}`,
    startTime: wordIndex * 0.04,
    endTime: wordIndex * 0.04 + 0.04,
  }));
  const segments = Array.from({ length: 4 }, (_, segmentIndex) => ({
    index: segmentIndex,
    text: `segment-${index}-${segmentIndex}`,
    startTime: segmentIndex * 0.12,
    endTime: segmentIndex * 0.12 + 0.12,
  }));

  return {
    text: `window ${index} transcript`,
    warnings: [],
    meta: {
      detailLevel: 'detailed',
      isFinal: true,
      sampleRate: SAMPLE_RATE,
      durationSeconds: 0.48,
    },
    segments,
    words,
    tokens,
  };
}

function createInputs(PcmAudioBuffer) {
  const frameCount = SAMPLE_RATE * 2;
  const left = new Float32Array(frameCount);
  const right = new Float32Array(frameCount);
  for (let index = 0; index < frameCount; index += 1) {
    left[index] = Math.sin(index * 0.01) * 0.25;
    right[index] = Math.cos(index * 0.013) * 0.15;
  }

  const logits = new Float32Array(4096);
  for (let index = 0; index < logits.length; index += 1) {
    logits[index] = Math.sin(index * 0.017) * 3;
  }

  return {
    mergeResults: Array.from({ length: 48 }, (_, index) => createTranscript(index)),
    normalizationText:
      'İstanbul’da hızlı konuşma — ölçüm için normalize edilen bir Türkçe cümle. '.repeat(24),
    audio: new PcmAudioBuffer({ sampleRate: SAMPLE_RATE, channels: [left, right] }),
    logits,
  };
}

function runCorrectnessChecks(inputs, runtime) {
  const emptyMerge = runtime.mergeTranscriptResults([]);
  assert.equal(emptyMerge.text, '');
  assert.equal(emptyMerge.meta.segmentCount, 0);

  const missingAlignment = runtime.mergeTranscriptResults([
    {
      text: 'short',
      warnings: [],
      meta: { detailLevel: 'text', isFinal: true },
    },
  ]);
  assert.equal(missingAlignment.tokens, undefined);

  assert.equal(runtime.argmax(new Float32Array([1, 9]), 5, 2), 5);

  const invalidQuality = runtime.tokenQualityFromLogits(
    new Float32Array([Number.NaN, Number.NaN]),
    0,
    2,
  );
  assert.equal(invalidQuality.confidence, 0);
  assert.equal(invalidQuality.logProb, Number.NEGATIVE_INFINITY);

  const mono = inputs.audio.toMono();
  assert.equal(mono.numberOfChannels, 1);
  assert.ok(Math.abs(mono.channels[0][0] - 0.075) < 1e-6);

  const shortAudio = inputs.audio.sliceFrames(0, 1);
  assert.equal(shortAudio.numberOfFrames, 1);

  const processor = new runtime.WhisperTimestampLogitProcessor({
    eosTokenId: 2,
    noTimestampsTokenId: 3,
    timestampBegin: 16,
    suppressTokens: [99999],
  });
  const shortLogits = new Float32Array(32);
  processor.process(shortLogits, [], 0);
  assert.equal(shortLogits[0], Number.NEGATIVE_INFINITY);

  return {
    checks: [
      'empty transcript merge',
      'missing token alignment',
      'out-of-range argmax',
      'NaN logits fallback',
      'multi-channel downmix',
      'short audio segment',
      'out-of-range Whisper suppression token',
    ],
  };
}

function summarize(samples) {
  const sorted = [...samples].sort((left, right) => left - right);
  const percentile = (fraction) =>
    sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1)];
  const total = samples.reduce((sum, value) => sum + value, 0);
  return {
    count: samples.length,
    mean_ms: total / samples.length,
    p50_ms: percentile(0.5),
    p90_ms: percentile(0.9),
    min_ms: sorted[0],
    max_ms: sorted[sorted.length - 1],
  };
}

function benchmark(label, fn, runs) {
  for (let index = 0; index < Math.min(5, runs); index += 1) {
    fn();
  }

  const samples = [];
  for (let index = 0; index < runs; index += 1) {
    const started = performance.now();
    fn();
    samples.push(performance.now() - started);
  }
  return { label, ...summarize(samples) };
}

function runBenchmarks(inputs, runs, runtime) {
  const timestampProcessor = new runtime.WhisperTimestampLogitProcessor({
    eosTokenId: 2,
    noTimestampsTokenId: 3,
    timestampBegin: 2048,
    suppressTokens: [1, 7, 11, 13, 17, 19],
    beginSuppressTokens: [23, 29, 31],
  });

  return [
    benchmark(
      'transcript-normalization',
      () => runtime.normalizeBenchmarkTranscript(inputs.normalizationText),
      runs,
    ),
    benchmark(
      'streaming-transcript-merge',
      () => runtime.mergeTranscriptResults(inputs.mergeResults),
      runs,
    ),
    benchmark('audio-stereo-downmix', () => inputs.audio.toMono(), runs),
    benchmark('inference-argmax', () => runtime.argmax(inputs.logits), runs),
    benchmark(
      'inference-token-quality',
      () => runtime.tokenQualityFromLogits(inputs.logits, 123, inputs.logits.length),
      runs,
    ),
    benchmark(
      'whisper-logit-processor',
      () => {
        const logits = new Float32Array(inputs.logits);
        timestampProcessor.process(logits, [2048, 2050], 2);
      },
      runs,
    ),
  ];
}

if (process.argv.includes('--help')) {
  printHelp();
  process.exit(0);
}

const runs = readRuns();
const [audioRuntime, mathRuntime, mergeRuntime, processorRuntime, benchmarkRuntime] =
  await Promise.all([
    import('../../dist/audio/audio.js'),
    import('../../dist/inference/math.js'),
    import('../../dist/inference/streaming/merge.js'),
    import('../../dist/models/whisper-seq2seq/processors.js'),
    import('../../dist/bench.js'),
  ]);
const runtime = {
  PcmAudioBuffer: audioRuntime.PcmAudioBuffer,
  argmax: mathRuntime.argmax,
  tokenQualityFromLogits: mathRuntime.tokenQualityFromLogits,
  mergeTranscriptResults: mergeRuntime.mergeTranscriptResults,
  WhisperTimestampLogitProcessor: processorRuntime.WhisperTimestampLogitProcessor,
  normalizeBenchmarkTranscript: benchmarkRuntime.normalizeBenchmarkTranscript,
};
const inputs = createInputs(runtime.PcmAudioBuffer);
const correctness = runCorrectnessChecks(inputs, runtime);
const report = {
  schema_version: 1,
  generated_at: new Date().toISOString(),
  node: process.version,
  platform: `${os.platform()}-${os.arch()}`,
  runs,
  correctness,
  benchmarks: runBenchmarks(inputs, runs, runtime),
};

if (process.argv.includes('--json')) {
  console.log(JSON.stringify(report, null, 2));
} else {
  console.log('ASR.js hot-path benchmark');
  console.log(`node=${report.node} platform=${report.platform} runs=${report.runs}`);
  console.log(`correctness checks=${report.correctness.checks.length} passed`);
  for (const result of report.benchmarks) {
    console.log(
      `${result.label}: mean=${result.mean_ms.toFixed(3)}ms p50=${result.p50_ms.toFixed(3)}ms ` +
        `p90=${result.p90_ms.toFixed(3)}ms min=${result.min_ms.toFixed(3)}ms max=${result.max_ms.toFixed(3)}ms`,
    );
  }
}
