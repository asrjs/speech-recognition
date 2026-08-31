#!/usr/bin/env node

// Nemotron 3.5 streaming 0.6B INT4 WASM benchmark (Node, ORT Web WASM 1.29).
//
// Measures warmed end-to-end transcription of the committed fixtures through
// the PUBLIC API (createBuiltInSpeechRuntime + nemotron-rnnt family + direct
// INT4 singles source), breaking down preprocess/encode/decode latency, RTFx,
// determinism, and WASM memory. The executor already reports the phase
// breakdown in native.metrics, so this script only orchestrates repeated runs
// and aggregates.
//
// Usage:
//   npm run build
//   node tools/scripts/benchmark-nemotron-wasm.mjs \
//     [--model-dir N:/models/onnx/nemo/nemotron-3.5-asr-streaming-int4-singles] \
//     [--runs-jfk 10] [--runs-librivox 3] [--json out.json]
//
// Env fallback: NEMOTRON_INT4_DIR (same env the gated fixture tests use).
// Exits non-zero if the transcript oracle or cross-run determinism fails.
import { performance } from 'node:perf_hooks';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..');

const { createWasmBackend } = await import('../../dist/index.js');
const { createNemotronRnntModelFamily } = await import(
  '../../dist/models/nemotron-rnnt/index.js'
);
const { createNemotronPresetFactory } = await import('../../dist/presets/nemotron/index.js');
const { createBuiltInSpeechRuntime } = await import('../../dist/runtime/builtins.js');
const { parseNemotronRnntConfig } = await import(
  '../../dist/models/nemotron-rnnt/config.js'
);

const FIXTURES_DIR = path.resolve(REPO_ROOT, 'tools', 'data', 'fixtures', 'audio');
const DEFAULT_OUT = path.resolve(
  REPO_ROOT,
  'tools',
  'data',
  'results',
  'nemotron',
  'nemotron-3.5-wasm-benchmark-2026-08-31.json',
);

const JFK_ORACLE = {
  tokenCount: 41,
  text: 'And so my fellow Americans ask not what your country can do for you at what you can do for your country',
};
const LIBRIVOX_ORACLE = {
  tokenCount: 88,
  contains: 'Librivox recordings are in the public domain',
};

function parseArgs(argv) {
  const args = {
    modelDir: process.env.NEMOTRON_INT4_DIR ?? 'N:/models/onnx/nemo/nemotron-3.5-asr-streaming-int4-singles',
    runsJfk: 10,
    runsLibrivox: 3,
    threads: 6,
    json: DEFAULT_OUT,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const value = argv[i + 1];
    if (argv[i] === '--model-dir' && value) args.modelDir = value;
    else if (argv[i] === '--runs-jfk' && value) args.runsJfk = Number(value);
    else if (argv[i] === '--runs-librivox' && value) args.runsLibrivox = Number(value);
    else if (argv[i] === '--threads' && value) args.threads = Number(value);
    else if (argv[i] === '--json' && value) args.json = value;
  }
  return args;
}

function readWavPcm16(filePath) {
  const buf = fs.readFileSync(filePath);
  if (buf.toString('ascii', 0, 4) !== 'RIFF') {
    throw new Error(`Not a RIFF file: ${filePath}`);
  }
  if (buf.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error(`Not a WAVE file: ${filePath}`);
  }
  let offset = 12;
  let sampleRate = 0;
  let channels = 0;
  let bitsPerSample = 0;
  let dataOffset = 0;
  let dataLength = 0;
  while (offset < buf.length - 8) {
    const id = buf.toString('ascii', offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    if (id === 'fmt ') {
      channels = buf.readUInt16LE(offset + 10);
      sampleRate = buf.readUInt32LE(offset + 12);
      bitsPerSample = buf.readUInt16LE(offset + 22);
    } else if (id === 'data') {
      dataOffset = offset + 8;
      dataLength = size;
      break;
    }
    offset += 8 + size + (size % 2);
  }
  if (dataOffset === 0) throw new Error('No PCM data chunk in WAV.');
  if (bitsPerSample !== 16) throw new Error(`Expected 16-bit PCM, got ${bitsPerSample}-bit.`);
  const samples = dataLength / 2;
  const float = new Float32Array(samples);
  for (let i = 0; i < samples; i += 1) {
    float[i] = buf.readInt16LE(dataOffset + i * 2) / 32768;
  }
  return { float, sampleRate, channels };
}

function stats(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const p = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  const sum = sorted.reduce((a, b) => a + b, 0);
  return {
    min: round(sorted[0]),
    p50: round(p(0.5)),
    p90: round(p(0.9)),
    max: round(sorted[sorted.length - 1]),
    mean: round(sum / sorted.length),
  };
}

function share(numerator, denominator) {
  const n = stats(numerator);
  const d = stats(denominator);
  if (!n || !d || d.mean === 0) return null;
  return round((n.mean / d.mean) * 100);
}

function round(value, digits = 2) {
  return Math.round(value * 10 ** digits) / 10 ** digits;
}

function checkOracle(label, result, oracle) {
  const tokenCount = result.tokens?.length ?? result.debug?.tokenIds?.length ?? 0;
  const text = result.utteranceText ?? '';
  const failures = [];
  if (tokenCount !== oracle.tokenCount) {
    failures.push(`token count ${tokenCount} != ${oracle.tokenCount}`);
  }
  if (oracle.text !== undefined && text !== oracle.text) {
    failures.push(`text mismatch: "${text}"`);
  }
  if (oracle.contains !== undefined && !text.includes(oracle.contains)) {
    failures.push(`text missing "${oracle.contains}": "${text.slice(0, 80)}..."`);
  }
  if (failures.length > 0) {
    throw new Error(`${label} oracle FAIL: ${failures.join('; ')}`);
  }
  return tokenCount;
}

async function loadModel(modelDir, cpuThreads) {
  const runtime = createBuiltInSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createNemotronRnntModelFamily()],
    presets: [createNemotronPresetFactory()],
  });
  const fileUrl = (p) => pathToFileURL(p).toString();
  const model = await runtime.loadModel({
    family: 'nemotron-rnnt',
    modelId: 'nemotron-3.5-asr-streaming-0.6b',
    classification: { family: 'nemotron' },
    options: {
      source: {
        kind: 'direct',
        cpuThreads,
        artifacts: {
          encoderUrl: fileUrl(path.join(modelDir, 'encoder.onnx')),
          decoderUrl: fileUrl(path.join(modelDir, 'decoder.onnx')),
          jointUrl: fileUrl(path.join(modelDir, 'joint.onnx')),
          tokenizerUrl: fileUrl(path.join(modelDir, 'vocab.txt')),
        },
      },
    },
  });
  await model.initialize?.();
  const session = await model.createSession();
  return { runtime, model, session };
}

async function benchFixture(session, { name, fixture, runs }) {
  const { float: pcm, sampleRate } = readWavPcm16(fixture);
  const durationSeconds = pcm.length / sampleRate;
  console.log(`\n== ${name} (${durationSeconds.toFixed(2)} s, ${runs} runs) ==`);

  // Warmup: also serves as the oracle + determinism baseline.
  const warm = await session.transcribe(pcm, {
    sampleRate,
    responseFlavor: 'native',
    returnTokenIds: true,
  });
  const warmTokenCount = checkOracle(name, warm, name.startsWith('jfk') ? JFK_ORACLE : LIBRIVOX_ORACLE);
  const baselineTokenIds = warm.debug?.tokenIds ?? warm.tokens?.map((t) => t.id) ?? [];
  const baselineText = warm.utteranceText;
  console.log(`warmup: tokens=${warmTokenCount} decodeMs=${round(warm.metrics?.decodeMs ?? 0)} totalMs=${round(warm.metrics?.totalMs ?? 0)}`);

  const perPhase = {
    preprocessMs: [],
    encodeMs: [],
    decodeMs: [],
    totalMs: [],
    wallMs: [],
    rtfx: [],
    decodeIterations: [],
  };
  let tokenIdsString = JSON.stringify(baselineTokenIds);
  let deterministic = true;
  let afterRunMem = null;

  for (let run = 0; run < runs; run += 1) {
    const wallStart = performance.now();
    const result = await session.transcribe(pcm, {
      sampleRate,
      responseFlavor: 'native',
      returnTokenIds: true,
    });
    const wallMs = performance.now() - wallStart;
    const m = result.metrics ?? {};
    perPhase.preprocessMs.push(m.preprocessMs ?? 0);
    perPhase.encodeMs.push(m.encodeMs ?? 0);
    perPhase.decodeMs.push(m.decodeMs ?? 0);
    perPhase.totalMs.push(m.totalMs ?? wallMs);
    perPhase.wallMs.push(wallMs);
    if (m.rtfx !== undefined) perPhase.rtfx.push(m.rtfx);
    perPhase.decodeIterations.push(m.decodeIterations ?? 0);
    const ids = result.debug?.tokenIds ?? result.tokens?.map((t) => t.id) ?? [];
    if (JSON.stringify(ids) !== tokenIdsString) deterministic = false;
    afterRunMem = process.memoryUsage();
    console.log(
      `run ${run + 1}/${runs}: pre=${round(m.preprocessMs ?? 0)}ms enc=${round(m.encodeMs ?? 0)}ms ` +
        `dec=${round(m.decodeMs ?? 0)}ms total=${round(m.totalMs ?? wallMs)}ms ` +
        `rtfx=${m.rtfx !== undefined ? round(m.rtfx, 3) : 'n/a'} tokens=${ids.length}`,
    );
  }

  return {
    fixture: path.basename(fixture),
    durationSeconds: round(durationSeconds, 4),
    runs,
    oracle: { tokenCount: warmTokenCount, text: baselineText },
    deterministic,
    phases: {
      preprocessMs: stats(perPhase.preprocessMs),
      encodeMs: stats(perPhase.encodeMs),
      decodeMs: stats(perPhase.decodeMs),
      totalMs: stats(perPhase.totalMs),
      wallMs: stats(perPhase.wallMs),
      rtfx: stats(perPhase.rtfx.length > 0 ? perPhase.rtfx : null),
      decodeIterations: stats(perPhase.decodeIterations),
    },
    encodeShareOfTotal: share(perPhase.encodeMs, perPhase.totalMs),
    decodeShareOfTotal: share(perPhase.decodeMs, perPhase.totalMs),
    memoryAfterLastRun: {
      rssMb: round(afterRunMem.rss / 1024 / 1024),
      heapUsedMb: round(afterRunMem.heapUsed / 1024 / 1024),
      arrayBuffersMb: round(afterRunMem.arrayBuffers / 1024 / 1024),
    },
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!fs.existsSync(args.modelDir)) {
    throw new Error(`Model dir not found: ${args.modelDir}`);
  }
  const config = parseNemotronRnntConfig('nemotron-3.5-asr-streaming-0.6b');
  console.log(`Node ${process.version}, onnxruntime-web@1.29.0 wasm, INT4 singles, cpuThreads=${args.threads}`);
  console.log(`Model dir: ${args.modelDir}`);
  console.log(`Config: chunkFrames=${config.chunkFrames} outputFramesPerChunk=${config.encoderOutputFramesPerChunk} jointWindowFrames=${config.jointWindowFrames}`);

  const loadStart = performance.now();
  const { runtime, model, session } = await loadModel(args.modelDir, args.threads);
  const loadMs = performance.now() - loadStart;
  const afterLoadMem = process.memoryUsage();
  console.log(`Model load: ${round(loadMs)} ms; wasm memory ${round(afterLoadMem.arrayBuffers / 1024 / 1024)} MB`);

  const jfk = await benchFixture(session, {
    name: 'jfk-short.wav',
    fixture: path.join(FIXTURES_DIR, 'jfk-short.wav'),
    runs: args.runsJfk,
  });
  const librivox = await benchFixture(session, {
    name: 'librivox-blankgaps-synthetic.wav',
    fixture: path.join(FIXTURES_DIR, 'librivox-blankgaps-synthetic.wav'),
    runs: args.runsLibrivox,
  });

  await model.dispose();
  await runtime.dispose?.();

  const report = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    purpose: 'Nemotron 3.5 streaming INT4 WASM latency/memory benchmark (public API)',
    runtime: `onnxruntime-web@1.29.0 wasm (node, ${args.threads} threads)`,
    node: process.version,
    platform: `${process.platform}-${process.arch}`,
    model: {
      id: 'nemotron-3.5-asr-streaming-0.6b',
      family: 'nemotron-rnnt',
      source: 'direct INT4 singles (onnx-community INT4 export)',
      dir: args.modelDir,
      cpuThreads: args.threads,
    },
    config: {
      sampleRate: config.sampleRate,
      melBins: config.melBins,
      chunkFrames: config.chunkFrames,
      encoderOutputFramesPerChunk: config.encoderOutputFramesPerChunk,
      jointWindowFrames: config.jointWindowFrames,
      maxDecodeSteps: config.maxDecodeSteps,
    },
    loadMs: round(loadMs),
    memoryAfterLoad: {
      rssMb: round(afterLoadMem.rss / 1024 / 1024),
      heapUsedMb: round(afterLoadMem.heapUsed / 1024 / 1024),
      arrayBuffersMb: round(afterLoadMem.arrayBuffers / 1024 / 1024),
    },
    fixtures: { 'jfk-short.wav': jfk, 'librivox-blankgaps-synthetic.wav': librivox },
  };

  fs.mkdirSync(path.dirname(args.json), { recursive: true });
  fs.writeFileSync(args.json, JSON.stringify(report, null, 2));
  console.log(`\nWrote ${args.json}`);

  const ok = jfk.deterministic && librivox.deterministic;
  if (!ok) {
    console.error('FAIL: cross-run token sequences diverged');
    process.exitCode = 1;
  }
  console.log(ok ? 'Determinism: PASS (identical token sequences across runs)' : '');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
