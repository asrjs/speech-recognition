#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';

import { OrtGigaAmRnntExecutor } from '@asrjs/speech-recognition/models/gigaam-rnnt';

const DEFAULT_MODEL_DIR = 'N:\\models\\onnx\\gigaam\\v3-e2e-rnnt';
const DEFAULT_REFERENCE = path.resolve('tools/data/results/gigaam/v3-e2e-rnnt-example-reference.json');

const CONFIG = {
  ecosystem: 'gigaam',
  architecture: 'gigaam-rnnt',
  processorArchitecture: 'gigaam-fbank',
  encoderArchitecture: 'gigaam-conformer',
  decoderArchitecture: 'rnnt',
  sampleRate: 16000,
  rawStride: 4,
  nMels: 64,
  featureHopSeconds: 0.01,
  vocabularySize: 1025,
  languages: ['ru'],
  tokenizer: { kind: 'sentencepiece', blankTokenId: 1024 },
  nFft: 320,
  winLength: 320,
  hopLength: 160,
  featureLayout: 'mel-major',
  predictionHiddenSize: 320,
  predictionRnnLayers: 1,
  maxTokensPerFrame: 10,
};

function parseArgs() {
  const options = {
    modelDir: DEFAULT_MODEL_DIR,
    reference: DEFAULT_REFERENCE,
    backend: 'wasm',
    encoderBackend: undefined,
    decoderBackend: undefined,
    jointBackend: undefined,
    runs: 3,
    warmup: 1,
    output: null,
  };
  const args = process.argv.slice(2);
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === '--model-dir') options.modelDir = path.resolve(args[++index]);
    else if (arg === '--reference') options.reference = path.resolve(args[++index]);
    else if (arg === '--backend') options.backend = args[++index];
    else if (arg === '--encoder-backend') options.encoderBackend = args[++index];
    else if (arg === '--decoder-backend') options.decoderBackend = args[++index];
    else if (arg === '--joint-backend') options.jointBackend = args[++index];
    else if (arg === '--runs') options.runs = Math.max(1, Number(args[++index]));
    else if (arg === '--warmup') options.warmup = Math.max(0, Number(args[++index]));
    else if (arg === '--output') options.output = path.resolve(args[++index]);
  }
  for (const key of ['backend', 'encoderBackend', 'decoderBackend', 'jointBackend']) {
    if (options[key] !== undefined && options[key] !== 'wasm' && options[key] !== 'webgpu') {
      throw new Error(`Unsupported ${key}: ${options[key]}. Use wasm or webgpu.`);
    }
  }
  return options;
}

function ensureFile(filePath, label) {
  if (!fs.existsSync(filePath)) throw new Error(`${label} not found: ${filePath}`);
  return filePath;
}

function loadNpyFloat32(filePath) {
  const buffer = fs.readFileSync(filePath);
  if (buffer[0] !== 0x93 || buffer.toString('ascii', 1, 6) !== 'NUMPY') {
    throw new Error(`Unsupported NPY file: ${filePath}`);
  }
  const major = buffer[6];
  const headerLength = major === 1 ? buffer.readUInt16LE(8) : Number(buffer.readBigUInt64LE(8));
  const headerStart = major === 1 ? 10 : 16;
  const dataOffset = headerStart + headerLength;
  if ((buffer.byteLength - dataOffset) % 4 !== 0) throw new Error(`NPY payload is not float32-aligned: ${filePath}`);
  return new Float32Array(buffer.buffer, buffer.byteOffset + dataOffset, (buffer.byteLength - dataOffset) / 4);
}

function artifacts(modelDir) {
  return {
    encoderUrl: pathToFileURL(ensureFile(path.join(modelDir, 'v3_e2e_rnnt_encoder.onnx'), 'Encoder model')).href,
    decoderUrl: pathToFileURL(ensureFile(path.join(modelDir, 'v3_e2e_rnnt_decoder.onnx'), 'Decoder model')).href,
    jointUrl: pathToFileURL(ensureFile(path.join(modelDir, 'v3_e2e_rnnt_joint.onnx'), 'Joint model')).href,
    tokenizerUrl: pathToFileURL(ensureFile(path.join(modelDir, 'v3_e2e_rnnt_vocab.txt'), 'Tokenizer')).href,
  };
}

function percentile(values, fraction) {
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1)];
}

function memorySnapshot() {
  const memory = process.memoryUsage();
  return {
    rssBytes: memory.rss,
    heapUsedBytes: memory.heapUsed,
    externalBytes: memory.external,
    arrayBuffersBytes: memory.arrayBuffers,
  };
}

async function main() {
  const options = parseArgs();
  const reference = JSON.parse(fs.readFileSync(ensureFile(options.reference, 'Reference capture'), 'utf8'));
  const sample = reference.samples?.[0];
  if (!sample?.audio?.waveform_npy || !sample.text) throw new Error('Reference capture has no first waveform/text sample.');
  const waveformPath = ensureFile(sample.audio.waveform_npy, 'Waveform NPY');
  const waveform = loadNpyFloat32(waveformPath);
  const modelDir = path.resolve(options.modelDir);
  const expected = sample.text;
  const audio = {
    sampleRate: CONFIG.sampleRate,
    numberOfChannels: 1,
    numberOfFrames: waveform.length,
    durationSeconds: waveform.length / CONFIG.sampleRate,
    channels: [waveform],
  };
  const executor = new OrtGigaAmRnntExecutor('gigaam-v3-e2e-rnnt', options.backend, CONFIG, {
    source: {
      kind: 'direct',
      cpuThreads: 1,
      artifacts: artifacts(modelDir),
      encoderBackend: options.encoderBackend,
      decoderBackend: options.decoderBackend,
      jointBackend: options.jointBackend,
    },
  });
  const loadStarted = performance.now();
  await executor.ready();
  const loadMs = performance.now() - loadStarted;
  try {
    for (let index = 0; index < options.warmup; index += 1) await executor.transcribe(audio);
    const runs = [];
    for (let index = 0; index < options.runs; index += 1) {
      const started = performance.now();
      const result = await executor.transcribe(audio);
      const elapsedMs = performance.now() - started;
      runs.push({
        runIndex: index + 1,
        elapsedMs: Number(elapsedMs.toFixed(3)),
        rtfx: result.metrics?.rtfx ?? null,
        preprocessMs: result.metrics?.preprocessMs ?? null,
        encodeMs: result.metrics?.encodeMs ?? null,
        decodeMs: result.metrics?.decodeMs ?? null,
        encoderBackend: result.metrics?.encoderBackend ?? null,
        decoderBackend: result.metrics?.decoderBackend ?? null,
        jointBackend: result.metrics?.jointBackend ?? null,
        tokens: result.tokens?.length ?? 0,
        textMatch: result.utteranceText === expected,
        text: result.utteranceText,
        memory: memorySnapshot(),
      });
    }
    const elapsed = runs.map((run) => run.elapsedMs);
    const payload = {
      schemaVersion: 1,
      capturedAt: new Date().toISOString(),
      model: {
        modelId: 'gigaam-v3-e2e-rnnt',
        modelDir,
        backend: options.backend,
        engine: 'onnxruntime-web',
        components: {
          encoder: options.encoderBackend ?? options.backend,
          decoder: options.decoderBackend ?? options.backend,
          joint: options.jointBackend ?? options.backend,
        },
      },
      artifact: { encoder: 'v3_e2e_rnnt_encoder.onnx', decoder: 'v3_e2e_rnnt_decoder.onnx', joint: 'v3_e2e_rnnt_joint.onnx' },
      audio: { sampleRate: audio.sampleRate, frames: waveform.length, durationSeconds: audio.durationSeconds, reference: options.reference },
      benchmark: {
        warmupRuns: options.warmup,
        measuredRuns: options.runs,
        loadMs: Number(loadMs.toFixed(3)),
        minMs: Math.min(...elapsed),
        p50Ms: percentile(elapsed, 0.5),
        p90Ms: percentile(elapsed, 0.9),
        maxMs: Math.max(...elapsed),
        p50Rtfx: percentile(runs.map((run) => run.rtfx).filter((value) => Number.isFinite(value)), 0.5),
        allTextMatch: runs.every((run) => run.textMatch),
      },
      runs,
    };
    const encoded = JSON.stringify(payload, null, 2);
    if (options.output) {
      fs.mkdirSync(path.dirname(options.output), { recursive: true });
      fs.writeFileSync(options.output, `${encoded}\n`, 'utf8');
    } else {
      console.log(encoded);
    }
  } finally {
    await executor.dispose();
  }
}

main().catch((error) => {
  console.error('[node-gigaam-rnnt-benchmark] failed:', error);
  process.exitCode = 1;
});
