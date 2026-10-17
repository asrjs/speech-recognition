#!/usr/bin/env node
// Nemotron 3.5 RNNT smoke test against our own NeMo-exported graphs
// (fp32/fp16, executor-aligned contract) via the public API on WASM.
//
// Env: NEMOTRON_EXPORT_DIR (default fp16 dir), NEMOTRON_THREADS (default 6).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createWasmBackend } from '../../../../dist/index.js';
import { createNemotronRnntModelFamily } from '../../../../dist/models/nemotron-rnnt/index.js';
import { createNemotronPresetFactory } from '../../../../dist/presets/nemotron/index.js';
import { createBuiltInSpeechRuntime } from '../../../../dist/runtime/builtins.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..', '..', '..');
const FIXTURE = path.resolve(REPO_ROOT, 'tools', 'data', 'fixtures', 'audio', 'jfk-short.wav');
const EXPORT_DIR = process.env.NEMOTRON_EXPORT_DIR
  ?? 'N:/models/onnx/nemo/nemotron-3.5-asr-streaming-fp16-nemo';
const THREADS = Number(process.env.NEMOTRON_THREADS ?? 6);
const WINDOW_MS = process.env.NEMOTRON_STREAMING_WINDOW_MS
  ? Number(process.env.NEMOTRON_STREAMING_WINDOW_MS)
  : undefined;
const WINDOW_SUFFIX = WINDOW_MS === undefined ? '' : `-window-${WINDOW_MS}ms`;
const OUT = path.resolve(
  REPO_ROOT,
  'tools',
  'data',
  'results',
  'nemotron',
  `nemotron-3.5-export-smoke-${path.basename(EXPORT_DIR)}${WINDOW_SUFFIX}-2026-08-31.json`,
);

// The established body oracle; the export may append a boundary tail
// (". <en-US>") matching the official NeMo reference.
const ORACLE_TEXT =
  'And so my fellow Americans ask not what your country can do for you at what you can do for your country';

function readWavPcm16(filePath) {
  const buf = fs.readFileSync(filePath);
  let offset = 12, sampleRate = 0, channels = 0, bits = 0, dataOffset = 0, dataLength = 0;
  while (offset < buf.length - 8) {
    const id = buf.toString('ascii', offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    if (id === 'fmt ') {
      channels = buf.readUInt16LE(offset + 10);
      sampleRate = buf.readUInt32LE(offset + 12);
      bits = buf.readUInt16LE(offset + 22);
    } else if (id === 'data') { dataOffset = offset + 8; dataLength = size; break; }
    offset += 8 + size + (size % 2);
  }
  if (bits !== 16) throw new Error(`Expected 16-bit PCM, got ${bits}-bit.`);
  const float = new Float32Array(dataLength / 2);
  for (let i = 0; i < float.length; i += 1) float[i] = buf.readInt16LE(dataOffset + i * 2) / 32768;
  return { float, sampleRate, channels };
}

async function main() {
  const { float: pcm, sampleRate } = readWavPcm16(FIXTURE);
  console.log(`audio=${(pcm.length / sampleRate).toFixed(2)}s dir=${EXPORT_DIR} threads=${THREADS}`);

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
        cpuThreads: THREADS,
        artifacts: {
          encoderUrl: fileUrl(path.join(EXPORT_DIR, 'encoder.onnx')),
          decoderUrl: fileUrl(path.join(EXPORT_DIR, 'decoder.onnx')),
          jointUrl: fileUrl(path.join(EXPORT_DIR, 'joint.onnx')),
          tokenizerUrl: fileUrl(path.join(EXPORT_DIR, 'vocab.txt')),
          encoderDataUrl: fileUrl(path.join(EXPORT_DIR, 'encoder.onnx.data')),
          decoderDataUrl: fileUrl(path.join(EXPORT_DIR, 'decoder.onnx.data')),
          jointDataUrl: fileUrl(path.join(EXPORT_DIR, 'joint.onnx.data')),
          encoderFilename: 'encoder.onnx',
          decoderFilename: 'decoder.onnx',
          jointFilename: 'joint.onnx',
        },
      },
    },
  });
  await model.initialize?.();
  const session = await model.createSession();

  const start = Date.now();
  const result = await session.transcribe(pcm, {
    sampleRate,
    responseFlavor: 'native',
    returnTokenIds: true,
    promptId: 0,
    streamingWindowMs: WINDOW_MS,
  });
  const elapsed = Date.now() - start;
  const native = result;
  const text = native.utteranceText ?? '';
  const ids = native.debug?.tokenIds ?? native.tokens?.map((t) => t.id) ?? [];
  console.log(`Elapsed: ${elapsed} ms`);
  console.log(`Tokens: ${ids.length}`);
  console.log(`Utterance: ${text}`);

  const body = text.replace(/(\s*<[^>]+>)+\s*$/, '').replace(/\.\s*$/, '').trim();
  const pass = body === ORACLE_TEXT;
  console.log(pass ? 'ORACLE PASS' : `ORACLE FAIL: "${body}"`);

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify({
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    purpose: 'public-API WASM smoke against our NeMo-exported graphs',
    exportDir: EXPORT_DIR,
    cpuThreads: THREADS,
    streamingWindowMs: WINDOW_MS ?? null,
    fixture: 'jfk-short.wav',
    tokenCount: ids.length,
    tokenIds: ids,
    utteranceText: text,
    oracle: { text: ORACLE_TEXT, pass },
    metrics: native.metrics,
    elapsedMs: elapsed,
  }, null, 2));
  console.log(`Wrote ${OUT}`);

  await model.dispose();
  await runtime.dispose?.();
  if (!pass) process.exitCode = 1;
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
