#!/usr/bin/env node
// Nemotron 3.5 RNNT smoke test against the local INT4 singles (Node, ORT WASM).

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
const FIXTURE = path.resolve(
  REPO_ROOT,
  'tools',
  'data',
  'fixtures',
  'audio',
  'jfk-short.wav',
);
const SINGLE_DIR = 'N:/models/onnx/nemo/nemotron-3.5-asr-streaming-int4-singles';
const OUT = path.resolve(
  REPO_ROOT,
  'tools',
  'data',
  'results',
  'nemotron',
  'nemotron-3.5-node-smoke-2026-08-31.json',
);

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
  if (dataOffset === 0) {
    throw new Error('No PCM data chunk in WAV.');
  }
  if (bitsPerSample !== 16) {
    throw new Error(`Expected 16-bit PCM, got ${bitsPerSample}-bit.`);
  }
  const samples = dataLength / 2;
  const float = new Float32Array(samples);
  for (let i = 0; i < samples; i += 1) {
    float[i] = buf.readInt16LE(dataOffset + i * 2) / 32768;
  }
  return { float, sampleRate, channels };
}

async function main() {
  const { float: pcm, sampleRate, channels } = readWavPcm16(FIXTURE);
  console.log(`Loaded WAV: ${pcm.length} samples @ ${sampleRate} Hz, ${channels} ch`);

  const runtime = createBuiltInSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createNemotronRnntModelFamily()],
    presets: [createNemotronPresetFactory()],
  });

  const fileUrl = (p) => pathToFileURL(p).toString();
  const artifacts = {
    encoderUrl: fileUrl(`${SINGLE_DIR}/encoder.onnx`),
    decoderUrl: fileUrl(`${SINGLE_DIR}/decoder.onnx`),
    jointUrl: fileUrl(`${SINGLE_DIR}/joint.onnx`),
    tokenizerUrl: fileUrl(`${SINGLE_DIR}/vocab.txt`),
  };

  const model = await runtime.loadModel({
    family: 'nemotron-rnnt',
    modelId: 'nemotron-3.5-asr-streaming-0.6b',
    classification: { family: 'nemotron' },
    options: { source: { kind: 'direct', artifacts } },
  });
  await model.initialize?.();

  const session = await model.createSession();

  console.log('Running transcription ...');
  const start = Date.now();
  const result = await session.transcribe(pcm, {
    sampleRate,
    responseFlavor: 'canonical+native',
  });
  const elapsed = Date.now() - start;
  const native = result.native;
  const text = result.canonical?.text ?? result.text ?? native.utteranceText;
  console.log(`Elapsed: ${elapsed} ms`);
  console.log(`Utterance: ${native.utteranceText}`);
  console.log(`Raw text:   ${native.rawUtteranceText ?? ''}`);
  console.log(`Tokens: ${native.tokens?.length ?? 0}`);
  if (native.specialTokens?.length) {
    console.log(
      `Special tokens: ${native.specialTokens
        .map((s) => `${s.kind}:${s.id}:${s.text}`)
        .join(', ')}`,
    );
  }
  console.log(`Lang segment detected: ${native.control?.containsLangSegment}`);

  const transcriptJson = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    purpose: 'Nemotron RNNT node smoke test against INT4 singles',
    modelId: 'nemotron-3.5-asr-streaming-0.6b',
    runtime: 'onnxruntime-web@1.29.0 wasm',
    fixture: 'jfk-short.wav',
    sampleRate,
    durationSeconds: pcm.length / sampleRate,
    tokenCount: native.tokens?.length ?? 0,
    tokenIds: native.debug?.tokenIds ?? native.tokens?.map((t) => t.id),
    utteranceText: native.utteranceText,
    rawUtteranceText: native.rawUtteranceText,
    metrics: native.metrics,
    warnings: native.warnings,
    elapsedMs: elapsed,
  };
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(transcriptJson, null, 2));
  console.log(`Wrote ${OUT}`);

  await model.dispose();
  await runtime.dispose?.();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
