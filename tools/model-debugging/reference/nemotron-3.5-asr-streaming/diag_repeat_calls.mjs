#!/usr/bin/env node
// Diagnostic: repeat transcribe calls on librivox in ONE session and dump
// per-call tokenIds + metrics to localize the 88-vs-91 first-call delta.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createWasmBackend } from '../../../../dist/index.js';
import { createNemotronRnntModelFamily } from '../../../../dist/models/nemotron-rnnt/index.js';
import { createNemotronPresetFactory } from '../../../../dist/presets/nemotron/index.js';
import { createBuiltInSpeechRuntime } from '../../../../dist/runtime/builtins.js';

const __filename = fileURLToPath(import.meta.url);
const REPO_ROOT = path.resolve(path.dirname(__filename), '..', '..', '..', '..');
const FIXTURE = path.resolve(REPO_ROOT, 'tools', 'data', 'fixtures', 'audio', 'librivox-blankgaps-synthetic.wav');
const SINGLE_DIR = process.env.NEMOTRON_INT4_DIR ?? 'N:/models/onnx/nemo/nemotron-3.5-asr-streaming-int4-singles';
const THREADS = Number(process.env.NEMOTRON_THREADS ?? 6);
const CALLS = Number(process.env.NEMOTRON_CALLS ?? 3);

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
  const float = new Float32Array(dataLength / 2);
  for (let i = 0; i < float.length; i += 1) float[i] = buf.readInt16LE(dataOffset + i * 2) / 32768;
  return { float, sampleRate, channels, bits };
}

async function main() {
  const { float: pcm, sampleRate } = readWavPcm16(FIXTURE);
  console.log(`audio=${pcm.length / sampleRate}s threads=${THREADS} calls=${CALLS}`);
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
          encoderUrl: fileUrl(`${SINGLE_DIR}/encoder.onnx`),
          decoderUrl: fileUrl(`${SINGLE_DIR}/decoder.onnx`),
          jointUrl: fileUrl(`${SINGLE_DIR}/joint.onnx`),
          tokenizerUrl: fileUrl(`${SINGLE_DIR}/vocab.txt`),
        },
      },
    },
  });
  await model.initialize?.();
  const session = await model.createSession();

  const calls = [];
  for (let i = 0; i < CALLS; i += 1) {
    const r = await session.transcribe(pcm, { sampleRate, responseFlavor: 'native', returnTokenIds: true });
    const ids = r.debug?.tokenIds ?? r.tokens?.map((t) => t.id) ?? [];
    calls.push({ ids, metrics: r.metrics, text: r.utteranceText });
    console.log(`call ${i}: tokens=${ids.length} encFrames=${r.metrics?.encoderFrameCount} iterations=${r.metrics?.decodeIterations} enc=${r.metrics?.encodeMs}ms dec=${r.metrics?.decodeMs}ms`);
    console.log(`  ids=${JSON.stringify(ids)}`);
  }

  // First divergence between call 0 and each later call.
  for (let i = 1; i < calls.length; i += 1) {
    const a = calls[0].ids; const b = calls[i].ids;
    let d = -1;
    for (let k = 0; k < Math.max(a.length, b.length); k += 1) {
      if (a[k] !== b[k]) { d = k; break; }
    }
    console.log(`divergence call0 vs call${i}: index=${d} ${d >= 0 ? `${a[d]} != ${b[d]}` : 'IDENTICAL'}`);
  }
  const same = calls.every((c) => JSON.stringify(c.ids) === JSON.stringify(calls[0].ids));
  console.log(same ? 'RESULT: all calls identical' : 'RESULT: calls DIFFER');

  await model.dispose();
  await runtime.dispose?.();
}
main().catch((e) => { console.error(e); process.exitCode = 1; });
