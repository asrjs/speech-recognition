#!/usr/bin/env node
// Timing probe: Nemotron executor on the long librivox fixture.
// Measures encode/decode wall time with the current full-window joint loop.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createWasmBackend } from '../../../../dist/index.js';
import { createNemotronRnntModelFamily } from '../../../../dist/models/nemotron-rnnt/index.js';
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
  'librivox-blankgaps-synthetic.wav',
);
const SINGLE_DIR = 'N:/models/onnx/nemo/nemotron-3.5-asr-streaming-int4-singles';

function readWavPcm16(filePath) {
  const buf = fs.readFileSync(filePath);
  let offset = 12;
  let sampleRate = 0;
  let dataOffset = 0;
  let dataLength = 0;
  while (offset < buf.length - 8) {
    const id = buf.toString('ascii', offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    if (id === 'fmt ') {
      sampleRate = buf.readUInt32LE(offset + 12);
    } else if (id === 'data') {
      dataOffset = offset + 8;
      dataLength = size;
      break;
    }
    offset += 8 + size + (size % 2);
  }
  const samples = dataLength / 2;
  const float = new Float32Array(samples);
  for (let i = 0; i < samples; i += 1) {
    float[i] = buf.readInt16LE(dataOffset + i * 2) / 32768;
  }
  return { float, sampleRate };
}

async function main() {
  const { float: pcm, sampleRate } = readWavPcm16(FIXTURE);
  const seconds = pcm.length / sampleRate;
  console.log(`Fixture: ${seconds.toFixed(1)} s (${pcm.length} samples)`);

  const runtime = createBuiltInSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createNemotronRnntModelFamily()],
    presets: [],
  });
  const fileUrl = (p) => pathToFileURL(p).toString();
  const model = await runtime.loadModel({
    family: 'nemotron-rnnt',
    modelId: 'nemotron-3.5-asr-streaming-0.6b',
    classification: { family: 'nemotron' },
    options: {
      source: {
        kind: 'direct',
        artifacts: {
          encoderUrl: fileUrl(`${SINGLE_DIR}/encoder.onnx`),
          decoderUrl: fileUrl(`${SINGLE_DIR}/decoder.onnx`),
          jointUrl: fileUrl(`${SINGLE_DIR}/joint.onnx`),
          tokenizerUrl: fileUrl(`${SINGLE_DIR}/vocab.txt`),
        },
      },
    },
  });
  const session = await model.createSession();

  const start = Date.now();
  const result = await session.transcribe(pcm, {
    sampleRate,
    responseFlavor: 'native',
  });
  const elapsed = Date.now() - start;
  const m = result.metrics ?? {};
  console.log(`totalMs=${elapsed} encodeMs=${m.encodeMs} decodeMs=${m.decodeMs}`);
  console.log(`tokens=${result.tokens?.length ?? 0}`);
  console.log(`text=${result.utteranceText}`);

  await model.dispose();
  await runtime.dispose?.();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
