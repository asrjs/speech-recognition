import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { describe, test, expect } from 'vitest';
import ort from 'onnxruntime-node';

import { MedAsrEncoder } from '../src/pipeline/encoder.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const refDir = path.resolve(projectRoot, 'tests/reference_medasr');
const onnxPath = path.resolve(projectRoot, 'models/medasr/model.onnx');

function hasPrereqs() {
  return (
    fs.existsSync(path.join(refDir, 'features.json')) &&
    fs.existsSync(path.join(refDir, 'attention_mask.json')) &&
    fs.existsSync(path.join(refDir, 'logits.json')) &&
    fs.existsSync(onnxPath)
  );
}

function loadRef() {
  return {
    features: JSON.parse(fs.readFileSync(path.join(refDir, 'features.json'), 'utf-8')),
    attentionMask: JSON.parse(fs.readFileSync(path.join(refDir, 'attention_mask.json'), 'utf-8')),
    logits: JSON.parse(fs.readFileSync(path.join(refDir, 'logits.json'), 'utf-8')),
  };
}

function flattenTxM(features) {
  const T = features.length;
  const M = features[0].length;
  const out = new Float32Array(T * M);
  for (let t = 0; t < T; t++) {
    for (let m = 0; m < M; m++) out[t * M + m] = features[t][m];
  }
  return out;
}

function flatten2d(rows) {
  const T = rows.length;
  const V = rows[0].length;
  const out = new Float32Array(T * V);
  for (let t = 0; t < T; t++) {
    for (let v = 0; v < V; v++) out[t * V + v] = rows[t][v];
  }
  return out;
}

function diffStats(a, b) {
  const n = Math.min(a.length, b.length);
  let max = 0;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > max) max = d;
    sum += d;
  }
  return { max, mean: n ? sum / n : 0, compared: n };
}

const parityTest = hasPrereqs() ? test : test.skip;

describe('medasrjs encoder parity', () => {
  parityTest('matches reference ONNX logits from reference features', async () => {
    const ref = loadRef();
    const session = await ort.InferenceSession.create(onnxPath, { executionProviders: ['cpu'] });
    const encoder = new MedAsrEncoder({ session, ort, nMels: 128 });

    const feats = flattenTxM(ref.features);
    const out = await encoder.infer(feats, ref.features.length, ref.attentionMask);
    const refLogits = flatten2d(ref.logits);

    const stats = diffStats(new Float32Array(out.logits), refLogits);
    expect(out.frames).toBe(ref.logits.length);
    expect(out.vocabSize).toBe(512);
    expect(stats.max).toBeLessThan(5e-4);
    expect(stats.mean).toBeLessThan(5e-6);
  });
});
