import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, test, expect } from 'vitest';
import ort from 'onnxruntime-node';
import wavefilePkg from 'wavefile';

import { MedAsrModel } from '../src/medasr.js';
import { hasArtifact, readJsonArtifact } from './reference-io.mjs';

const { WaveFile } = wavefilePkg;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

const REF_DIR = path.join(__dirname, 'reference_medasr');
const ONNX_PATH = path.join(projectRoot, 'models', 'medasr', 'model.onnx');
const TOKENS_PATH = path.join(projectRoot, 'models', 'medasr', 'tokens.txt');

function loadReference() {
  const metadata = readJsonArtifact(REF_DIR, 'metadata.json');
  metadata.features = readJsonArtifact(REF_DIR, 'features.json');
  metadata.attention_mask = readJsonArtifact(REF_DIR, 'attention_mask.json');
  metadata.logits = readJsonArtifact(REF_DIR, 'logits.json');
  return metadata;
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

function flattenLogits(logits) {
  const T = logits.length;
  const V = logits[0].length;
  const out = new Float32Array(T * V);
  for (let t = 0; t < T; t++) {
    for (let v = 0; v < V; v++) out[t * V + v] = logits[t][v];
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

function levenshtein(a, b) {
  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Uint32Array(n + 1));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }
  return dp[m][n];
}

function normalizeText(s) {
  return (s || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function loadAudioMono16k(audioPath) {
  const wavBuffer = fs.readFileSync(audioPath);
  const wav = new WaveFile(wavBuffer);
  if (wav.fmt.sampleRate !== 16000) wav.toSampleRate(16000);
  wav.toBitDepth('32f');
  let samples = wav.getSamples(false, Float32Array);

  if (Array.isArray(samples)) {
    if (samples.length === 1) return samples[0];
    const mono = new Float32Array(samples[0].length);
    for (let i = 0; i < mono.length; i++) {
      let acc = 0;
      for (let c = 0; c < samples.length; c++) acc += samples[c][i];
      mono[i] = acc / samples.length;
    }
    return mono;
  }

  return samples;
}

function hasPrereqs() {
  const metadataPath = path.join(REF_DIR, 'metadata.json');
  if (!hasArtifact(REF_DIR, 'metadata.json')) return false;
  const metadata = readJsonArtifact(REF_DIR, 'metadata.json');
  return (
    !!metadata?.audio_path &&
    fs.existsSync(metadata.audio_path) &&
    hasArtifact(REF_DIR, 'features.json') &&
    hasArtifact(REF_DIR, 'attention_mask.json') &&
    hasArtifact(REF_DIR, 'logits.json') &&
    fs.existsSync(ONNX_PATH) &&
    fs.existsSync(TOKENS_PATH)
  );
}

const parityTest = hasPrereqs() ? test : test.skip;

describe('medasrjs Node wrapper parity', () => {
  parityTest('inferFromFeatures matches reference logits closely', async () => {
    const ref = loadReference();
    const session = await ort.InferenceSession.create(ONNX_PATH, { executionProviders: ['cpu'] });
    const model = await MedAsrModel.fromSession({
      session,
      ort,
      tokenizerPath: TOKENS_PATH,
      nMels: 128,
    });

    const feats = flattenTxM(ref.features);
    const out = await model.inferFromFeatures(feats, ref.features.length, ref.attention_mask);
    const refLogits = flattenLogits(ref.logits);
    const outLogits = new Float32Array(out.logits);

    const stats = diffStats(outLogits, refLogits);
    expect(stats.max).toBeLessThan(5e-4);
    expect(stats.mean).toBeLessThan(5e-6);
    expect(out.collapsedIds).toEqual(ref.onnx_collapsed_ids);
    expect(out.text).toBe(ref.text_onnx_collapsed);
    expect(out.seconds_per_frame).toBeGreaterThan(0);
    expect(out.utterance?.has_speech).toBe(true);
    expect(out.start_time).toBeGreaterThanOrEqual(0);
    expect(out.end_time).toBeGreaterThan(out.start_time);
    expect(out.confidence).toBeGreaterThan(0);
    expect(out.confidence).toBeLessThanOrEqual(1);
    expect(Array.isArray(out.token_spans)).toBe(true);
    expect(out.token_spans.length).toBe(out.collapsedIds.length);
    expect(Array.isArray(out.sentences)).toBe(true);
    expect(out.sentences.length).toBeGreaterThan(0);
  });

  parityTest('transcribe(audio) stays near reference decode', async () => {
    const ref = loadReference();
    const session = await ort.InferenceSession.create(ONNX_PATH, { executionProviders: ['cpu'] });
    const model = await MedAsrModel.fromSession({
      session,
      ort,
      tokenizerPath: TOKENS_PATH,
      nMels: 128,
    });

    const audio = loadAudioMono16k(ref.audio_path);
    const out = await model.transcribe(audio, 16000);
    const got = normalizeText(out.utterance_text);
    const exp = normalizeText(ref.text_onnx_collapsed || ref.text_pt_collapsed || '');

    const dist = levenshtein(got, exp);
    const cer = exp.length ? dist / exp.length : 0;
    expect(cer).toBeLessThanOrEqual(0.05);
    expect(out.utterance?.has_speech).toBe(true);
    expect(out.start_time).toBeGreaterThanOrEqual(0);
    expect(out.end_time).toBeGreaterThan(out.start_time);
    expect(out.confidence).toBeGreaterThan(0);
    expect(out.confidence).toBeLessThanOrEqual(1);
    expect(Array.isArray(out.sentences)).toBe(true);
    expect(out.sentences.length).toBeGreaterThan(0);
  });
});
