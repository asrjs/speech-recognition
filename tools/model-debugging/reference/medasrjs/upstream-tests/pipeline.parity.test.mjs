import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { describe, test, expect } from 'vitest';
import ort from 'onnxruntime-node';
import wavefilePkg from 'wavefile';

import { MedAsrPipeline } from '../src/index.js';

const { WaveFile } = wavefilePkg;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const refDir = path.resolve(projectRoot, 'tests/reference_medasr');
const onnxPath = path.resolve(projectRoot, 'models/medasr/model.onnx');
const tokensPath = path.resolve(projectRoot, 'models/medasr/tokens.txt');

function hasPrereqs() {
  const metadataPath = path.join(refDir, 'metadata.json');
  if (!fs.existsSync(metadataPath)) return false;
  const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
  return (
    !!metadata?.audio_path &&
    fs.existsSync(metadata.audio_path) &&
    fs.existsSync(onnxPath) &&
    fs.existsSync(tokensPath)
  );
}

function loadRef() {
  return JSON.parse(fs.readFileSync(path.join(refDir, 'metadata.json'), 'utf-8'));
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

function normalize(s) {
  return (s || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
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
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[m][n];
}

const parityTest = hasPrereqs() ? test : test.skip;

describe('medasrjs full pipeline parity', () => {
  parityTest('audio -> text matches reference decode closely', async () => {
    const ref = loadRef();
    const audio = loadAudioMono16k(ref.audio_path);
    const session = await ort.InferenceSession.create(onnxPath, { executionProviders: ['cpu'] });
    const pipeline = MedAsrPipeline.fromNodeSession({ session, ort, tokensPath, nMels: 128 });

    const out = await pipeline.transcribe(audio);
    const got = normalize(out.utterance_text);
    const exp = normalize(ref.text_onnx_collapsed || ref.text_pt_collapsed || '');

    const dist = levenshtein(got, exp);
    const cer = exp.length ? dist / exp.length : 0;

    expect(cer).toBeLessThanOrEqual(0.05);
    expect(out.vocabSize).toBe(512);
  });
});
