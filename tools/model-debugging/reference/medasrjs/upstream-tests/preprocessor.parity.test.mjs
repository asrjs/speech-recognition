import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { describe, test, expect } from 'vitest';
import wavefilePkg from 'wavefile';

import { MedAsrPreprocessor } from '../src/pipeline/preprocessor.js';
import { hasArtifact, readJsonArtifact } from './reference-io.mjs';

const { WaveFile } = wavefilePkg;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const REF_DIR = path.resolve(__dirname, './reference_medasr');

function hasReference() {
  if (!hasArtifact(REF_DIR, 'metadata.json') || !hasArtifact(REF_DIR, 'features.json')) {
    return false;
  }
  const metadata = readJsonArtifact(REF_DIR, 'metadata.json');
  return !!metadata?.audio_path && fs.existsSync(metadata.audio_path);
}

function loadReference() {
  const metadata = readJsonArtifact(REF_DIR, 'metadata.json');
  const features = readJsonArtifact(REF_DIR, 'features.json');
  return { metadata, features };
}

function loadAudioMono16k(audioPath) {
  const wavBuffer = fs.readFileSync(audioPath);
  const wav = new WaveFile(wavBuffer);
  if (wav.fmt.sampleRate !== 16000) wav.toSampleRate(16000);
  wav.toBitDepth('32f');

  const samples = wav.getSamples(false, Float32Array);
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

function flatten2d(featuresTxM) {
  const T = featuresTxM.length;
  const M = featuresTxM[0].length;
  const out = new Float32Array(T * M);
  for (let t = 0; t < T; t++) {
    for (let m = 0; m < M; m++) out[t * M + m] = featuresTxM[t][m];
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

const parityTest = hasReference() ? test : test.skip;

describe('medasrjs preprocessor parity', () => {
  parityTest('matches reference MedASR features within tolerance', () => {
    const { metadata, features } = loadReference();
    const audio = loadAudioMono16k(metadata.audio_path);

    const preprocessor = new MedAsrPreprocessor({ nMels: 128 });
    const out = preprocessor.process(audio);

    const refFlat = flatten2d(features);
    const stats = diffStats(out.featuresTxM, refFlat);

    expect(out.frames).toBe(features.length);
    expect(stats.max).toBeLessThan(1e-2);
    expect(stats.mean).toBeLessThan(2e-3);
  });
});
