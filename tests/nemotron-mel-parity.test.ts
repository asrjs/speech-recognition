import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { JSMelProcessor } from '../src/audio/js-mel.js';

/**
 * Step 3b: verify the JS 128-bin mel frontend produces the correct shape
 * and reasonable values on the committed jfk-short.wav fixture.
 *
 * Full feature-level parity against the official NeMo preprocessor is a
 * follow-up; this test confirms the JS mel output contract.
 */

const FIXTURE_DIR = join(__dirname, '..', 'tools', 'data', 'fixtures', 'audio');

function readWavSamples(path: string): Float32Array {
  const buf = readFileSync(path);
  const dataOffset = 44;
  const samples = new Float32Array((buf.length - dataOffset) / 2);
  for (let i = 0; i < samples.length; i++) {
    samples[i] = buf.readInt16LE(dataOffset + i * 2) / 32768;
  }
  return samples;
}

describe('Nemotron 3.5 JS mel frontend (step 3b)', () => {
  it('produces 128-bin mel features with correct frame count on jfk-short.wav', () => {
    const samples = readWavSamples(join(FIXTURE_DIR, 'jfk-short.wav'));
    const processor = new JSMelProcessor({ nMels: 128, validLengthMode: 'onnx' });
    const result = processor.process(samples);

    expect(result.features.length).toBeGreaterThan(0);
    expect(result.length).toBeGreaterThan(0);

    // 128 bins per frame; features array has frameCount frames (includes trailing zeros)
    expect(result.features.length).toBe(result.frameCount * 128);

    // jfk-short.wav is ~11 s at 16 kHz; at hop_length=160, expect ~1100 frames
    // NeMo config says 25 frames per first chunk; verified encoder expects [1,25,128]
    expect(result.length).toBeGreaterThanOrEqual(1050);
    expect(result.length).toBeLessThanOrEqual(1150);

    // jfk-short.wav is ~11 s at 16 kHz = ~176000 samples
    // at hop_length=160, expect ~1100 frames (verified: 1101 raw, 1100 valid)
    expect(result.frameCount).toBe(1101);
    expect(result.length).toBe(1100);
  });

  it('produces non-trivial features (not all zeros)', () => {
    const samples = readWavSamples(join(FIXTURE_DIR, 'jfk-short.wav'));
    const processor = new JSMelProcessor({ nMels: 128, validLengthMode: 'onnx' });
    const result = processor.process(samples);

    let nonZero = 0;
    for (let i = 0; i < result.features.length; i++) {
      if (result.features[i] !== 0) nonZero++;
    }
    expect(nonZero).toBeGreaterThan(result.features.length * 0.5);
  });

  it('produces log-mel values in expected range', () => {
    const samples = readWavSamples(join(FIXTURE_DIR, 'jfk-short.wav'));
    const processor = new JSMelProcessor({ nMels: 128, validLengthMode: 'onnx' });
    const result = processor.process(samples);

    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < result.features.length; i++) {
      const v = result.features[i]!;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    // log-mel features without per-feature normalization
    // can reach ~16 on high-energy bins; valid range is wider
    expect(min).toBeGreaterThan(-30);
    expect(max).toBeLessThan(20);
  });
});
