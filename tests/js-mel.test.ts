import { describe, expect, it } from 'vitest';

import { IncrementalJSMelProcessor, JSMelProcessor, MEL_CONSTANTS } from '../src/audio/js-mel.js';

function createSineWave(samples: number, frequency = 440): Float32Array {
  const pcm = new Float32Array(samples);
  for (let index = 0; index < samples; index += 1) {
    pcm[index] = Math.sin((2 * Math.PI * frequency * index) / MEL_CONSTANTS.SAMPLE_RATE);
  }
  return pcm;
}

describe('js mel processor', () => {
  it('returns empty output for empty audio', () => {
    const processor = new JSMelProcessor();

    const result = processor.process(new Float32Array(0));

    expect(result.features).toEqual(new Float32Array(0));
    expect(result.length).toBe(0);
  });

  it('produces finite normalized mel features with deterministic frame counts', () => {
    const processor = new JSMelProcessor({ nMels: 128 });
    const audio = createSineWave(1600);

    const result = processor.process(audio);

    expect(result.frameCount).toBe(11);
    expect(result.length).toBe(10);
    expect(result.features.length).toBe(128 * 11);
    for (let melIndex = 0; melIndex < 128; melIndex += 1) {
      expect(result.features[melIndex * result.frameCount + result.length]).toBe(0);
    }
    expect(Array.from(result.features).every(Number.isFinite)).toBe(true);
  });

  it('supports centered-frame mode for models that treat the trailing padded frame as valid', () => {
    const processor = new JSMelProcessor({ nMels: 128, validLengthMode: 'centered' });
    const audio = createSineWave(1600);

    const result = processor.process(audio);

    expect(result.frameCount).toBe(11);
    expect(result.length).toBe(11);
    expect(result.features.length).toBe(128 * 11);
    expect(Array.from(result.features).every(Number.isFinite)).toBe(true);
  });

  it('returns stable feature buffers across repeated process calls', () => {
    const processor = new JSMelProcessor({ nMels: 128 });
    const firstAudio = createSineWave(1600, 440);
    const secondAudio = createSineWave(3200, 660);

    const first = processor.process(firstAudio);
    const firstSnapshot = new Float32Array(first.features);
    const second = processor.process(secondAudio);

    expect(first.features).not.toBe(second.features);
    expect(first.features).toEqual(firstSnapshot);
    expect(second.features.length).toBe(128 * second.frameCount);
  });

  it('can emit raw log-mel features without per-feature normalization', () => {
    const processor = new JSMelProcessor({
      nMels: 128,
      validLengthMode: 'centered',
      normalization: 'none',
    });
    const audio = createSineWave(1600);

    const raw = processor.computeRawMel(audio);
    const result = processor.process(audio);

    expect(result.frameCount).toBe(11);
    expect(result.length).toBe(11);
    expect(result.features.length).toBe(128 * 11);
    expect(result.features).toEqual(raw.rawMel);
  });

  it('reuses cached prefix work in incremental mode while keeping feature shapes stable', () => {
    const processor = new IncrementalJSMelProcessor({ nMels: 128 });
    const first = processor.process(createSineWave(1600));
    const second = processor.process(createSineWave(3200), 1600);

    expect(first.cached).toBe(false);
    expect(first.frameCount).toBe(11);
    expect(first.length).toBe(10);
    expect(second.cached).toBe(true);
    expect(second.cachedFrames).toBeGreaterThan(0);
    expect(second.newFrames).toBeGreaterThan(0);
    expect(second.frameCount).toBe(21);
    expect(second.length).toBe(20);
    expect(second.features.length).toBe(128 * 21);
    expect(Array.from(second.features).every(Number.isFinite)).toBe(true);
  });
});
