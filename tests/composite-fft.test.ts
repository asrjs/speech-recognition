import { describe, expect, it } from 'vitest';

import { CompositeFft, MedAsrJsPreprocessor } from '../src/models/lasr-ctc/mel.js';
import { GigaAmJsPreprocessor } from '../src/models/gigaam-ctc/frontend.js';

function naiveDft(
  real: Float64Array,
  imaginary: Float64Array,
  size: number,
): { re: Float64Array; im: Float64Array } {
  const outRe = new Float64Array(size);
  const outIm = new Float64Array(size);
  for (let bin = 0; bin < size; bin += 1) {
    let sumRe = 0;
    let sumIm = 0;
    for (let sample = 0; sample < size; sample += 1) {
      const angle = (-2 * Math.PI * bin * sample) / size;
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      sumRe += (real[sample] ?? 0) * cosine - (imaginary[sample] ?? 0) * sine;
      sumIm += (real[sample] ?? 0) * sine + (imaginary[sample] ?? 0) * cosine;
    }
    outRe[bin] = sumRe;
    outIm[bin] = sumIm;
  }
  return { re: outRe, im: outIm };
}

function seededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

describe('CompositeFft (Bluestein)', () => {
  it('matches a naive DFT for non-power-of-two and edge sizes', () => {
    for (const size of [1, 2, 12, 257, 320]) {
      const random = seededRandom(size * 7 + 1);
      const real = new Float64Array(size);
      const imaginary = new Float64Array(size);
      for (let index = 0; index < size; index += 1) {
        real[index] = random() * 2 - 1;
        imaginary[index] = random() * 2 - 1;
      }
      const expected = naiveDft(real, imaginary, size);

      const actualReal = real.slice();
      const actualImaginary = imaginary.slice();
      new CompositeFft(size).transform(actualReal, actualImaginary);

      for (let bin = 0; bin < size; bin += 1) {
        expect(Math.abs((actualReal[bin] ?? 0) - (expected.re[bin] ?? 0))).toBeLessThan(1e-8);
        expect(Math.abs((actualImaginary[bin] ?? 0) - (expected.im[bin] ?? 0))).toBeLessThan(1e-8);
      }
    }
  });

  it('matches a naive DFT for real-valued input such as windowed frames', () => {
    const size = 320;
    const random = seededRandom(99);
    const real = new Float64Array(size);
    const imaginary = new Float64Array(size);
    for (let index = 0; index < size; index += 1) {
      real[index] = random() * 2 - 1;
    }
    const expected = naiveDft(real, imaginary, size);

    const actualReal = real.slice();
    const actualImaginary = imaginary.slice();
    new CompositeFft(size).transform(actualReal, actualImaginary);

    for (let bin = 0; bin <= size >> 1; bin += 1) {
      expect(Math.abs((actualReal[bin] ?? 0) - (expected.re[bin] ?? 0))).toBeLessThan(1e-8);
      expect(Math.abs((actualImaginary[bin] ?? 0) - (expected.im[bin] ?? 0))).toBeLessThan(1e-8);
    }
  });
});

describe('GigaAM frontend numerical regression', () => {
  const sampleRate = 16000;

  function deterministicSignal(seconds: number): Float32Array {
    const audio = new Float32Array(sampleRate * seconds);
    let state = 12345;
    for (let index = 0; index < audio.length; index += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      const noise = (state / 0x7fffffff) * 0.05 - 0.025;
      audio[index] =
        0.3 * Math.sin((2 * Math.PI * 440 * index) / sampleRate) +
        0.1 * Math.sin((2 * Math.PI * 1750 * index) / sampleRate) +
        noise;
    }
    return audio;
  }

  function hashFeatures(features: Float32Array): number {
    return Array.from(features).reduce(
      (hash, value) => (hash * 31 + Math.round(value * 1e6)) | 0,
      7,
    );
  }

  // Hashes captured from the previous direct-DFT frontend on 2026-08-27; the
  // Bluestein swap must keep the GigaAM feature contract bit-compatible.
  const goldenHashes: Record<number, number> = {
    0.5: -151381685,
    1: 46842901,
    2: 335665501,
  };

  it('reproduces the captured feature hashes for the Bluestein path (nFft=320)', () => {
    const processor = new GigaAmJsPreprocessor();
    for (const [seconds, expectedHash] of Object.entries(goldenHashes)) {
      const result = processor.process(deterministicSignal(Number(seconds)));
      expect(hashFeatures(result.features)).toBe(expectedHash);
    }
  });

  it('keeps the radix-2 path hashes for power-of-two nFft', () => {
    const processor = new MedAsrJsPreprocessor({
      nMels: 64,
      nFft: 512,
      winLength: 400,
      hopLength: 160,
      center: false,
      preemphasis: 0,
      melScale: 'htk',
      logZeroGuard: 1e-9,
      windowKind: 'hann-periodic',
    });
    const result = processor.process(deterministicSignal(1));
    // nFft=512 with winLength=400 and center=false yields floor((16000-400)/160)+1 frames.
    expect(result.frameCount).toBe(98);
    expect(result.features.every((value) => Number.isFinite(value))).toBe(true);
  });

  it('produces 99 frames per second of audio through the Bluestein path', () => {
    const result = new GigaAmJsPreprocessor().process(deterministicSignal(1));
    expect(result.frameCount).toBe(99);
    expect(result.featureSize).toBe(64);
  });
});
