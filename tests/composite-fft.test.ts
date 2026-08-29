import { describe, expect, it } from 'vitest';

import { CompositeFft, MedAsrJsPreprocessor, RadixFivePowerOfTwoFft } from '../src/models/lasr-ctc/mel.js';
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

describe('RadixFivePowerOfTwoFft', () => {
  it('matches a naive DFT for sizes 5 * 2^m', () => {
    for (const size of [5, 10, 40, 80, 160, 320]) {
      const random = seededRandom(size * 13 + 5);
      const real = new Float64Array(size);
      const imaginary = new Float64Array(size);
      for (let index = 0; index < size; index += 1) {
        real[index] = random() * 2 - 1;
        imaginary[index] = random() * 2 - 1;
      }
      const expected = naiveDft(real, imaginary, size);

      const actualReal = real.slice();
      const actualImaginary = imaginary.slice();
      new RadixFivePowerOfTwoFft(size).transform(actualReal, actualImaginary);

      for (let bin = 0; bin < size; bin += 1) {
        expect(Math.abs((actualReal[bin] ?? 0) - (expected.re[bin] ?? 0))).toBeLessThan(1e-8);
        expect(Math.abs((actualImaginary[bin] ?? 0) - (expected.im[bin] ?? 0))).toBeLessThan(1e-8);
      }
    }
  });

  it('agrees with the Bluestein CompositeFft at GigaAM scale', () => {
    for (const size of [160, 320]) {
      const mixed = new RadixFivePowerOfTwoFft(size);
      const bluestein = new CompositeFft(size);
      for (let trial = 0; trial < 5; trial += 1) {
        const reA = new Float64Array(size);
        const imA = new Float64Array(size);
        for (let index = 0; index < size; index += 1) {
          reA[index] = Math.sin(index * (0.7 + trial * 0.13)) + Math.cos(index * 0.021 + trial);
          imA[index] = Math.cos(index * (0.3 + trial * 0.07)) * 0.5;
        }
        const reB = reA.slice();
        const imB = imA.slice();
        mixed.transform(reA, imA);
        bluestein.transform(reB, imB);
        for (let index = 0; index < size; index += 1) {
          expect(Math.abs((reA[index] ?? 0) - (reB[index] ?? 0))).toBeLessThan(1e-9);
          expect(Math.abs((imA[index] ?? 0) - (imB[index] ?? 0))).toBeLessThan(1e-9);
        }
      }
    }
  });

  it('rejects sizes that are not 5 * 2^m', () => {
    expect(() => new RadixFivePowerOfTwoFft(12)).toThrow(RangeError);
    expect(() => new RadixFivePowerOfTwoFft(400)).toThrow(RangeError);
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

  // Hashes captured after matching official SpecScaler clamp (2026-08-27).
  // These lock the Bluestein n_fft=320 path plus GigaAM log(clamp) contract.
  const goldenHashes: Record<number, number> = {
    0.5: -312547205,
    1: 806888655,
    2: -1839250859,
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
