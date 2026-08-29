import { describe, expect, it } from 'vitest';

import { CompositeFft } from '../src/models/lasr-ctc/mel.js';
import { isMixedRadixSize, MixedRadixFft } from '../src/audio/mixed-radix-fft.js';

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

describe('isMixedRadixSize', () => {
  it('accepts sizes of the form 5^a * 2^b', () => {
    for (const size of [1, 2, 4, 5, 10, 20, 25, 40, 50, 80, 100, 160, 200, 320, 400, 500, 800, 1000, 1600]) {
      expect(isMixedRadixSize(size)).toBe(true);
    }
  });

  it('rejects other sizes', () => {
    for (const size of [0, -5, 3, 6, 7, 12, 15, 256.5, 321, 1023]) {
      expect(isMixedRadixSize(size)).toBe(false);
    }
  });
});

describe('MixedRadixFft', () => {
  it('matches a naive DFT for supported sizes incl. the Whisper n_fft=400', () => {
    for (const size of [5, 10, 25, 40, 50, 100, 200, 400, 500]) {
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
      new MixedRadixFft(size).transform(actualReal, actualImaginary);

      for (let bin = 0; bin < size; bin += 1) {
        expect(Math.abs((actualReal[bin] ?? 0) - (expected.re[bin] ?? 0))).toBeLessThan(1e-8);
        expect(Math.abs((actualImaginary[bin] ?? 0) - (expected.im[bin] ?? 0))).toBeLessThan(1e-8);
      }
    }
  });

  it('matches a naive DFT for power-of-two sizes through the radix-2 kernel', () => {
    for (const size of [1, 2, 4, 8, 16]) {
      const random = seededRandom(size * 31 + 3);
      const real = new Float64Array(size);
      const imaginary = new Float64Array(size);
      for (let index = 0; index < size; index += 1) {
        real[index] = random() * 2 - 1;
        imaginary[index] = random() * 2 - 1;
      }
      const expected = naiveDft(real, imaginary, size);

      const actualReal = real.slice();
      const actualImaginary = imaginary.slice();
      new MixedRadixFft(size).transform(actualReal, actualImaginary);

      for (let bin = 0; bin < size; bin += 1) {
        expect(Math.abs((actualReal[bin] ?? 0) - (expected.re[bin] ?? 0))).toBeLessThan(1e-9);
        expect(Math.abs((actualImaginary[bin] ?? 0) - (expected.im[bin] ?? 0))).toBeLessThan(1e-9);
      }
    }
  });

  it('transformRealInput writes the correct leading bins', () => {
    const size = 400;
    const binCount = size / 2 + 1;
    const random = seededRandom(42);
    const input = new Float32Array(size);
    for (let index = 0; index < size; index += 1) {
      input[index] = random() * 2 - 1;
    }
    const real = new Float64Array(size);
    for (let index = 0; index < size; index += 1) {
      real[index] = input[index] as number;
    }
    const expected = naiveDft(real, new Float64Array(size), size);

    const outReal = new Float64Array(binCount);
    const outImaginary = new Float64Array(binCount);
    new MixedRadixFft(size).transformRealInput(input, outReal, outImaginary, binCount);

    let maxError = 0;
    for (let bin = 0; bin < binCount; bin += 1) {
      maxError = Math.max(
        maxError,
        Math.abs((outReal[bin] ?? 0) - (expected.re[bin] ?? 0)),
        Math.abs((outImaginary[bin] ?? 0) - (expected.im[bin] ?? 0)),
      );
    }
    expect(maxError).toBeLessThan(1e-9);
  });

  it('agrees with the Bluestein CompositeFft on windowed-real frames', () => {
    for (const size of [200, 400]) {
      const mixed = new MixedRadixFft(size);
      const bluestein = new CompositeFft(size);
      for (let trial = 0; trial < 5; trial += 1) {
        const reA = new Float64Array(size);
        const imA = new Float64Array(size);
        const reB = new Float64Array(size);
        const imB = new Float64Array(size);
        for (let index = 0; index < size; index += 1) {
          const value =
            Math.sin(index * (0.7 + trial * 0.13)) + 0.5 * Math.cos(index * 0.021 + trial);
          reA[index] = value;
          reB[index] = value;
          imA[index] = Math.cos(index * (0.3 + trial * 0.07)) * 0.25;
          imB[index] = imA[index] as number;
        }
        mixed.transform(reA, imA);
        bluestein.transform(reB, imB);
        for (let index = 0; index < size; index += 1) {
          expect(Math.abs((reA[index] ?? 0) - (reB[index] ?? 0))).toBeLessThan(1e-9);
          expect(Math.abs((imA[index] ?? 0) - (imB[index] ?? 0))).toBeLessThan(1e-9);
        }
      }
    }
  });

  it('zeroImaginary matches the explicit-zero path', () => {
    const size = 400;
    const random = seededRandom(7);
    const real = new Float64Array(size);
    for (let index = 0; index < size; index += 1) {
      real[index] = random() * 2 - 1;
    }
    const reA = real.slice();
    const imA = new Float64Array(size);
    for (let index = 0; index < size; index += 1) {
      imA[index] = index % 3 === 0 ? 0.25 : 0;
    }
    const reB = real.slice();
    const imB = new Float64Array(size);
    new MixedRadixFft(size).transform(reA, imA, true);
    new MixedRadixFft(size).transform(reB, imB, false);
    for (let index = 0; index < size; index += 1) {
      expect(Math.abs((reA[index] ?? 0) - (reB[index] ?? 0))).toBeLessThan(1e-12);
      expect(Math.abs((imA[index] ?? 0) - (imB[index] ?? 0))).toBeLessThan(1e-12);
    }
  });

  it('rejects unsupported sizes with a RangeError', () => {
    expect(() => new MixedRadixFft(12)).toThrow(RangeError);
    expect(() => new MixedRadixFft(321)).toThrow(RangeError);
    expect(() => new MixedRadixFft(3)).toThrow(RangeError);
  });
});

