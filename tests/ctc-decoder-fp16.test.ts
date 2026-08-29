import { describe, expect, test } from 'vitest';

import { argmaxAndSelectedLogProbs, argmaxAndSelectedLogProbsFp16 } from '../src/ctc/decoder.js';

// ---------------------------------------------------------------------------
// Parity tests for the float16 lookup-table fast path
// ---------------------------------------------------------------------------
// The fast path consumes raw fp16 bit patterns; the reference pipeline is
// the generic float argmax over the exact fp16->fp32 conversion of the same
// bits (Float32 represents every fp16 value exactly). The converter below
// mirrors the decoder-internal and runtime readLogits conversions verbatim
// so both sides of each assertion see identical float values.

function fp16BitsToFloat(bits: number): number {
  const sign = (bits & 0x8000) << 16;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) {
    if (mantissa === 0) {
      return sign ? -0 : 0;
    }
    let normalized = mantissa;
    let exponentValue = -14;
    while ((normalized & 0x400) === 0) {
      normalized <<= 1;
      exponentValue -= 1;
    }
    normalized &= 0x3ff;
    return (sign ? -1 : 1) * (1 + normalized / 1024) * 2 ** exponentValue;
  }
  if (exponent === 0x1f) {
    return mantissa === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }
  return (sign ? -1 : 1) * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

function convertAll(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  for (let index = 0; index < bits.length; index += 1) {
    out[index] = fp16BitsToFloat(bits[index]!);
  }
  return out;
}

/** Deterministic xorshift PRNG so failures are reproducible. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0x100000000;
  };
}

/**
 * Sample fp16 codes by rejection over the uniform 16-bit code space, keeping
 * only codes whose exact float value falls inside [minValue, maxValue]. This
 * naturally covers normals, both denormal bands, and signed zeros.
 */
function sampleCodes(count: number, minValue: number, maxValue: number, random: () => number): Uint16Array {
  const out = new Uint16Array(count);
  for (let index = 0; index < count; index += 1) {
    for (;;) {
      const code = Math.floor(random() * 0x10000) & 0xffff;
      const value = fp16BitsToFloat(code);
      if (value >= minValue && value <= maxValue) {
        out[index] = code;
        break;
      }
    }
  }
  return out;
}

function expectParity(bits: Uint16Array, frameCount: number, vocabSize: number): void {
  const reference = argmaxAndSelectedLogProbs(convertAll(bits), frameCount, vocabSize);
  const fast = argmaxAndSelectedLogProbsFp16(bits, frameCount, vocabSize);
  expect(fast.frameIds).toEqual(reference.frameIds);
  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const expected = reference.selectedLogProbs[frameIndex]!;
    const actual = fast.selectedLogProbs[frameIndex]!;
    if (Number.isNaN(expected)) {
      expect(Number.isNaN(actual)).toBe(true);
    } else {
      // The fast path evaluates best - log(sum(exp(x))) without the max
      // shift, so results can differ by a few fp64 rounding steps.
      expect(Math.abs(actual - expected)).toBeLessThan(1e-5);
    }
  }
}

describe('argmaxAndSelectedLogProbsFp16', () => {
  test('matches the generic pipeline on realistic log-probability rows', () => {
    const random = createRandom(20260830);
    for (let trial = 0; trial < 12; trial += 1) {
      const frameCount = 5 + (trial % 4);
      const vocabSize = 37 + trial * 11;
      const bits = sampleCodes(frameCount * vocabSize, -30, 0.5, random);
      expectParity(bits, frameCount, vocabSize);
    }
  });

  test('handles denormals, signed zeros, and negatives', () => {
    const vocabSize = 12;
    const bits = new Uint16Array(vocabSize * 2);
    // Row 0: signed zeros, smallest denormals, and normal negatives.
    const row0 = [0x0000, 0x8000, 0x0001, 0x0003, 0x0400, 0x8001, 0x8400, 0xb400, 0xc400, 0x1c00, 0x03ff, 0x8002];
    for (let index = 0; index < vocabSize; index += 1) bits[index] = row0[index]!;
    // Row 1: denormal-dominant negative row.
    const row1 = [0x0200, 0x8200, 0x0100, 0x8100, 0x0002, 0x8002, 0x03ff, 0x83ff, 0x0004, 0x8004, 0x0005, 0x8005];
    for (let index = 0; index < vocabSize; index += 1) bits[vocabSize + index] = row1[index]!;
    expectParity(bits, 2, vocabSize);
    // Sanity: both rows stay inside the fast-path safe zone.
    const result = argmaxAndSelectedLogProbsFp16(bits, 2, vocabSize);
    expect(result.frameIds[0]).toBe(9);
    expect(result.frameIds[1]).toBe(6);
  });

  test('falls back to the generic pipeline outside the exp safe zone', () => {
    const random = createRandom(777);
    const frameCount = 4;
    const vocabSize = 64;
    // Raw-logit style rows with maxima well above +80 and below -80.
    const wide = sampleCodes(frameCount * vocabSize, -300, 300, random);
    expectParity(wide, frameCount, vocabSize);
  });

  test('falls back on rows containing NaN and infinity codes', () => {
    const random = createRandom(4242);
    const frameCount = 3;
    const vocabSize = 48;
    const bits = sampleCodes(frameCount * vocabSize, -30, 0.5, random);
    bits[10] = 0x7c00; // +infinity
    bits[vocabSize + 11] = 0xfc00; // -infinity
    bits[2 * vocabSize + 12] = 0x7e01; // quiet NaN
    expectParity(bits, frameCount, vocabSize);
  });

  test('pads short buffers like the generic pipeline', () => {
    const random = createRandom(31337);
    const vocabSize = 32;
    const frameCount = 3;
    const full = sampleCodes(frameCount * vocabSize, -30, 0.5, random);
    const truncated = full.slice(0, vocabSize + 5);
    // Rows 0 and 1 are complete-ish; row 2 is entirely missing.
    expectParity(truncated, frameCount, vocabSize);
  });
});

