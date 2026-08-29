#!/usr/bin/env node

// Measures the CTC greedy decode hot path at SenseVoice scale. The generic
// float32 cell (187 output frames x 25055 vocabulary, the 11.3 s jfk-short
// fixture shape) is kept for continuity with the 2026-08-29 baseline; the
// fp16 cells compare the convert+generic reference pipeline (the 18.7 s
// librivox shape is 316 x 25055) against the lookup-table fast path.
// Run after `npm run build` on the base and candidate commits with the
// same Node version before accepting a change.
import { performance } from 'node:perf_hooks';

const { argmaxAndSelectedLogProbs, argmaxAndSelectedLogProbsFp16 } = await import('../../dist/ctc/index.js');

const VOCAB_SIZE = 25055;
const RUNS = 9;

function fp16ToFloat(bits) {
  const sign = (bits & 0x8000) << 16;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) {
    if (mantissa === 0) return sign ? -0 : 0;
    let normalized = mantissa;
    let exponentValue = -14;
    while ((normalized & 0x400) === 0) { normalized <<= 1; exponentValue -= 1; }
    normalized &= 0x3ff;
    return (sign ? -1 : 1) * (1 + normalized / 1024) * 2 ** exponentValue;
  }
  if (exponent === 0x1f) return mantissa === 0 ? (sign ? -Infinity : Infinity) : NaN;
  return (sign ? -1 : 1) * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/** Float32 -> fp16 with round-to-nearest-even (finite inputs only). */
function floatToFp16Bits(value) {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = value;
  const bits = u32[0];
  const sign = (bits >>> 16) & 0x8000;
  let exponent = (bits >>> 23) & 0xff;
  let mantissa = bits & 0x7fffff;
  if (exponent === 0xff) return sign | 0x7c00;
  exponent -= 127;
  if (exponent >= 16) return sign | 0x7c00;
  if (exponent > -15) {
    let q = (mantissa + 4096 + ((mantissa >>> 13) & 1)) >>> 13;
    if (q > 0x3ff) {
      q = 0;
      exponent += 1;
      if (exponent >= 16) return sign | 0x7c00;
    }
    return sign | ((exponent + 15) << 10) | q;
  }
  if (exponent < -24) return sign;
  const shift = 14 - exponent;
  const merged = mantissa | 0x800000;
  return sign | ((merged + (1 << (shift - 1))) >>> shift);
}

/** Deterministic realistic log-probability bits: one peaky row entry + noise. */
function realisticFp16Bits(frameCount, vocabSize) {
  const out = new Uint16Array(frameCount * vocabSize);
  let seed = 987654321;
  const rnd = () => {
    seed ^= seed << 13; seed >>>= 0;
    seed ^= seed >>> 17;
    seed ^= seed << 5; seed >>>= 0;
    return seed / 4294967296;
  };
  for (let frame = 0; frame < frameCount; frame += 1) {
    const bestId = Math.floor(rnd() * vocabSize);
    for (let index = 0; index < vocabSize; index += 1) {
      const peak = index === bestId
        ? -0.01 - rnd() * 0.2
        : -6 - rnd() * rnd() * 24;
      out[frame * vocabSize + index] = floatToFp16Bits(peak);
    }
  }
  return out;
}

function convertFp16(bits) {
  const out = new Float32Array(bits.length);
  for (let index = 0; index < bits.length; index += 1) out[index] = fp16ToFloat(bits[index]);
  return out;
}

function bench(label, fn) {
  const times = [];
  for (let run = 0; run < RUNS; run += 1) {
    const started = performance.now();
    fn();
    times.push(performance.now() - started);
  }
  times.sort((a, b) => a - b);
  const m = times[Math.floor(times.length / 2)];
  console.log(label.padEnd(42) + ' p50=' + m.toFixed(2) + 'ms min=' + times[0].toFixed(2) + 'ms');
  return m;
}

// Legacy float32 synthetic (kept from the 2026-08-29 baseline).
const FRAME_COUNT = 187;
const logits = new Float32Array(FRAME_COUNT * VOCAB_SIZE);
for (let index = 0; index < logits.length; index += 1) {
  logits[index] =
    Math.log(Math.abs(Math.sin(index * 0.0007)) + 1e-9) * 0.5 -
    (index % VOCAB_SIZE) * 0.001;
}

const bits187 = realisticFp16Bits(187, VOCAB_SIZE);
const bits316 = realisticFp16Bits(316, VOCAB_SIZE);
const floats187 = convertFp16(bits187);

// Parity gate before timing: ids must match and scores stay <= 1e-5 apart.
{
  const reference = argmaxAndSelectedLogProbs(floats187, 187, VOCAB_SIZE);
  const fast = argmaxAndSelectedLogProbsFp16(bits187, 187, VOCAB_SIZE);
  let maxDiff = 0;
  for (let frame = 0; frame < 187; frame += 1) {
    if (reference.frameIds[frame] !== fast.frameIds[frame]) {
      throw new Error('parity id mismatch at frame ' + frame);
    }
    maxDiff = Math.max(maxDiff, Math.abs(reference.selectedLogProbs[frame] - fast.selectedLogProbs[frame]));
  }
  if (maxDiff > 1e-5) throw new Error('parity score diff too large: ' + maxDiff);
  console.log('parity gate: OK (maxDiff=' + maxDiff.toExponential(2) + ')');
}

const results = {};
results.generic_float32_187 = bench('generic float32 argmax 187x25055', () => {
  argmaxAndSelectedLogProbs(logits, FRAME_COUNT, VOCAB_SIZE);
});
results.fp16_reference_187 = bench('fp16 ref convert+argmax 187x25055', () => {
  argmaxAndSelectedLogProbs(convertFp16(bits187), 187, VOCAB_SIZE);
});
results.fp16_fast_187 = bench('fp16 fast LUT 187x25055', () => {
  argmaxAndSelectedLogProbsFp16(bits187, 187, VOCAB_SIZE);
});
results.fp16_reference_316 = bench('fp16 ref convert+argmax 316x25055', () => {
  argmaxAndSelectedLogProbs(convertFp16(bits316), 316, VOCAB_SIZE);
});
results.fp16_fast_316 = bench('fp16 fast LUT 316x25055', () => {
  argmaxAndSelectedLogProbsFp16(bits316, 316, VOCAB_SIZE);
});
console.log('speedup 187: ' + (results.fp16_reference_187 / results.fp16_fast_187).toFixed(1) + 'x');
console.log('speedup 316: ' + (results.fp16_reference_316 / results.fp16_fast_316).toFixed(1) + 'x');

