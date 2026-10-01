import { test } from 'node:test';
import assert from 'node:assert/strict';
import { cascadeSum, normalizeCandidate } from './normalization-reduction.mjs';

test('exact integer sums across scalar/vector/cascade boundaries', () => {
  for (const width of [8, 16]) for (const size of [1, 2, 15, 16, 17, 63, 64, 65, 101, 129, 257, 1101, 4097]) {
    assert.equal(cascadeSum(new Float32Array(size).fill(1), width), size);
  }
});

test('masked physical tail and input bytes are preserved', () => {
  const raw = Float32Array.of(1, 2, 3, 999);
  const before = new Uint8Array(raw.buffer).slice();
  const result = normalizeCandidate(raw, 4, 3, 8);
  assert.deepEqual(new Uint8Array(raw.buffer), before);
  assert.equal(result.mean[0], 2);
  assert.equal(result.std[0], Math.fround(1 + Math.fround(1e-5)));
  assert.equal(result.features[3], 0);
  assert.equal(result.features[0], Math.fround(-1 / result.std[0]));
});

test('observed constant masked reference discriminates explicit widths', () => {
  const raw = new Float32Array(128 * 101).fill(-16.63553237915039);
  const result = normalizeCandidate(raw, 101, 100, 8);
  for (let row = 0; row < 128; row++) {
    assert.equal(result.mean[row], -16.635530471801758);
    assert.equal(result.std[row], 1.1916957191715483e-5);
    for (let frame = 0; frame < 100; frame++) assert.equal(result.features[row*101+frame], -0.1600533276796341);
    assert.equal(result.features[row*101+100], 0);
  }
  assert.notDeepEqual(normalizeCandidate(raw, 101, 100, 16).features, result.features);
});

test('singleton valid prefix normalizes to zero', () => {
  const result = normalizeCandidate(Float32Array.of(3, 999), 2, 1, 16);
  assert.deepEqual(Array.from(result.features), [0, 0]);
  assert.equal(result.std[0], Math.fround(1e-5));
});

test('invalid tensor contracts fail closed without coercion', () => {
  for (const raw of [[], new Float64Array(3), new Float32Array(), Float32Array.of(NaN), Float32Array.of(Infinity)])
    assert.throws(() => cascadeSum(raw, 8));
  for (const width of [0, 7, true, '8']) assert.throws(() => cascadeSum(Float32Array.of(1), width));
  for (const frames of [0, 1.5, true, 3]) assert.throws(() => normalizeCandidate(Float32Array.of(1, 2), frames, 1, 8));
  for (const length of [0, 3, true, 1.5]) assert.throws(() => normalizeCandidate(Float32Array.of(1, 2), 2, length, 8));
});
