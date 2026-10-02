import assert from 'node:assert/strict';
import test from 'node:test';
import { fftRadix4F32 } from './radix4-f32.mjs';

function dft(real, imag) {
  const n = real.length;
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) {
    const angle = -2 * Math.PI * k * j / n;
    re[k] += real[j] * Math.cos(angle) - imag[j] * Math.sin(angle);
    im[k] += real[j] * Math.sin(angle) + imag[j] * Math.cos(angle);
  }
  return { real: re, imag: im };
}

test('radix4 four-point butterfly has exact sign and bin order', () => {
  const result = fftRadix4F32(Float32Array.of(1, 2, 3, 4), new Float32Array(4));
  assert.deepEqual(result.real, Float32Array.of(10, -2, -2, -2));
  assert.ok(result.imag[0] === 0 && result.imag[2] === 0);
  assert.equal(result.imag[1], 2);
  assert.equal(result.imag[3], -2);
});

test('radix4 impulse preserves all bins and owns its outputs', () => {
  const re = new Float32Array(256), im = new Float32Array(256);
  re[0] = 1;
  const result = fftRadix4F32(re, im);
  assert.deepEqual(result.real, new Float32Array(256).fill(1));
  assert.ok(result.imag.every(value => value === 0));
  assert.notEqual(result.real, re);
  assert.equal(re[0], 1);
  assert.equal(re[1], 0);
});

test('radix4 cardinal rotations do not introduce spurious components', () => {
  for (const products of ['separate', 'single-round']) {
    for (const amplitude of [1, Math.fround(1e38)]) {
      const re = new Float32Array(16);
      re[2] = amplitude;
      const result = fftRadix4F32(re, new Float32Array(16), { products });
      for (const k of [2, 6, 10, 14]) {
        assert.ok(result.real[k] === 0, `${products}/${amplitude}/bin${k}`);
        assert.equal(result.imag[k], (k % 8 === 2 ? -1 : 1) * amplitude);
      }
    }
  }
});

test('radix4 has negative exponent and natural output order', () => {
  const re = Float32Array.from({ length: 64 }, (_, j) => Math.cos(2 * Math.PI * 5 * j / 64));
  const im = Float32Array.from({ length: 64 }, (_, j) => Math.sin(2 * Math.PI * 5 * j / 64));
  const result = fftRadix4F32(re, im);
  assert.ok(Math.abs(result.real[5] - 64) < 1e-5);
  for (let k = 0; k < 64; k++) if (k !== 5) {
    assert.ok(Math.abs(result.real[k]) < 1e-5);
    assert.ok(Math.abs(result.imag[k]) < 1e-5);
  }
});

test('radix4 both product modes match independent complex DFT', () => {
  for (const n of [4, 16, 64, 256]) {
    const re = Float32Array.from({ length: n }, (_, j) => Math.sin(j * 0.73) * 0.4);
    const im = Float32Array.from({ length: n }, (_, j) => Math.cos(j * 0.31) * 0.3);
    const beforeRe = re.slice(), beforeIm = im.slice(), expected = dft(re, im);
    for (const products of ['separate', 'single-round']) {
      const actual = fftRadix4F32(re, im, { products });
      for (const component of ['real', 'imag']) for (let k = 0; k < n; k++) {
        assert.ok(Math.abs(actual[component][k] - expected[component][k]) < 2e-5,
          `${n}/${products}/${component}/${k}`);
      }
    }
    assert.deepEqual(re, beforeRe);
    assert.deepEqual(im, beforeIm);
  }
});

test('radix4 rejects malformed or unsupported diagnostic inputs', () => {
  for (const n of [0, 1, 2, 8, 32, 512]) {
    assert.throws(() => fftRadix4F32(new Float32Array(n), new Float32Array(n)));
  }
  assert.throws(() => fftRadix4F32(new Float64Array(4), new Float32Array(4)));
  assert.throws(() => fftRadix4F32(new Float32Array(4), new Float32Array(16)));
  assert.throws(() => fftRadix4F32(Float32Array.of(NaN, 0, 0, 0), new Float32Array(4)));
  assert.throws(() => fftRadix4F32(new Float32Array(4), new Float32Array(4), { products: 'fma' }));
  assert.throws(() => fftRadix4F32(new Float32Array(4), new Float32Array(4), null));
  assert.throws(() => fftRadix4F32(new Float32Array(4), new Float32Array(4), { unexpected: true }));
});
