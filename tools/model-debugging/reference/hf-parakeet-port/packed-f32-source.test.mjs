import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { packedF32Source } from './packed-f32-source.mjs';
const source = readFileSync(new URL('../../../../dist/audio/js-mel.js', import.meta.url), 'utf8');
const candidate = await import('data:text/javascript;base64,' + Buffer.from(packedF32Source(source)).toString('base64'));
test('source drift and duplicate application fail closed', () => {
  assert.throws(() => packedF32Source(source.replace('const xeR =', 'const changed =')));
  assert.throws(() => packedF32Source(packedF32Source(source)));
});
test('exact impulse spectrum and negative-exponent quarter turn', () => {
  const re = new Float64Array(256), im = new Float64Array(256);
  re[0] = 1;
  candidate.fft(re, im, 256, candidate.precomputeTwiddles(256));
  assert.ok(re.every(value => value === 1)); assert.ok(im.every(value => value === 0));
  re.fill(0); im.fill(0); re[1] = 1;
  candidate.fft(re, im, 256, candidate.precomputeTwiddles(256));
  assert.ok(Math.abs(re[64]) < 1e-15); assert.equal(im[64], -1);
});
test('all butterfly stores match independent scalar f32 radix2 reference', () => {
  const original = Float32Array.from({length: 256}, (_, i) => Math.sin(i * 0.17) + Math.cos(i * 0.03));
  const re = Float64Array.from(original), im = new Float64Array(256);
  const r = Float32Array.from(original), q = new Float32Array(256);
  const tw = candidate.precomputeTwiddles(256), f = Math.fround;
  for (let i = 0; i < 256; i++) if (i < tw.bitrev[i]) {
    const j = tw.bitrev[i]; [r[i], r[j]] = [r[j], r[i]];
  }
  for (let size = 2; size <= 256; size *= 2) for (let i = 0; i < 256; i += size)
    for (let k = 0; k < size / 2; k++) {
      const a = i + k, b = a + size / 2, index = k * 256 / size;
      const c = size <= 8 && k === size / 4 ? 0 : f(tw.cos[index]);
      const s = size <= 8 && k === size / 4 ? -1 : f(tw.sin[index]);
      const tr = f(f(r[b] * c) - f(q[b] * s));
      const ti = f(f(r[b] * s) + f(q[b] * c));
      r[b] = f(r[a] - tr); q[b] = f(q[a] - ti);
      r[a] = f(r[a] + tr); q[a] = f(q[a] + ti);
    }
  candidate.fft(re, im, 256, tw);
  assert.deepEqual(Array.from(re), Array.from(r)); assert.deepEqual(Array.from(im), Array.from(q));
});
