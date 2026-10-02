import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { packedF32Source } from './packed-f32-source.mjs';
const source = readFileSync(new URL('../../../../dist/audio/js-mel.js', import.meta.url), 'utf8');
const candidate = await import('data:text/javascript;base64,' + Buffer.from(packedF32Source(source)).toString('base64'));

test('product option matrix preserves defaults and changes only named boundaries', () => {
  const baseline = packedF32Source(source);
  assert.equal(baseline, packedF32Source(source, {
    butterflyProducts: 'separate', recombinationProducts: 'separate',
  }));
  const split = text => text.indexOf('export class JSMelProcessor');
  for (const butterflyProducts of ['separate', 'single-round']) {
    for (const recombinationProducts of ['separate', 'single-round']) {
      const output = packedF32Source(source, { butterflyProducts, recombinationProducts });
      const changed = [...output.matchAll(/const (?:tRe|tIm|tr|ti) = Math\.fround\([^;]+;/g)];
      assert.equal(changed.length, 8);
      assert.equal(changed.filter(item => !item[0].includes('Math.fround(Math.fround(')).length,
        (butterflyProducts === 'single-round' ? 6 : 0) + (recombinationProducts === 'single-round' ? 2 : 0));
      if (butterflyProducts === 'separate') assert.equal(output.slice(0, split(output)), baseline.slice(0, split(baseline)));
      if (recombinationProducts === 'separate') assert.equal(output.slice(split(output)), baseline.slice(split(baseline)));
      assert.throws(() => packedF32Source(output), /contract/);
    }
  }
  for (const options of [null, [], { typo: 'single-round' }, { butterflyProducts: true },
    { recombinationProducts: 'hardware-fma' }]) assert.throws(() => packedF32Source(source, options), /Invalid/);
});

test('source drift and duplicate application fail closed', () => {
  assert.throws(() => packedF32Source(source.replace('const xeR =', 'const changed =')));
  assert.throws(() => packedF32Source(packedF32Source(source)));
});

test('generated product sums preserve distinct cancellation and ties-to-even rounding', () => {
  const cases = [
    { a: 1 + 2 ** -23, b: 1 - 2 ** -23, c: 1, d: 1, separate: 0, single: -(2 ** -46) },
    { a: 1 + 2 ** -23, b: 1.5, c: 1.5, d: 1, separate: 2 ** -22, single: 3 * 2 ** -24 },
  ];
  for (const mode of ['separate', 'single-round']) {
    const output = packedF32Source(source, { butterflyProducts: mode, recombinationProducts: mode });
    const butterfly = output.match(/const tRe = (Math\.fround\([^;]+);/)[1];
    const recombination = output.match(/const tr = (Math\.fround\([^;]+);/)[1];
    const evaluateButterfly = new Function('re', 'im', 'q', 'wCos', 'wSin', `return ${butterfly};`);
    const evaluateRecombination = new Function('xoR', 'wc', 'xoI', 'ws', `return ${recombination};`);
    for (const point of cases) {
      for (const key of ['a', 'b', 'c', 'd']) assert.equal(Math.fround(point[key]), point[key]);
      const expected = mode === 'separate' ? point.separate : point.single;
      assert.equal(evaluateButterfly([point.a], [point.c], 0, point.b, point.d), expected);
      assert.equal(evaluateRecombination(point.a, point.b, point.c, point.d), expected);
    }
  }
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

test('actual generated FFT variants differ but retain the negative-exponent DFT contract', async () => {
  const x = Math.fround(1 + 2 ** -12), spectra = [];
  for (const butterflyProducts of ['separate', 'single-round']) {
    const module = await import('data:text/javascript;base64,' +
      Buffer.from(packedF32Source(source, { butterflyProducts })).toString('base64'));
    const re = new Float64Array(16), im = new Float64Array(16);
    re[0] = 1; re[3] = x;
    module.fft(re, im, 16, module.precomputeTwiddles(16));
    for (let k = 0; k < 16; k++) {
      const angle = -2 * Math.PI * 3 * k / 16;
      assert.ok(Math.hypot(re[k] - (1 + x * Math.cos(angle)), im[k] - x * Math.sin(angle)) < 1e-6);
    }
    spectra.push([...re, ...im]);
  }
  assert.notDeepEqual(spectra[0], spectra[1]);
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
