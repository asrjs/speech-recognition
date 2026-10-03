import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createWindowTokenFixture } from './window-token-fixture.mjs';

test('rejects oversized or empty fixtures before creating repeated PCM', () => {
  assert.throws(() => createWindowTokenFixture(new Float32Array(19 * 16000 + 1)), /19 seconds/);
  assert.throws(() => createWindowTokenFixture(new Float32Array(0)), /19 seconds/);
});

test('creates bounded disjoint windows without changing the source', () => {
  const source = Float32Array.of(0.25, -0.5);
  const fixture = createWindowTokenFixture(source);
  assert.equal(fixture.pcm.length, 40 * 16000);
  assert.deepEqual([...fixture.pcm.subarray(0, 2)], [...source]);
  assert.deepEqual([...fixture.pcm.subarray(20 * 16000, 20 * 16000 + 2)], [...source]);
  assert.equal(fixture.pcm[2], 0);
  assert.deepEqual([...source], [0.25, -0.5]);
  assert.deepEqual(fixture.windowing, { windowSeconds: 20, durationSeconds: 40 });
  assert.equal(fixture.options.returnTokens, true);
  assert.equal(fixture.options.windowing, 'force');
  assert.equal(fixture.options.overlapSeconds, 0);
});

test('places speech once wholly inside two overlapping windows', () => {
  const source = Float32Array.of(0.25, -0.5);
  const fixture = createWindowTokenFixture(source, true);
  assert.equal(fixture.pcm.length, 34 * 16000);
  assert.deepEqual([...fixture.pcm.subarray(15 * 16000, 15 * 16000 + 2)], [...source]);
  assert.equal(fixture.pcm[0], 0);
  assert.equal(fixture.options.overlapSeconds, 6);
  assert.deepEqual(fixture.windowing, { windowSeconds: 20, durationSeconds: 34, overlapSeconds: 6 });
  assert.throws(() => createWindowTokenFixture(new Float32Array(5 * 16000 + 1), true), /5 seconds/);
});
