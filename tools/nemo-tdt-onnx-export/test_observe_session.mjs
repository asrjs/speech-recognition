import test from 'node:test';
import assert from 'node:assert/strict';
import { observeSessionRuns } from './observe_session.mjs';

const tensor = (value) => ({ type: 'float32', dims: [1], data: new Float32Array([value]) });
test('copies pre-call inputs and post-call outputs without replacing original result', async () => {
  const output = { y: tensor(2) },
    feed = { x: tensor(1) };
  class Session {
    inputNames = ['x'];
    async run(feeds, options) {
      assert.equal(options, 'forward');
      feeds.x.data[0] = 3;
      return output;
    }
  }
  const original = Object.getOwnPropertyDescriptor(Session.prototype, 'run');
  const observer = observeSessionRuns(Session, ['x'], ['y'], 1);
  try {
    assert.equal(await new Session().run(feed, 'forward'), output);
    output.y.data[0] = 4;
    assert.deepEqual(observer.calls[0], {
      index: 0,
      feeds: { x: { dtype: 'float32', shape: [1], data: [1] } },
      outputs: { y: { dtype: 'float32', shape: [1], data: [2] } },
    });
    await assert.rejects(new Session().run(feed), /excessive/);
  } finally {
    observer.restore();
    observer.restore();
  }
  assert.deepEqual(Object.getOwnPropertyDescriptor(Session.prototype, 'run'), original);
});

test('does not observe a different session contract and preserves inference errors', async () => {
  const failure = new Error('real inference failure');
  class Session {
    inputNames = ['encoder'];
    async run() {
      throw failure;
    }
  }
  const observer = observeSessionRuns(Session, ['x'], ['y'], 2);
  try {
    await assert.rejects(new Session().run({}), (e) => e === failure);
    assert.equal(observer.calls.length, 0);
  } finally {
    observer.restore();
  }
});

test('fails closed on malformed storage/output and restores after a failed observed call', async () => {
  class Session {
    inputNames = ['x'];
    async run() {
      return { y: tensor(NaN) };
    }
  }
  const observer = observeSessionRuns(Session, ['x'], ['y'], 2);
  try {
    await assert.rejects(new Session().run({ x: { ...tensor(1), data: [1] } }), /storage/);
    await assert.rejects(new Session().run({ x: tensor(1) }), /nonfinite/);
    assert.equal(observer.calls.length, 0);
  } finally {
    observer.restore();
  }
  for (const count of [0, -1, 2001, 1.5, true])
    assert.throws(() => observeSessionRuns(Session, ['x'], ['y'], count), /bounded/);
});

test('rejects concurrent calls and in-flight restoration without changing outputs', async () => {
  let finish;
  const gate = new Promise((resolve) => {
    finish = resolve;
  });
  const output = { y: tensor(2) };
  class Session {
    inputNames = ['x'];
    async run() {
      await gate;
      return output;
    }
  }
  const observer = observeSessionRuns(Session, ['x'], ['y'], 2);
  try {
    const pending = new Session().run({ x: tensor(1) });
    assert.throws(() => observer.restore(), /in-flight/);
    await assert.rejects(new Session().run({ x: tensor(1) }), /Concurrent/);
    finish();
    assert.equal(await pending, output);
    assert.equal(observer.calls.length, 1);
  } finally {
    finish();
    observer.restore();
  }
});
