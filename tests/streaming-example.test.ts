import { beforeEach, expect, it, vi } from 'vitest';
import { streamingExample } from '../examples/streaming.js';

const mocks = vi.hoisted(() => ({ runtime: vi.fn(), streaming: vi.fn() }));
vi.mock('@asrjs/speech-recognition', () => ({
  createSpeechRuntime: mocks.runtime,
  createWasmBackend: vi.fn(() => ({})),
}));
vi.mock('@asrjs/speech-recognition/inference', () => ({
  DefaultStreamingTranscriber: class {
    constructor(...args: unknown[]) {
      return mocks.streaming(...args);
    }
  },
}));
vi.mock('@asrjs/speech-recognition/models/nemo-tdt', () => ({
  createNemoTdtModelFamily: vi.fn(() => ({})),
}));
vi.mock('@asrjs/speech-recognition/presets/parakeet', () => ({
  createParakeetPresetFactory: vi.fn(() => ({})),
}));
beforeEach(() => vi.resetAllMocks());

// Consumer lifecycle simulation, not engine-release or recognition evidence.
function fixture() {
  const order: string[] = [];
  const session = {};
  const model = { createSession: vi.fn(async () => session), dispose: vi.fn() };
  const runtime = {
    loadModel: vi.fn(async () => model),
    dispose: vi.fn(async () => {
      order.push('runtime');
    }),
  };
  const partial = { text: 'simulation', kind: 'partial' };
  const final = { text: 'simulation', kind: 'final' };
  const streaming = {
    pushAudio: vi.fn(async () => partial),
    finalize: vi.fn(async () => final),
    dispose: vi.fn(async () => {
      order.push('streaming');
    }),
  };
  mocks.runtime.mockReturnValue(runtime);
  mocks.streaming.mockReturnValue(streaming);
  return { order, model, session, runtime, partial, final, streaming };
}

it('returns both updates after wrapper and runtime-owner cleanup, without explicit model disposal', async () => {
  const f = fixture();
  expect(await streamingExample()).toEqual({ partial: f.partial, final: f.final });
  expect(mocks.streaming).toHaveBeenCalledWith(f.session, {
    detail: 'words',
    overlapMs: 500,
    maxWindowMs: 5000,
  });
  expect(f.streaming.pushAudio).toHaveBeenCalledWith(new Float32Array(8000));
  expect(f.order).toEqual(['streaming', 'runtime']);
  expect(f.model.dispose).not.toHaveBeenCalled();
});

it.each(['load', 'session', 'constructor', 'push', 'finalize'] as const)(
  'cleans acquisitions after %s failure and preserves the original error',
  async (stage) => {
    const f = fixture();
    const error = new Error(`${stage} failure`);
    if (stage === 'load') f.runtime.loadModel.mockRejectedValue(error);
    if (stage === 'session') f.model.createSession.mockRejectedValue(error);
    if (stage === 'constructor')
      mocks.streaming.mockImplementation(() => {
        throw error;
      });
    if (stage === 'push') f.streaming.pushAudio.mockRejectedValue(error);
    if (stage === 'finalize') f.streaming.finalize.mockRejectedValue(error);
    await expect(streamingExample()).rejects.toBe(error);
    expect(f.runtime.dispose).toHaveBeenCalledOnce();
    expect(f.streaming.dispose).toHaveBeenCalledTimes(
      stage === 'push' || stage === 'finalize' ? 1 : 0,
    );
  },
);

it.each([new Error('primary'), undefined, null])(
  'preserves primary %s and both cleanup failures',
  async (primary) => {
    const f = fixture();
    const wrapper = new Error('wrapper cleanup');
    const owner = new AggregateError([new Error('owned session')], 'runtime cleanup');
    f.streaming.pushAudio.mockRejectedValue(primary);
    f.streaming.dispose.mockRejectedValue(wrapper);
    f.runtime.dispose.mockRejectedValue(owner);
    const error = await streamingExample().catch((failure: unknown) => failure);
    expect(error).toBeInstanceOf(AggregateError);
    expect((error as AggregateError).errors).toEqual([primary, wrapper, owner]);
    expect((error as AggregateError).cause).toBe(primary);
    expect(f.runtime.dispose).toHaveBeenCalledOnce();
  },
);

it('fails instead of returning success if wrapper cleanup fails, still releasing the runtime', async () => {
  const f = fixture();
  const cleanup = new Error('wrapper cleanup');
  f.streaming.dispose.mockRejectedValue(cleanup);
  const error = await streamingExample().catch((failure: unknown) => failure);
  expect((error as AggregateError).errors).toEqual([cleanup]);
  expect((error as AggregateError).cause).toBe(cleanup);
  expect(f.runtime.dispose).toHaveBeenCalledOnce();
});

it('awaits wrapper cleanup before runtime cleanup and result settlement', async () => {
  const f = fixture();
  const started = Promise.withResolvers<void>();
  const pending = Promise.withResolvers<void>();
  f.streaming.dispose.mockImplementation(() => {
    started.resolve();
    return pending.promise;
  });
  let settled = false;
  const run = streamingExample().then((value) => {
    settled = true;
    return value;
  });
  await started.promise;
  expect(settled).toBe(false);
  expect(f.runtime.dispose).not.toHaveBeenCalled();
  pending.resolve();
  await run;
  expect(f.runtime.dispose).toHaveBeenCalledOnce();
});

it('preserves a runtime-only cleanup failure instead of returning success', async () => {
  const f = fixture();
  const cleanup = new Error('runtime cleanup');
  f.runtime.dispose.mockRejectedValue(cleanup);
  const error = await streamingExample().catch((failure: unknown) => failure);
  expect((error as AggregateError).errors).toEqual([cleanup]);
  expect((error as AggregateError).cause).toBe(cleanup);
  expect(f.streaming.dispose).toHaveBeenCalledOnce();
});

it('does not dispose while a streaming operation is pending', async () => {
  const f = fixture();
  const started = Promise.withResolvers<void>();
  const pending = Promise.withResolvers<typeof f.final>();
  f.streaming.finalize.mockImplementation(() => {
    started.resolve();
    return pending.promise;
  });
  const run = streamingExample();
  await started.promise;
  expect(f.streaming.dispose).not.toHaveBeenCalled();
  expect(f.runtime.dispose).not.toHaveBeenCalled();
  pending.resolve(f.final);
  expect(await run).toEqual({ partial: f.partial, final: f.final });
});

it('does not settle until runtime cleanup finishes', async () => {
  const f = fixture();
  const started = Promise.withResolvers<void>();
  const pending = Promise.withResolvers<void>();
  f.runtime.dispose.mockImplementation(() => {
    started.resolve();
    return pending.promise;
  });
  let settled = false;
  const run = streamingExample().then((value) => {
    settled = true;
    return value;
  });
  await started.promise;
  expect(settled).toBe(false);
  expect(f.streaming.dispose).toHaveBeenCalledOnce();
  pending.resolve();
  await run;
  expect(settled).toBe(true);
});
