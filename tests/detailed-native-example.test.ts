import { beforeEach, expect, it, vi } from 'vitest';
import { detailedAndNativeExample } from '../examples/detailed-and-native.js';
import { DefaultSpeechRuntime } from '../src/runtime/session.js';
import type { SpeechModel } from '../src/types/runtime.js';

const { createRuntime } = vi.hoisted(() => ({ createRuntime: vi.fn() }));
vi.mock('@asrjs/speech-recognition', () => ({
  createSpeechRuntime: createRuntime,
  createWasmBackend: vi.fn(() => ({})),
}));
vi.mock('@asrjs/speech-recognition/models/nemo-tdt', () => ({
  createNemoTdtModelFamily: vi.fn(() => ({})),
}));
vi.mock('@asrjs/speech-recognition/presets/parakeet', () => ({
  createParakeetPresetFactory: vi.fn(() => ({})),
}));

beforeEach(() => createRuntime.mockReset());

function fixture() {
  const order: string[] = [];
  const response = { canonical: { text: 'simulation only' }, native: { tokenIds: [1] } };
  const dispose = (name: string) =>
    vi.fn(async () => {
      await Promise.resolve();
      order.push(name);
    });
  const session = { transcribe: vi.fn(async () => response), dispose: dispose('session') };
  let sessionOwned = false;
  const modelDispose = vi.fn(async () => {
    try {
      if (sessionOwned) await session.dispose();
    } finally {
      order.push('model');
    }
  });
  const model = {
    createSession: vi.fn(async () => {
      sessionOwned = true;
      return session;
    }),
    dispose: modelDispose,
  };
  // Real runtime ownership; only model inference/session resources are simulated.
  const runtime = new DefaultSpeechRuntime({
    backends: [
      {
        id: 'wasm',
        displayName: 'simulation',
        async probeCapabilities() {
          return {
            id: 'wasm',
            displayName: 'simulation',
            available: true,
            priority: 1,
            environments: ['node'],
            acceleration: ['cpu'],
            supportedPrecisions: ['fp32'],
            supportsFp16: false,
            supportsInt8: false,
            supportsSharedArrayBuffer: false,
            requiresSharedArrayBuffer: false,
            fallbackSuitable: true,
            notes: [],
          };
        },
        async createExecutionContext() {
          throw new Error('No actual inference allowed');
        },
      },
    ],
    modelFamilies: [
      {
        family: 'nemo-tdt',
        supports: () => true,
        async createModel() {
          return model as unknown as SpeechModel;
        },
      },
    ],
    presets: [
      {
        preset: 'parakeet',
        supports: () => true,
        async resolveModelRequest(request) {
          return { family: 'nemo-tdt', modelId: request.modelId! };
        },
      },
    ],
  });
  const originalDispose = runtime.dispose.bind(runtime);
  vi.spyOn(runtime, 'dispose').mockImplementation(async () => {
    try {
      await originalDispose();
    } finally {
      order.push('runtime');
    }
  });
  const loadModel = vi.spyOn(runtime, 'loadModel');
  createRuntime.mockReturnValue(runtime);
  return { order, response, session, model, modelDispose, runtime, loadModel };
}

it('returns the unchanged detailed envelope only after owned cleanup is awaited', async () => {
  const f = fixture();
  expect(await detailedAndNativeExample()).toBe(f.response);
  expect(f.order).toEqual(['session', 'model', 'runtime']);
  expect(f.session.transcribe).toHaveBeenCalledWith(new Float32Array(32000), {
    detail: 'detailed',
    responseFlavor: 'canonical+native',
    returnFrameIndices: true,
    returnLogProbs: true,
    returnTdtSteps: true,
  });
});

it('does not dispose a session while its transcription is pending', async () => {
  const f = fixture();
  const pending = Promise.withResolvers<typeof f.response>();
  const started = Promise.withResolvers<void>();
  f.session.transcribe.mockImplementation(() => {
    started.resolve();
    return pending.promise;
  });
  const run = detailedAndNativeExample();
  await started.promise;
  expect(f.order).toEqual([]);
  pending.resolve(f.response);
  await run;
  expect(f.order).toEqual(['session', 'model', 'runtime']);
});

it('does not resolve before asynchronous owner cleanup finishes', async () => {
  const f = fixture();
  const pending = Promise.withResolvers<void>();
  const started = Promise.withResolvers<void>();
  f.modelDispose.mockImplementation(async () => {
    started.resolve();
    await pending.promise;
    await f.session.dispose();
  });
  let resolved = false;
  const run = detailedAndNativeExample().then((response) => {
    resolved = true;
    return response;
  });
  await started.promise;
  expect(resolved).toBe(false);
  expect(f.session.dispose).not.toHaveBeenCalled();
  pending.resolve();
  expect(await run).toBe(f.response);
  expect(f.runtime.dispose).toHaveBeenCalledOnce();
});

it.each(['load', 'session', 'transcribe'] as const)(
  'cleans acquired resources after %s failure',
  async (stage) => {
    const f = fixture();
    const error = new Error(`${stage} failed`);
    if (stage === 'load') f.loadModel.mockRejectedValue(error);
    if (stage === 'session') f.model.createSession.mockRejectedValue(error);
    if (stage === 'transcribe') f.session.transcribe.mockRejectedValue(error);
    await expect(detailedAndNativeExample()).rejects.toBe(error);
    expect(f.order).toEqual(
      stage === 'load'
        ? ['runtime']
        : stage === 'session'
          ? ['model', 'runtime']
          : ['session', 'model', 'runtime'],
    );
  },
);

it('preserves primary and nested owner cleanup errors without flattening them', async () => {
  const f = fixture();
  const primary = new Error('transcription failed');
  const cleanup = [
    new Error('session failed'),
    new Error('model failed'),
    new Error('runtime failed'),
  ];
  f.session.transcribe.mockRejectedValue(primary);
  const ownerError = new AggregateError(cleanup, 'owner cleanup failed');
  f.modelDispose.mockRejectedValue(ownerError);
  const error = await detailedAndNativeExample().catch((failure: unknown) => failure);
  expect(error).toBeInstanceOf(AggregateError);
  expect((error as AggregateError).errors).toEqual([primary, ownerError]);
  expect(ownerError.errors).toEqual(cleanup);
  expect((error as AggregateError).cause).toBe(primary);
  expect(f.modelDispose).toHaveBeenCalledOnce();
  expect(f.runtime.dispose).toHaveBeenCalledOnce();
});

it('does not report success when cleanup alone fails', async () => {
  const f = fixture();
  const cleanup = new Error('session cleanup failed');
  f.session.dispose.mockRejectedValue(cleanup);
  const error = await detailedAndNativeExample().catch((failure: unknown) => failure);
  expect(error).toBeInstanceOf(AggregateError);
  expect((error as AggregateError).errors).toEqual([cleanup]);
  expect(f.modelDispose).toHaveBeenCalledOnce();
  expect(f.runtime.dispose).toHaveBeenCalledOnce();
});

it.each([undefined, null])(
  'preserves a falsy primary throwable (%s) alongside cleanup failure',
  async (primary) => {
    const f = fixture();
    const cleanup = new Error('cleanup failed');
    f.session.transcribe.mockRejectedValue(primary);
    f.session.dispose.mockRejectedValue(cleanup);
    const error = await detailedAndNativeExample().catch((failure: unknown) => failure);
    expect(error).toBeInstanceOf(AggregateError);
    expect((error as AggregateError).errors).toEqual([primary, cleanup]);
    expect((error as AggregateError).cause).toBe(primary);
  },
);
