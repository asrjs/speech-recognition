import { describe, expect, it, vi } from 'vitest';
import {
  createLoadedSpeechModelHandle,
  createSpeechRuntime,
  type ExecutionBackend,
  type SpeechModel,
  type SpeechSession,
  type StreamingTranscriber,
} from '@asrjs/speech-recognition';
import { loadBuiltInSpeechModel } from '@asrjs/speech-recognition/builtins';
import {
  createActiveOperationLease,
  createTrackedStreamingTranscriberFactory,
} from '../src/runtime/operation-lease.js';

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function streaming(dispose: () => Promise<void>): StreamingTranscriber {
  return {
    pushAudio: vi.fn(),
    flush: vi.fn(),
    finalize: vi.fn(),
    reset: vi.fn(),
    dispose,
    getState: () => ({
      revision: 0,
      bufferedDurationSeconds: 0,
      committedText: '',
      previewText: '',
      isFinalized: false,
    }),
  };
}

describe('streaming owner failure cleanup', () => {
  it.each([
    ['loaded', false],
    ['built-in', false],
    ['loaded', true],
    ['built-in', true],
  ] as const)(
    '%s waits for every cleanup after failure (owner fails: %s)',
    async (kind, ownerFails) => {
      const runtime = createSpeechRuntime();
      const backend: ExecutionBackend = {
        id: 'cleanup-test',
        displayName: 'Cleanup test',
        probeCapabilities: async () => ({
          id: 'cleanup-test',
          displayName: 'Cleanup test',
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
        }),
        createExecutionContext: vi.fn(),
      };
      const failure = new Error('first transcriber release failed');
      const ownerFailure = new Error('owner release failed');
      const firstStarted = deferred(),
        secondStarted = deferred();
      const secondGate = deferred(),
        ownerStarted = deferred(),
        ownerGate = deferred();
      const firstDispose = vi.fn(async () => {
        firstStarted.resolve();
        throw failure;
      });
      const secondDispose = vi.fn(async () => {
        secondStarted.resolve();
        await secondGate.promise;
      });
      const ownerDispose = vi.fn(async () => {
        ownerStarted.resolve();
        await ownerGate.promise;
        if (ownerFails) throw ownerFailure;
      });
      const session: SpeechSession = { transcribe: vi.fn(), dispose: vi.fn() };
      const transcribers = [streaming(firstDispose), streaming(secondDispose)];
      const model: SpeechModel = {
        info: {
          family: 'cleanup-test',
          modelId: 'cleanup-model',
          classification: { ecosystem: 'test', task: 'asr' },
        },
        backend,
        createSession: async () => session,
        createStreamingTranscriber: async () => transcribers.shift()!,
        dispose: ownerDispose,
      };
      runtime.registerBackend(backend);
      runtime.registerModelFamily({
        family: 'cleanup-test',
        supports: () => true,
        createModel: async () => model,
      });
      const handle =
        kind === 'built-in'
          ? await loadBuiltInSpeechModel({
              runtime,
              family: 'cleanup-test',
              modelId: 'cleanup-model',
              backend: 'cleanup-test',
            })
          : createLoadedSpeechModelHandle({
              runtime,
              model,
              session,
              transcribe: session.transcribe,
              dispose: async () => {
                await model.dispose();
              },
            });
      const create = handle.createStreamingTranscriber;
      if (!create) throw new Error('Test handle must support streaming');
      await create.call(handle);
      await create.call(handle);
      let settled = false;
      const disposal = handle.dispose().then(
        () => {
          settled = true;
          return undefined;
        },
        (error: unknown) => {
          settled = true;
          return error;
        },
      );
      try {
        await Promise.all([firstStarted.promise, secondStarted.promise]);
        // Let the rejecting branch propagate without using wall-clock sleeps.
        for (let tick = 0; tick < 20; tick++) await Promise.resolve();
        expect(settled).toBe(false);
        expect(ownerDispose).not.toHaveBeenCalled();
        secondGate.resolve();
        await ownerStarted.promise;
        expect(settled).toBe(false);
        ownerGate.resolve();
        const result = await disposal;
        if (ownerFails) {
          expect(result).toBeInstanceOf(AggregateError);
          expect(result).toMatchObject({ errors: [failure, ownerFailure], cause: failure });
        } else {
          expect(result).toBe(failure);
        }
        await expect(handle.dispose()).rejects.toBe(result);
        expect(firstDispose).toHaveBeenCalledTimes(1);
        expect(secondDispose).toHaveBeenCalledTimes(1);
        expect(ownerDispose).toHaveBeenCalledTimes(1);
        await expect(create.call(handle)).rejects.toThrow('disposed');
      } finally {
        secondGate.resolve();
        ownerGate.resolve();
        await disposal;
      }
    },
  );

  it('collects every transcriber failure in creation order, including synchronous throws', async () => {
    const first = new Error('sync'),
      second = new Error('async');
    const firstDispose = vi.fn(() => {
      throw first;
    });
    const secondDispose = vi.fn(async () => {
      throw second;
    });
    const sources = [streaming(firstDispose), streaming(secondDispose)];
    const factory = createTrackedStreamingTranscriberFactory(
      createActiveOperationLease(),
      async () => sources.shift()!,
    );
    await factory.create();
    await factory.create();
    await expect(factory.disposeAll()).rejects.toMatchObject({
      errors: [first, second],
      cause: first,
    });
    await expect(factory.disposeAll()).rejects.toMatchObject({ errors: [first, second] });
    expect(firstDispose).toHaveBeenCalledTimes(1);
    expect(secondDispose).toHaveBeenCalledTimes(1);
  });
});
