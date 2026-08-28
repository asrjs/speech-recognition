import { afterEach, describe, expect, it, vi } from 'vitest';

type WorkerMessage = {
  readonly id: number;
  readonly type: string;
  readonly payload?: unknown;
};

type WorkerScope = {
  onmessage: ((event: MessageEvent<unknown>) => void) | null;
  postMessage: ReturnType<typeof vi.fn>;
};

const previousSelfDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'self');

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (predicate()) {
      return;
    }
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('Timed out waiting for the browser worker response.');
}

function send(scope: WorkerScope, message: WorkerMessage): void {
  scope.onmessage?.({ data: message } as MessageEvent<unknown>);
}

describe('browser transcription worker thread cancellation', () => {
  afterEach(() => {
    vi.doUnmock('../src/runtime/load.js');
    vi.doUnmock('../src/runtime/local-browser.js');
    vi.resetModules();
    if (previousSelfDescriptor) {
      Object.defineProperty(globalThis, 'self', previousSelfDescriptor);
    } else {
      Reflect.deleteProperty(globalThis, 'self');
    }
  });

  it('cancels a queued transcription before decode and keeps the loaded model reusable', async () => {
    let releaseLoad!: () => void;
    const loadGate = new Promise<void>((resolve) => {
      releaseLoad = resolve;
    });
    const transcribeMonoPcm = vi.fn(async () => ({ text: 'ok' }));
    const model = {
      model: { id: 'parakeet', info: { family: 'nemo-tdt' } },
      transcribeMonoPcm,
      dispose: vi.fn(async () => undefined),
    };
    const loadSpeechModel = vi.fn(async () => {
      await loadGate;
      return model;
    });

    vi.doMock('../src/runtime/load.js', () => ({ loadSpeechModel }));
    vi.doMock('../src/runtime/local-browser.js', () => ({
      loadSpeechModelFromLocalEntries: vi.fn(),
    }));

    const scope: WorkerScope = {
      onmessage: null,
      postMessage: vi.fn(),
    };
    Object.defineProperty(globalThis, 'self', {
      configurable: true,
      value: scope,
    });

    await import('../src/runtime/browser-transcription-worker-thread.js');

    send(scope, {
      id: 1,
      type: 'LOAD_BUILT_IN_MODEL',
      payload: { modelId: 'parakeet' },
    });
    send(scope, {
      id: 2,
      type: 'TRANSCRIBE_MONO_PCM',
      payload: { pcm: new Float32Array([0, 0.1]), sampleRate: 16000, options: null },
    });
    send(scope, {
      id: 3,
      type: 'CANCEL_TRANSCRIBE',
      payload: { requestId: 2 },
    });

    releaseLoad();
    await waitFor(() =>
      scope.postMessage.mock.calls.some(
        ([message]) => message?.id === 2 && message?.type === 'ERROR',
      ),
    );

    expect(transcribeMonoPcm).not.toHaveBeenCalled();
    expect(scope.postMessage.mock.calls).toContainEqual([
      expect.objectContaining({ id: 1, type: 'SUCCESS' }),
    ]);

    send(scope, {
      id: 4,
      type: 'TRANSCRIBE_MONO_PCM',
      payload: { pcm: new Float32Array([0, 0.1]), sampleRate: 16000, options: null },
    });
    await waitFor(() =>
      scope.postMessage.mock.calls.some(
        ([message]) => message?.id === 4 && message?.type === 'SUCCESS',
      ),
    );

    expect(transcribeMonoPcm).toHaveBeenCalledTimes(1);
    expect(scope.postMessage.mock.calls).toContainEqual([
      expect.objectContaining({ id: 4, type: 'SUCCESS', payload: { text: 'ok' } }),
    ]);
  });

  it('aborts an active model decode and reuses the loaded model', async () => {
    type WorkerTranscriptionOptions = {
      readonly signal?: AbortSignal;
    };

    const transcribeMonoPcm = vi.fn(
      async (_pcm: Float32Array, _sampleRate: number, options?: WorkerTranscriptionOptions) => {
        if (transcribeMonoPcm.mock.calls.length === 1) {
          const signal = options?.signal;
          await new Promise<never>((_resolve, reject) => {
            if (signal?.aborted) {
              reject(new Error('cooperative abort'));
              return;
            }
            signal?.addEventListener('abort', () => reject(new Error('cooperative abort')), {
              once: true,
            });
          });
        }
        return { text: 'ok' };
      },
    );
    const dispose = vi.fn(async () => undefined);
    const model = {
      model: { id: 'parakeet', info: { family: 'nemo-tdt' } },
      transcribeMonoPcm,
      dispose,
    };

    vi.doMock('../src/runtime/load.js', () => ({
      loadSpeechModel: vi.fn(async () => model),
    }));
    vi.doMock('../src/runtime/local-browser.js', () => ({
      loadSpeechModelFromLocalEntries: vi.fn(),
    }));

    const scope: WorkerScope = {
      onmessage: null,
      postMessage: vi.fn(),
    };
    Object.defineProperty(globalThis, 'self', {
      configurable: true,
      value: scope,
    });

    await import('../src/runtime/browser-transcription-worker-thread.js');

    send(scope, {
      id: 1,
      type: 'LOAD_BUILT_IN_MODEL',
      payload: { modelId: 'parakeet' },
    });
    await waitFor(() =>
      scope.postMessage.mock.calls.some(
        ([message]) => message?.id === 1 && message?.type === 'SUCCESS',
      ),
    );

    send(scope, {
      id: 2,
      type: 'TRANSCRIBE_MONO_PCM',
      payload: { pcm: new Float32Array([0, 0.1]), sampleRate: 16000, options: null },
    });
    await waitFor(() => transcribeMonoPcm.mock.calls.length === 1);
    const signal = transcribeMonoPcm.mock.calls[0]?.[2]?.signal;
    expect(signal).toBeInstanceOf(AbortSignal);
    expect(signal?.aborted).toBe(false);

    send(scope, {
      id: 3,
      type: 'CANCEL_TRANSCRIBE',
      payload: { requestId: 2 },
    });
    await waitFor(() =>
      scope.postMessage.mock.calls.some(
        ([message]) => message?.id === 2 && message?.type === 'ERROR',
      ),
    );

    expect(signal?.aborted).toBe(true);
    expect(dispose).not.toHaveBeenCalled();

    send(scope, {
      id: 4,
      type: 'TRANSCRIBE_MONO_PCM',
      payload: { pcm: new Float32Array([0, 0.1]), sampleRate: 16000, options: null },
    });
    await waitFor(() =>
      scope.postMessage.mock.calls.some(
        ([message]) => message?.id === 4 && message?.type === 'SUCCESS',
      ),
    );

    expect(transcribeMonoPcm).toHaveBeenCalledTimes(2);
    expect(scope.postMessage.mock.calls).toContainEqual([
      expect.objectContaining({ id: 4, type: 'SUCCESS', payload: { text: 'ok' } }),
    ]);

    send(scope, { id: 5, type: 'DISPOSE_MODEL', payload: null });
    await waitFor(() =>
      scope.postMessage.mock.calls.some(
        ([message]) => message?.id === 5 && message?.type === 'SUCCESS',
      ),
    );
    expect(dispose).toHaveBeenCalledOnce();
  });
});
