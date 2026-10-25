import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createSpeechController } from '../src/runtime/speech-controller.js';

const { loadSpeechModel, decodeAudioSourceToMonoPcm } = vi.hoisted(() => ({
  loadSpeechModel: vi.fn(),
  decodeAudioSourceToMonoPcm: vi.fn(),
}));
vi.mock('../src/runtime/load.js', () => ({ loadSpeechModel }));
vi.mock('../src/runtime/media.js', () => ({ decodeAudioSourceToMonoPcm }));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

function fakeModel(overrides: Record<string, unknown> = {}) {
  return {
    model: { info: { modelId: 'test-model' } },
    supportsStreaming: true,
    transcribeMonoPcm: vi.fn(async () => ({ text: 'done', warnings: [], meta: {} })),
    createStreamingTranscriber: vi.fn(async () => ({
      pushAudio: vi.fn(async () => ({
        kind: 'partial',
        revision: 1,
        text: 'a',
        committedText: '',
        previewText: 'a',
        warnings: [],
        meta: {},
      })),
      finalize: vi.fn(async () => ({
        kind: 'final',
        revision: 2,
        text: 'ab',
        committedText: 'ab',
        previewText: '',
        warnings: [],
        meta: {},
      })),
      dispose: vi.fn(async () => undefined),
    })),
    dispose: vi.fn(async () => undefined),
    ...overrides,
  };
}

describe('createSpeechController', () => {
  beforeEach(() => {
    loadSpeechModel.mockReset();
    decodeAudioSourceToMonoPcm.mockReset();
  });

  it('decodes a file at the loaded model rate and keeps canonical transcription in the controller', async () => {
    const model = fakeModel({
      model: { info: { modelId: '8khz-model', inference: { sampleRate: 8000 } } },
    });
    const pcm = new Float32Array([0.25]);
    decodeAudioSourceToMonoPcm.mockResolvedValue({ pcm, sampleRate: 8000 });
    loadSpeechModel.mockResolvedValue(model);
    const controller = createSpeechController();
    await controller.load({ modelId: '8khz-model' });
    const file = new Blob(['audio']);
    await controller.transcribeSource(file, { language: 'en', responseFlavor: 'native' });
    expect(decodeAudioSourceToMonoPcm).toHaveBeenCalledWith(
      file,
      expect.objectContaining({ targetSampleRate: 8000 }),
    );
    expect(model.transcribeMonoPcm).toHaveBeenCalledWith(
      pcm,
      8000,
      expect.objectContaining({ language: 'en', responseFlavor: 'canonical' }),
    );
    expect(controller.getSnapshot()).toMatchObject({ status: 'ready', result: { text: 'done' } });
    await controller.dispose();
  });

  it('owns decoding as an active operation and waits for it during disposal', async () => {
    const model = fakeModel();
    const decoding = deferred<{ pcm: Float32Array; sampleRate: number }>();
    loadSpeechModel.mockResolvedValue(model);
    decodeAudioSourceToMonoPcm.mockReturnValue(decoding.promise);
    const controller = createSpeechController();
    await controller.load({ modelId: 'test-model' });
    const transcription = controller.transcribeSource('audio.wav').catch((error: unknown) => error);
    await vi.waitFor(() => expect(decodeAudioSourceToMonoPcm).toHaveBeenCalledOnce());
    expect(controller.getSnapshot().status).toBe('transcribing');
    await expect(controller.load({ modelId: 'other' })).rejects.toThrow('while transcribing');
    await expect(controller.transcribeSource('other.wav')).rejects.toThrow(
      'Another controller operation',
    );
    const disposal = controller.dispose();
    expect(model.dispose).not.toHaveBeenCalled();
    decoding.resolve({ pcm: new Float32Array([0]), sampleRate: 16000 });
    expect(await transcription).toMatchObject({ name: 'AbortError' });
    await disposal;
    expect(model.transcribeMonoPcm).not.toHaveBeenCalled();
    expect(model.dispose).toHaveBeenCalledOnce();
    expect(controller.getSnapshot().status).toBe('disposed');
  });

  it('rejects a pre-aborted file operation without decoding or inference', async () => {
    const model = fakeModel();
    loadSpeechModel.mockResolvedValue(model);
    const controller = createSpeechController();
    await controller.load({ modelId: 'test-model' });
    const abort = new AbortController();
    abort.abort();
    await expect(
      controller.transcribeSource('audio.wav', { signal: abort.signal }),
    ).rejects.toMatchObject({ name: 'AbortError' });
    expect(decodeAudioSourceToMonoPcm).not.toHaveBeenCalled();
    expect(model.transcribeMonoPcm).not.toHaveBeenCalled();
    expect(controller.getSnapshot().status).toBe('error');
    await controller.dispose();
  });

  it('retains load warnings after later progress and clears them on the next load', async () => {
    const warning = {
      code: 'precision-fallback',
      message: 'FP16 failed; trying FP32.',
      recoverable: true,
    };
    loadSpeechModel
      .mockImplementationOnce(async (options) => {
        options.onProgress({ phase: 'model-load:fallback', warning });
        options.onProgress({ phase: 'download', message: 'Downloading FP32 encoder.' });
        return fakeModel();
      })
      .mockResolvedValue(fakeModel());
    const controller = createSpeechController();
    await controller.load({ modelId: 'first' });
    const first = controller.getSnapshot();
    expect(first.status).toBe('ready');
    expect(first.progress?.phase).toBe('download');
    expect(first.loadWarnings).toEqual([warning]);
    expect(Object.isFrozen(first.loadWarnings)).toBe(true);
    await controller.load({ modelId: 'second' });
    expect(controller.getSnapshot().loadWarnings).toEqual([]);
    expect(first.loadWarnings).toEqual([warning]);
    await controller.dispose();
  });

  it('ignores warnings from a superseded load', async () => {
    const pending = deferred<ReturnType<typeof fakeModel>>();
    let staleProgress!: (event: {
      phase: string;
      warning: { code: string; message: string };
    }) => void;
    loadSpeechModel
      .mockImplementationOnce((options) => {
        staleProgress = options.onProgress;
        return pending.promise;
      })
      .mockResolvedValue(fakeModel());
    const controller = createSpeechController();
    const first = controller.load({ modelId: 'old' });
    await vi.waitFor(() => expect(loadSpeechModel).toHaveBeenCalledOnce());
    await controller.load({ modelId: 'new' });
    staleProgress({
      phase: 'model-load:fallback',
      warning: { code: 'stale', message: 'Old load.' },
    });
    pending.resolve(fakeModel());
    await first;
    expect(controller.getSnapshot().loadWarnings).toEqual([]);
    await controller.dispose();
  });

  it('keeps snapshot identity until a change and does not notify on subscribe', async () => {
    const model = fakeModel();
    loadSpeechModel.mockResolvedValue(model);
    const controller = createSpeechController();
    const listener = vi.fn();
    const first = controller.getSnapshot();
    const unsubscribe = controller.subscribe(listener);
    expect(controller.getSnapshot()).toBe(first);
    expect(listener).not.toHaveBeenCalled();
    await controller.load({ modelId: 'test-model' });
    expect(controller.getSnapshot()).not.toBe(first);
    expect(listener).toHaveBeenCalledTimes(2);
    unsubscribe();
    await controller.dispose();
  });

  it('cancels a stale load and disposes its late result without publishing it', async () => {
    const pending = deferred<ReturnType<typeof fakeModel>>();
    const old = fakeModel();
    const current = fakeModel({ model: { info: { modelId: 'new-model' } } });
    loadSpeechModel.mockImplementationOnce(() => pending.promise).mockResolvedValueOnce(current);
    const controller = createSpeechController();
    const first = controller.load({ modelId: 'old-model' });
    await vi.waitFor(() => expect(loadSpeechModel).toHaveBeenCalledTimes(1));
    const second = controller.load({ modelId: 'new-model' });
    pending.resolve(old);
    await Promise.all([first, second]);
    expect(old.dispose).toHaveBeenCalledOnce();
    expect(controller.getSnapshot().modelId).toBe('new-model');
    expect(controller.getSnapshot().status).toBe('ready');
    await controller.dispose();
  });

  it('preserves a stale candidate cleanup failure without disposing it twice', async () => {
    const pending = deferred<ReturnType<typeof fakeModel>>();
    const cleanupFailure = new Error('candidate cleanup failed');
    const stale = fakeModel({
      dispose: vi.fn(async () => {
        throw cleanupFailure;
      }),
    });
    const current = fakeModel({ model: { info: { modelId: 'new-model' } } });
    loadSpeechModel.mockReturnValueOnce(pending.promise).mockResolvedValueOnce(current);
    const controller = createSpeechController();
    const staleLoad = controller.load({ modelId: 'old-model' });
    await vi.waitFor(() => expect(loadSpeechModel).toHaveBeenCalledOnce());
    const currentLoad = controller.load({ modelId: 'new-model' });
    pending.resolve(stale);
    await expect(staleLoad).rejects.toMatchObject({
      name: 'AggregateError',
      errors: [expect.objectContaining({ name: 'AbortError' }), cleanupFailure],
    });
    await currentLoad;
    expect(stale.dispose).toHaveBeenCalledOnce();
    expect(controller.getSnapshot().modelId).toBe('new-model');
    await controller.dispose();
  });

  it('cancels a replacement load aborted while the previous model is disposing', async () => {
    const disposingPrevious = deferred<void>();
    const previous = fakeModel({
      model: { info: { modelId: 'initial' } },
      dispose: vi.fn(() => disposingPrevious.promise),
    });
    const candidate = fakeModel({ model: { info: { modelId: 'replacement' } } });
    loadSpeechModel.mockResolvedValueOnce(previous).mockResolvedValueOnce(candidate);
    const controller = createSpeechController();
    await controller.load({ modelId: 'initial' });
    const abort = new AbortController();
    const replacement = controller.load({ modelId: 'replacement', signal: abort.signal });
    await vi.waitFor(() => expect(previous.dispose).toHaveBeenCalledOnce());
    abort.abort();
    disposingPrevious.resolve();
    await expect(replacement).rejects.toMatchObject({ name: 'AbortError' });
    expect(candidate.dispose).toHaveBeenCalledOnce();
    expect(controller.getSnapshot().modelId).toBeNull();
    await controller.dispose();
  });

  it('drains inference before disposing and suppresses late result state', async () => {
    const pending = deferred<{ text: string; warnings: []; meta: object }>();
    const model = fakeModel({ transcribeMonoPcm: vi.fn(() => pending.promise) });
    loadSpeechModel.mockResolvedValue(model);
    const controller = createSpeechController();
    await controller.load({ modelId: 'test-model' });
    const work = controller.transcribe(new Float32Array([0]), 16000);
    const disposing = controller.dispose();
    expect(model.dispose).not.toHaveBeenCalled();
    pending.resolve({ text: 'late', warnings: [], meta: {} });
    await expect(work).rejects.toMatchObject({ name: 'AbortError' });
    await disposing;
    expect(controller.getSnapshot().status).toBe('disposed');
    expect(controller.getSnapshot().result).toBeNull();
    expect(model.dispose).toHaveBeenCalledOnce();
  });

  it('resets operation state after a synchronous transcription throw', async () => {
    const failure = new Error('synchronous model failure');
    const transcribe = vi
      .fn(async () => ({ text: 'done', warnings: [], meta: {} }))
      .mockImplementationOnce(() => {
        throw failure;
      });
    const model = fakeModel({ transcribeMonoPcm: transcribe });
    loadSpeechModel.mockResolvedValue(model);
    const controller = createSpeechController();
    await controller.load({ modelId: 'test-model' });
    await expect(controller.transcribe(new Float32Array([0]), 16000)).rejects.toBe(failure);
    await expect(controller.transcribe(new Float32Array([0]), 16000)).resolves.toMatchObject({
      text: 'done',
    });
    await controller.dispose();
  });

  it('drains an unresolved load on dispose and disposes its late model', async () => {
    const pending = deferred<ReturnType<typeof fakeModel>>();
    const lateModel = fakeModel();
    loadSpeechModel.mockReturnValue(pending.promise);
    const controller = createSpeechController();
    const loading = controller.load({ modelId: 'test-model' });
    await vi.waitFor(() => expect(loadSpeechModel).toHaveBeenCalledOnce());
    const disposing = controller.dispose();
    pending.resolve(lateModel);
    await Promise.all([loading, disposing]);
    expect(lateModel.dispose).toHaveBeenCalledOnce();
    expect(controller.getSnapshot().status).toBe('disposed');
  });

  it('retains cleanup errors and makes dispose idempotent', async () => {
    const failure = new Error('cleanup failed');
    const model = fakeModel({
      dispose: vi.fn(async () => {
        throw failure;
      }),
    });
    loadSpeechModel.mockResolvedValue(model);
    const controller = createSpeechController();
    await controller.load({ modelId: 'test-model' });
    const first = controller.dispose();
    const second = controller.dispose();
    await expect(Promise.all([first, second])).rejects.toBe(failure);
  });

  it('serializes streaming chunks and finalization after queued chunks', async () => {
    const firstChunk = deferred<void>();
    const calls: number[] = [];
    const transcriber = {
      pushAudio: vi.fn(async (input: { sampleRate: number }) => {
        calls.push(input.sampleRate);
        if (calls.length === 1) await firstChunk.promise;
        return {
          kind: 'partial' as const,
          revision: calls.length,
          text: '',
          committedText: '',
          previewText: '',
          warnings: [],
          meta: {},
        };
      }),
      finalize: vi.fn(async () => ({
        kind: 'final' as const,
        revision: 3,
        text: 'ok',
        committedText: 'ok',
        previewText: '',
        warnings: [],
        meta: {},
      })),
      dispose: vi.fn(async () => undefined),
    };
    loadSpeechModel.mockResolvedValue(
      fakeModel({ createStreamingTranscriber: vi.fn(async () => transcriber) }),
    );
    const controller = createSpeechController();
    await controller.load({ modelId: 'test-model' });
    await controller.startStreaming();
    const one = controller.pushAudio(new Float32Array([0]), 16000);
    const two = controller.pushAudio(new Float32Array([0]), 22050);
    const final = controller.finalizeStreaming();
    await Promise.resolve();
    expect(calls).toEqual([16000]);
    firstChunk.resolve(undefined);
    await Promise.all([one, two]);
    expect((await final).text).toBe('ok');
    expect(calls).toEqual([16000, 22050]);
    expect(transcriber.finalize).toHaveBeenCalledOnce();
    await controller.dispose();
  });

  it('blocks new chunks and duplicate finalization as soon as finalization starts', async () => {
    const finalResult = deferred<{
      kind: 'final';
      revision: number;
      text: string;
      committedText: string;
      previewText: string;
      warnings: [];
      meta: object;
    }>();
    const transcriber = {
      pushAudio: vi.fn(async () => ({
        kind: 'partial' as const,
        revision: 1,
        text: '',
        committedText: '',
        previewText: '',
        warnings: [],
        meta: {},
      })),
      finalize: vi.fn(() => finalResult.promise),
      dispose: vi.fn(async () => undefined),
    };
    loadSpeechModel.mockResolvedValue(
      fakeModel({ createStreamingTranscriber: vi.fn(async () => transcriber) }),
    );
    const controller = createSpeechController();
    await controller.load({ modelId: 'test-model' });
    await controller.startStreaming();
    const finalizing = controller.finalizeStreaming();
    await vi.waitFor(() => expect(transcriber.finalize).toHaveBeenCalledOnce());
    await expect(controller.pushAudio(new Float32Array([0]), 16000)).rejects.toThrow(
      /Streaming has not been started/,
    );
    await expect(controller.finalizeStreaming()).rejects.toThrow(/Streaming has not been started/);
    finalResult.resolve({
      kind: 'final',
      revision: 2,
      text: '',
      committedText: '',
      previewText: '',
      warnings: [],
      meta: {},
    });
    await finalizing;
    await controller.dispose();
  });

  it('awaits raced stream candidate cleanup before disposing the model', async () => {
    const creation = deferred<{
      pushAudio: () => Promise<never>;
      finalize: () => Promise<never>;
      dispose: () => Promise<void>;
      getState?: () => never;
    }>();
    const candidateDisposal = deferred<void>();
    const order: string[] = [];
    const candidate = {
      pushAudio: vi.fn(async () => {
        throw new Error('unused');
      }),
      finalize: vi.fn(async () => {
        throw new Error('unused');
      }),
      dispose: vi.fn(async () => {
        order.push('candidate:start');
        await candidateDisposal.promise;
        order.push('candidate:done');
      }),
    };
    const model = fakeModel({
      createStreamingTranscriber: vi.fn(() => creation.promise),
      dispose: vi.fn(async () => {
        order.push('model');
      }),
    });
    loadSpeechModel.mockResolvedValue(model);
    const controller = createSpeechController();
    await controller.load({ modelId: 'test-model' });
    const starting = controller.startStreaming();
    const startOutcome = expect(starting).rejects.toMatchObject({ name: 'AbortError' });
    await vi.waitFor(() => expect(model.createStreamingTranscriber).toHaveBeenCalledOnce());
    const disposing = controller.dispose();
    creation.resolve(candidate);
    await vi.waitFor(() => expect(candidate.dispose).toHaveBeenCalledOnce());
    expect(order).toEqual(['candidate:start']);
    let disposeSettled = false;
    void disposing.finally(() => {
      disposeSettled = true;
    });
    await Promise.resolve();
    expect(disposeSettled).toBe(false);
    candidateDisposal.resolve();
    await startOutcome;
    await disposing;
    expect(order).toEqual(['candidate:start', 'candidate:done', 'model']);
  });

  it('retains both a finalization failure and a cleanup failure, including thrown undefined', async () => {
    const cleanupFailure = new Error('stream cleanup failed');
    const transcriber = {
      pushAudio: vi.fn(async () => ({
        kind: 'partial' as const,
        revision: 1,
        text: '',
        committedText: '',
        previewText: '',
        warnings: [],
        meta: {},
      })),
      finalize: vi.fn(async () => {
        throw undefined;
      }),
      dispose: vi.fn(async () => {
        throw cleanupFailure;
      }),
    };
    loadSpeechModel.mockResolvedValue(
      fakeModel({ createStreamingTranscriber: vi.fn(async () => transcriber) }),
    );
    const controller = createSpeechController();
    await controller.load({ modelId: 'test-model' });
    await controller.startStreaming();
    await expect(controller.finalizeStreaming()).rejects.toMatchObject({
      name: 'AggregateError',
      errors: [undefined, cleanupFailure],
    });
    await controller.dispose();
  });

  it('reports unsupported streaming instead of claiming readiness', async () => {
    loadSpeechModel.mockResolvedValue(fakeModel({ supportsStreaming: false }));
    const controller = createSpeechController();
    await controller.load({ modelId: 'test-model' });
    await expect(controller.startStreaming()).rejects.toThrow(/does not support streaming/);
    expect(controller.getSnapshot().status).toBe('error');
    expect(controller.getSnapshot().error).toBeInstanceOf(Error);
    await controller.dispose();
  });

  it('forwards caller abort signals and progress callbacks', async () => {
    const model = fakeModel();
    const progress = vi.fn();
    const controller = createSpeechController();
    const abort = new AbortController();
    const pending = deferred<ReturnType<typeof fakeModel>>();
    loadSpeechModel.mockReturnValue(pending.promise);
    const loading = controller.load({
      modelId: 'test-model',
      signal: abort.signal,
      onProgress: progress,
    });
    await vi.waitFor(() => expect(loadSpeechModel).toHaveBeenCalledOnce());
    const options = loadSpeechModel.mock.calls[0]![0] as {
      signal: AbortSignal;
      onProgress: (value: { phase: string }) => void;
    };
    expect(options.signal).toBeInstanceOf(AbortSignal);
    expect(options.signal.aborted).toBe(false);
    options.onProgress({ phase: 'fetch' });
    expect(progress).toHaveBeenCalledWith({ phase: 'fetch' });
    expect(controller.getSnapshot().progress).toEqual({ phase: 'fetch' });
    abort.abort();
    expect(options.signal.aborted).toBe(true);
    pending.resolve(model);
    await expect(loading).rejects.toMatchObject({ name: 'AbortError' });
    expect(model.dispose).toHaveBeenCalledOnce();
    await controller.dispose();
  });

  it('forces canonical PCM output and forwards caller plus disposal cancellation to inference', async () => {
    const model = fakeModel();
    loadSpeechModel.mockResolvedValue(model);
    const controller = createSpeechController();
    await controller.load({ modelId: 'test-model' });
    const callerAbort = new AbortController();
    await controller.transcribe(new Float32Array([0]), 16000, {
      responseFlavor: 'native',
      signal: callerAbort.signal,
    });
    const passed = model.transcribeMonoPcm.mock.calls[0]![2] as {
      responseFlavor: string;
      signal: { readonly aborted: boolean };
    };
    expect(passed.responseFlavor).toBe('canonical');
    expect(passed.signal.aborted).toBe(false);
    callerAbort.abort();
    expect(passed.signal.aborted).toBe(true);
    await controller.dispose();
  });
});
