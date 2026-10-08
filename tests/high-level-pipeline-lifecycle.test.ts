import {
  createSpeechPipeline,
  createSpeechRuntime,
  loadSpeechModel,
  transcribeSpeech,
  transcribeSpeechFromMonoPcm,
  transcribeSpeechBatch,
  PipelineAbortedError,
  type BackendCapabilities,
  type ExecutionBackend,
  type AudioInputLike,
  type BaseTranscriptionOptions,
  type TranscriptResponse,
  type TranscriptResponseFlavor,
} from '@asrjs/speech-recognition';
import { loadBuiltInSpeechModel } from '@asrjs/speech-recognition/builtins';
import { AssetLoadAbortedError } from '../src/io/abort.js';
import * as sessionRuntime from '../src/runtime/session.js';
import { describe, expect, it, vi } from 'vitest';

function createStaticBackend(capabilities: BackendCapabilities): ExecutionBackend {
  return {
    id: capabilities.id,
    displayName: capabilities.displayName,
    async probeCapabilities() {
      return capabilities;
    },
    async createExecutionContext() {
      return {
        backendId: capabilities.id,
        capabilities,
        dispose() {},
      };
    },
  };
}

type ProbeNative = { readonly utterance: string };

function probeResponse<TFlavor extends TranscriptResponseFlavor>(
  flavor?: TFlavor,
): TranscriptResponse<ProbeNative, TFlavor> {
  const canonical = {
    text: 'probe',
    warnings: [],
    meta: { detailLevel: 'text' as const, isFinal: true },
  };
  const native: ProbeNative = { utterance: 'probe' };
  return (
    flavor === 'native' ? native : flavor === 'canonical+native' ? { canonical, native } : canonical
  ) as TranscriptResponse<ProbeNative, TFlavor>;
}

function createProbeRuntime(options: {
  onCreateModel: () => Promise<void>;
  onDisposeModel: () => void | Promise<void>;
  onCreateSession?: () => Promise<void>;
  onTranscribe?: (signal?: { readonly aborted: boolean } | null) => Promise<void>;
  onTranscribeBatch?: (signal?: { readonly aborted: boolean } | null) => Promise<void>;
}) {
  const runtime = createSpeechRuntime();
  runtime.registerBackend(
    createStaticBackend({
      id: 'test',
      displayName: 'Test',
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
  );
  runtime.registerModelFamily({
    family: 'pipeline-lifecycle-probe',
    supports(modelId: string) {
      return modelId === 'pipeline-lifecycle-probe-1';
    },
    async createModel(request, context) {
      await options.onCreateModel();
      return {
        info: {
          family: 'pipeline-lifecycle-probe',
          modelId: request.modelId,
          classification: { ecosystem: 'test', task: 'asr' },
        },
        backend: context.backend,
        async createSession() {
          await options.onCreateSession?.();
          return {
            async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
              _input: AudioInputLike,
              transcribeOptions?: BaseTranscriptionOptions & { readonly responseFlavor?: TFlavor },
            ): Promise<TranscriptResponse<ProbeNative, TFlavor>> {
              await options.onTranscribe?.(transcribeOptions?.signal);
              return probeResponse(transcribeOptions?.responseFlavor);
            },
            async transcribeBatch<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
              inputs: readonly AudioInputLike[],
              transcribeOptions?: BaseTranscriptionOptions & { readonly responseFlavor?: TFlavor },
            ): Promise<readonly TranscriptResponse<ProbeNative, TFlavor>[]> {
              await options.onTranscribeBatch?.(transcribeOptions?.signal);
              return inputs.map(() => probeResponse(transcribeOptions?.responseFlavor));
            },
            dispose() {},
          };
        },
        dispose() {
          return options.onDisposeModel();
        },
      };
    },
  });
  return runtime;
}

const REQUEST = {
  family: 'pipeline-lifecycle-probe',
  modelId: 'pipeline-lifecycle-probe-1',
  backend: 'test',
};

describe.each([
  { name: 'root / borrowed runtime', load: loadSpeechModel, ownsRuntime: false },
  { name: 'builtins / borrowed runtime', load: loadBuiltInSpeechModel, ownsRuntime: false },
  { name: 'root / owned runtime', load: loadSpeechModel, ownsRuntime: true },
  { name: 'builtins / owned runtime', load: loadBuiltInSpeechModel, ownsRuntime: true },
])('Load failure preservation: $name', ({ load, ownsRuntime }) => {
  for (const callback of ['hooks', 'direct'] as const) {
    for (const failsCleanup of [false, true]) {
      it.each([new Error('notification failed'), undefined, null])(
        `retains cancellation and ${callback} failure after cleanup (failure=${failsCleanup}): %s`,
        async (notification) => {
          const primary = new AssetLoadAbortedError('session-create');
          const cleanup = new AggregateError([new Error('release failed')], 'cleanup');
          const order: string[] = [];
          const runtime = createProbeRuntime({
            onCreateModel: async () => undefined,
            onCreateSession: async () => {
              throw primary;
            },
            onDisposeModel: async () => {
              await Promise.resolve();
              order.push('cleanup');
              if (failsCleanup) throw cleanup;
            },
          });
          const factory = ownsRuntime
            ? vi.spyOn(sessionRuntime, 'createSpeechRuntime').mockReturnValueOnce(runtime)
            : undefined;
          const onProgress = (event: { readonly phase: string }) => {
            if (event.phase === 'cancelled') {
              order.push('cancelled');
              throw notification;
            }
          };
          try {
            const result = await load({
              ...REQUEST,
              runtime: ownsRuntime ? undefined : runtime,
              ...(callback === 'hooks' ? { hooks: { onProgress } } : { onProgress }),
            }).then(
              () => {
                throw new Error('Expected rejection');
              },
              (error: unknown) => error,
            );
            expect(result).toBeInstanceOf(AggregateError);
            const errors = (result as AggregateError).errors;
            expect(errors).toHaveLength(failsCleanup ? 3 : 2);
            expect(errors[0]).toBeInstanceOf(PipelineAbortedError);
            expect(errors[0]).toMatchObject({ stageId: 'load', cause: primary });
            expect((result as AggregateError).cause).toBe(errors[0]);
            if (failsCleanup) expect(errors[1]).toBe(cleanup);
            expect(errors.at(-1)).toBe(notification);
            expect(order).toEqual(['cleanup', 'cancelled']);
          } finally {
            factory?.mockRestore();
            await runtime.dispose().catch(() => undefined);
          }
        },
      );
    }
  }

  it.each([
    ['session error', new Error('session creation failed')],
    ['nested session error', new AggregateError([new Error('init failed')], 'session init')],
    ['cancellation', new PipelineAbortedError('session-create')],
    ['undefined', undefined],
    ['null', null],
  ])('retains %s and nested cleanup failure', async (_label, primary) => {
    const cleanup = new AggregateError([new Error('resource release failed')], 'cleanup');
    let releases = 0;
    const phases: string[] = [];
    const runtime = createProbeRuntime({
      onCreateModel: async () => undefined,
      onCreateSession: async () => {
        throw primary;
      },
      onDisposeModel: () => {
        releases++;
        throw cleanup;
      },
    });
    // Only intercept allocation; registrations, model tracking and disposal are real.
    const factory = ownsRuntime
      ? vi.spyOn(sessionRuntime, 'createSpeechRuntime').mockReturnValueOnce(runtime)
      : undefined;
    try {
      const result = await load({
        ...REQUEST,
        runtime: ownsRuntime ? undefined : runtime,
        onProgress: (event) => phases.push(event.phase),
      }).then(
        () => {
          throw new Error('Expected rejection');
        },
        (error: unknown) => error,
      );
      expect(result).toBeInstanceOf(AggregateError);
      expect((result as AggregateError).errors[0]).toBe(primary);
      expect((result as AggregateError).errors[1]).toBe(cleanup);
      expect((result as AggregateError).cause).toBe(primary);
      expect(releases).toBe(1);
      expect(phases.includes('ready')).toBe(false);
      expect(phases.includes('cancelled')).toBe(primary instanceof PipelineAbortedError);
      if (ownsRuntime) await expect(runtime.dispose()).rejects.toBe(cleanup);
      else await runtime.dispose();
      expect(releases).toBe(1);
    } finally {
      factory?.mockRestore();
      await runtime.dispose().catch(() => undefined);
    }
  });

  it.each([false, true])(
    'normalizes asset cancellation; cleanup failure=%s',
    async (failsCleanup) => {
      const primary = new AssetLoadAbortedError('session-create');
      const cleanup = new Error('cleanup failed');
      const phases: string[] = [];
      const runtime = createProbeRuntime({
        onCreateModel: async () => undefined,
        onCreateSession: async () => {
          throw primary;
        },
        onDisposeModel: () => {
          if (failsCleanup) throw cleanup;
        },
      });
      const factory = ownsRuntime
        ? vi.spyOn(sessionRuntime, 'createSpeechRuntime').mockReturnValueOnce(runtime)
        : undefined;
      try {
        const result = await load({
          ...REQUEST,
          runtime: ownsRuntime ? undefined : runtime,
          onProgress: (event) => phases.push(event.phase),
        }).then(
          () => {
            throw new Error('Expected rejection');
          },
          (error: unknown) => error,
        );
        const cancellation = failsCleanup ? (result as AggregateError).errors[0] : result;
        expect(cancellation).toBeInstanceOf(PipelineAbortedError);
        expect(cancellation).toMatchObject({ stageId: 'load', cause: primary });
        expect(phases.filter((phase) => phase === 'cancelled')).toHaveLength(1);
        if (failsCleanup) {
          expect((result as AggregateError).errors[1]).toBe(cleanup);
          expect((result as AggregateError).cause).toBe(cancellation);
        }
      } finally {
        factory?.mockRestore();
        await runtime.dispose().catch(() => undefined);
      }
    },
  );

  it('retains the primary error identity when cleanup succeeds', async () => {
    const primary = new Error('session creation failed');
    let releases = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => undefined,
      onCreateSession: async () => {
        throw primary;
      },
      onDisposeModel: () => {
        releases++;
      },
    });
    const factory = ownsRuntime
      ? vi.spyOn(sessionRuntime, 'createSpeechRuntime').mockReturnValueOnce(runtime)
      : undefined;
    try {
      await expect(load({ ...REQUEST, runtime: ownsRuntime ? undefined : runtime })).rejects.toBe(
        primary,
      );
      expect(releases).toBe(1);
    } finally {
      factory?.mockRestore();
      await runtime.dispose();
    }
  });

  it.each([undefined, null, new Error('async cleanup failed')])(
    'awaits asynchronous cleanup and retains its rejection value %s',
    async (cleanup) => {
      const primary = new Error('session creation failed');
      let markStarted!: () => void;
      const started = new Promise<void>((resolve) => {
        markStarted = resolve;
      });
      let release!: () => void;
      const gate = new Promise<void>((resolve) => {
        release = resolve;
      });
      const runtime = createProbeRuntime({
        onCreateModel: async () => undefined,
        onCreateSession: async () => {
          throw primary;
        },
        onDisposeModel: async () => {
          markStarted();
          await gate;
          throw cleanup;
        },
      });
      const factory = ownsRuntime
        ? vi.spyOn(sessionRuntime, 'createSpeechRuntime').mockReturnValueOnce(runtime)
        : undefined;
      try {
        let settled = false;
        const result = load({ ...REQUEST, runtime: ownsRuntime ? undefined : runtime }).then(
          () => {
            throw new Error('Expected rejection');
          },
          (error: unknown) => {
            settled = true;
            return error;
          },
        );
        await started;
        expect(settled).toBe(false);
        release();
        const error = await result;
        expect(error).toBeInstanceOf(AggregateError);
        expect((error as AggregateError).errors[0]).toBe(primary);
        expect((error as AggregateError).errors[1]).toBe(cleanup);
        expect((error as AggregateError).cause).toBe(primary);
      } finally {
        release();
        factory?.mockRestore();
        await runtime.dispose().catch(() => undefined);
      }
    },
  );
});

const ONE_SHOT_CALLS = [
  {
    name: 'single',
    invoke: (runtime: ReturnType<typeof createProbeRuntime>) =>
      transcribeSpeech(new Float32Array(160), { ...REQUEST, runtime }),
  },
  {
    name: 'mono PCM',
    invoke: (runtime: ReturnType<typeof createProbeRuntime>) =>
      transcribeSpeechFromMonoPcm(new Float32Array(160), 16000, { ...REQUEST, runtime }),
  },
  {
    name: 'batch',
    invoke: (runtime: ReturnType<typeof createProbeRuntime>) =>
      transcribeSpeechBatch([new Float32Array(160), new Float32Array(80)], {
        ...REQUEST,
        runtime,
      }),
  },
];

describe.each(ONE_SHOT_CALLS)('One-shot $name failure preservation', ({ invoke }) => {
  it.each([
    ['inference error', new Error('inference failed')],
    ['cancellation', new PipelineAbortedError('probe')],
    ['undefined', undefined],
    ['null', null],
  ])('retains %s and nested cleanup failure', async (_label, primary) => {
    const cleanup = new AggregateError([new Error('resource release failed')], 'cleanup');
    let releases = 0;
    const fail = async () => {
      throw primary;
    };
    const runtime = createProbeRuntime({
      onCreateModel: async () => undefined,
      onTranscribe: fail,
      onTranscribeBatch: fail,
      onDisposeModel: () => {
        if (++releases === 1) throw cleanup;
      },
    });
    const outcome = await invoke(runtime).then(
      () => {
        throw new Error('Expected rejection');
      },
      (error: unknown) => error,
    );
    expect(outcome).toBeInstanceOf(AggregateError);
    expect((outcome as AggregateError).errors).toEqual([primary, cleanup]);
    expect((outcome as AggregateError).errors[0]).toBe(primary);
    expect((outcome as AggregateError).errors[1]).toBe(cleanup);
    expect((outcome as AggregateError).cause).toBe(primary);
    expect(releases).toBe(1);
    await runtime.dispose();
  });

  it.each([
    ['inference error', new Error('inference failed')],
    ['cancellation', new PipelineAbortedError('probe')],
    ['undefined', undefined],
    ['null', null],
  ])('retains %s identity when cleanup succeeds', async (_label, primary) => {
    let releases = 0;
    const fail = async () => {
      throw primary;
    };
    const runtime = createProbeRuntime({
      onCreateModel: async () => undefined,
      onTranscribe: fail,
      onTranscribeBatch: fail,
      onDisposeModel: () => {
        releases++;
      },
    });
    await expect(invoke(runtime)).rejects.toBe(primary);
    expect(releases).toBe(1);
    await runtime.dispose();
  });

  it('awaits cleanup and retains its identity after successful inference', async () => {
    const cleanup = new Error('release failed');
    let started!: () => void;
    const cleanupStarted = new Promise<void>((resolve) => {
      started = resolve;
    });
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    let releases = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => undefined,
      onDisposeModel: async () => {
        if (++releases !== 1) return;
        started();
        await gate;
        throw cleanup;
      },
    });
    let settled = false;
    const result = invoke(runtime);
    const outcome = result.then(
      () => {
        settled = true;
      },
      () => {
        settled = true;
      },
    );
    await cleanupStarted;
    await Promise.resolve();
    expect(settled).toBe(false);
    release();
    await expect(result).rejects.toBe(cleanup);
    await outcome;
    expect(releases).toBe(1);
    await runtime.dispose();
  });
});

describe('SpeechPipeline lifecycle races', () => {
  it('rejects an aborted joined load without cancelling the owner or evicting its handle', async () => {
    let releaseCreate!: () => void;
    let markStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const gate = new Promise<void>((resolve) => {
      releaseCreate = resolve;
    });
    let createCount = 0;
    let disposeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => {
        createCount += 1;
        markStarted();
        await gate;
      },
      onDisposeModel: () => {
        disposeCount += 1;
      },
    });
    const pipeline = createSpeechPipeline({ runtime });
    const signal = { aborted: false };
    const request = { ...REQUEST, cacheKey: 'shared-cancellation-probe' };
    const owner = pipeline.loadModel(request);
    await started;
    const joined = pipeline.loadModel({ ...request, signal });
    const joinedOutcome = joined.then(
      () => 'returned-model',
      (error: unknown) => error,
    );
    await Promise.resolve();
    await Promise.resolve();
    signal.aborted = true;
    releaseCreate();
    try {
      const handle = await owner;
      expect(await joinedOutcome).toBeInstanceOf(PipelineAbortedError);
      expect(await pipeline.loadModel(request)).toBe(handle);
      expect(createCount).toBe(1);
      expect(disposeCount).toBe(0);
      expect(await handle.transcribe(new Float32Array(16000))).toMatchObject({ text: 'probe' });
    } finally {
      releaseCreate();
      await pipeline.dispose();
      await runtime.dispose();
    }
    expect(disposeCount).toBe(1);
  });

  it('does not return or force-reload a cached handle for an already aborted caller', async () => {
    let createCount = 0;
    let disposeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => {
        createCount += 1;
      },
      onDisposeModel: () => {
        disposeCount += 1;
      },
    });
    const pipeline = createSpeechPipeline({ runtime });
    const request = { ...REQUEST, cacheKey: 'cached-cancellation-probe' };
    try {
      const handle = await pipeline.loadModel(request);
      for (const forceReload of [false, true]) {
        await expect(
          pipeline.loadModel({ ...request, forceReload, signal: { aborted: true } }),
        ).rejects.toBeInstanceOf(PipelineAbortedError);
      }
      expect(await pipeline.loadModel(request)).toBe(handle);
      expect(createCount).toBe(1);
      expect(disposeCount).toBe(0);
    } finally {
      await pipeline.dispose();
      await runtime.dispose();
    }
  });

  it('propagates owner load cancellation to a live joiner and allows an explicit retry', async () => {
    let releaseCreate!: () => void;
    let markStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const gate = new Promise<void>((resolve) => {
      releaseCreate = resolve;
    });
    let createCount = 0;
    let disposeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => {
        createCount += 1;
        markStarted();
        await gate;
      },
      onDisposeModel: () => {
        disposeCount += 1;
      },
    });
    const pipeline = createSpeechPipeline({ runtime });
    const request = { ...REQUEST, cacheKey: 'owner-cancellation-probe' };
    const ownerSignal = { aborted: false };
    const owner = pipeline.loadModel({ ...request, signal: ownerSignal });
    const ownerOutcome = owner.then(
      () => 'returned-model',
      (error: unknown) => error,
    );
    await started;
    const joined = pipeline.loadModel(request);
    const joinedOutcome = joined.then(
      () => 'returned-model',
      (error: unknown) => error,
    );
    await Promise.resolve();
    await Promise.resolve();
    ownerSignal.aborted = true;
    releaseCreate();
    try {
      expect(await ownerOutcome).toBeInstanceOf(PipelineAbortedError);
      expect(await joinedOutcome).toBeInstanceOf(PipelineAbortedError);
      expect(pipeline.listLoadedModels()).toEqual([]);
      expect(disposeCount).toBe(1);
      const retry = await pipeline.loadModel(request);
      expect(createCount).toBe(2);
      expect(await retry.transcribe(new Float32Array(16000))).toMatchObject({ text: 'probe' });
    } finally {
      releaseCreate();
      await pipeline.dispose();
      await runtime.dispose();
    }
    expect(disposeCount).toBe(2);
  });

  it('does not create a replacement when forceReload is aborted during active-operation disposal', async () => {
    let releaseOperation!: () => void;
    let markStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const gate = new Promise<void>((resolve) => {
      releaseOperation = resolve;
    });
    let createCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => {
        createCount += 1;
      },
      onDisposeModel: () => undefined,
      onTranscribe: async () => {
        markStarted();
        await gate;
      },
    });
    const pipeline = createSpeechPipeline({ runtime });
    const request = { ...REQUEST, cacheKey: 'force-reload-cancel-probe' };
    await pipeline.loadModel(request);
    const active = pipeline.transcribe(new Float32Array(16000), request);
    await started;
    const signal = { aborted: false };
    const replacement = pipeline.loadModel({ ...request, forceReload: true, signal });
    const outcome = replacement.then(
      () => 'returned-model',
      (error: unknown) => error,
    );
    await Promise.resolve();
    await Promise.resolve();
    signal.aborted = true;
    releaseOperation();
    try {
      await active;
      expect(await outcome).toBeInstanceOf(PipelineAbortedError);
      expect(createCount).toBe(1);
      expect(pipeline.listLoadedModels()).toEqual([]);
    } finally {
      releaseOperation();
      await pipeline.dispose();
      await runtime.dispose();
    }
  });

  it.each(['canonical', 'canonical+native', 'native'] as const)(
    'preserves %s response shape for single and batch calls',
    async (responseFlavor) => {
      let disposeCount = 0;
      const runtime = createProbeRuntime({
        onCreateModel: async () => undefined,
        onDisposeModel: () => {
          disposeCount += 1;
        },
      });
      const loaded = await loadSpeechModel({ ...REQUEST, runtime });
      try {
        const expected =
          responseFlavor === 'native'
            ? { utterance: 'probe' }
            : responseFlavor === 'canonical+native'
              ? {
                  canonical: {
                    text: 'probe',
                    warnings: [],
                    meta: { detailLevel: 'text', isFinal: true },
                  },
                  native: { utterance: 'probe' },
                }
              : { text: 'probe', warnings: [], meta: { detailLevel: 'text', isFinal: true } };
        expect(await loaded.transcribe(new Float32Array(16000), { responseFlavor })).toEqual(
          expected,
        );
        expect(
          await loaded.transcribeBatch([new Float32Array(16000), new Float32Array(8000)], {
            responseFlavor,
          }),
        ).toEqual([expected, expected]);
      } finally {
        await loaded.dispose();
        await runtime.dispose();
      }
      expect(disposeCount).toBe(1);
    },
  );

  it('does not cache a handle invalidated by disposeModel during load', async () => {
    let releaseCreate!: () => void;
    const createGate = new Promise<void>((resolve) => {
      releaseCreate = resolve;
    });
    let createCount = 0;
    let disposeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => {
        createCount += 1;
        await createGate;
      },
      onDisposeModel: () => {
        disposeCount += 1;
      },
    });
    const pipeline = createSpeechPipeline({ runtime });

    const loading = pipeline.loadModel(REQUEST);
    await Promise.resolve();
    const removing = pipeline.disposeModel(REQUEST);
    releaseCreate();

    await expect(loading).rejects.toThrow('disposed');
    await removing;
    expect(disposeCount).toBe(1);
    expect(pipeline.listLoadedModels()).toEqual([]);

    const reloaded = await pipeline.loadModel(REQUEST);
    expect(createCount).toBe(2);
    expect(await reloaded.transcribe(new Float32Array(16000))).toMatchObject({ text: 'probe' });

    await pipeline.dispose();
    await runtime.dispose();
  });

  it('forceReload creates a fresh handle when it invalidates an in-flight load', async () => {
    let releaseCreate!: () => void;
    const createGate = new Promise<void>((resolve) => {
      releaseCreate = resolve;
    });
    let createCount = 0;
    let disposeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => {
        createCount += 1;
        if (createCount === 1) {
          await createGate;
        }
      },
      onDisposeModel: () => {
        disposeCount += 1;
      },
    });
    const pipeline = createSpeechPipeline({ runtime });

    const first = pipeline.loadModel(REQUEST);
    await Promise.resolve();
    const second = pipeline.loadModel({ ...REQUEST, forceReload: true });
    releaseCreate();

    await expect(first).rejects.toThrow('disposed');
    const fresh = await second;
    expect(createCount).toBe(2);
    expect(disposeCount).toBe(1);
    expect(await fresh.transcribe(new Float32Array(16000))).toMatchObject({ text: 'probe' });
    expect(pipeline.listLoadedModels()).toHaveLength(1);

    await pipeline.dispose();
    await runtime.dispose();
  });

  it('does not cache a handle invalidated by flushAllModels during load', async () => {
    let releaseCreate!: () => void;
    const createGate = new Promise<void>((resolve) => {
      releaseCreate = resolve;
    });
    let createCount = 0;
    let disposeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => {
        createCount += 1;
        if (createCount === 1) {
          await createGate;
        }
      },
      onDisposeModel: () => {
        disposeCount += 1;
      },
    });
    const pipeline = createSpeechPipeline({ runtime });

    const loading = pipeline.loadModel(REQUEST);
    await Promise.resolve();
    const flushing = pipeline.flushAllModels();
    releaseCreate();

    await expect(loading).rejects.toThrow('disposed');
    await flushing;
    expect(disposeCount).toBe(1);
    expect(pipeline.listLoadedModels()).toEqual([]);

    const reloaded = await pipeline.loadModel(REQUEST);
    expect(createCount).toBe(2);
    expect(await reloaded.transcribe(new Float32Array(16000))).toMatchObject({ text: 'probe' });

    await pipeline.dispose();
    await runtime.dispose();
  });

  it('coalesces concurrent pipeline disposal while a load is in flight', async () => {
    let releaseCreate!: () => void;
    const createGate = new Promise<void>((resolve) => {
      releaseCreate = resolve;
    });
    let disposeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => {
        await createGate;
      },
      onDisposeModel: () => {
        disposeCount += 1;
      },
    });
    const pipeline = createSpeechPipeline({ runtime });

    const loading = pipeline.loadModel(REQUEST);
    await Promise.resolve();
    const firstDispose = pipeline.dispose();
    const secondDispose = pipeline.dispose();
    releaseCreate();

    await expect(loading).rejects.toThrow('disposed');
    await Promise.all([firstDispose, secondDispose]);
    expect(disposeCount).toBe(1);
    expect(pipeline.listLoadedModels()).toEqual([]);

    await runtime.dispose();
  });

  it('waits for active pipeline transcription before disposeModel disposes the handle', async () => {
    let releaseTranscribe!: () => void;
    const transcribeGate = new Promise<void>((resolve) => {
      releaseTranscribe = resolve;
    });
    let transcribeStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      transcribeStarted = resolve;
    });
    let disposeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => undefined,
      onDisposeModel: () => {
        disposeCount += 1;
      },
      onTranscribe: async () => {
        transcribeStarted();
        await transcribeGate;
      },
    });
    const pipeline = createSpeechPipeline({ runtime });
    await pipeline.loadModel(REQUEST);

    const loaded = await pipeline.loadModel(REQUEST);
    const transcribing = loaded.transcribe(new Float32Array(16000));
    await started;
    let disposed = false;
    const disposing = pipeline.disposeModel(REQUEST).then(() => {
      disposed = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(disposed).toBe(false);
    expect(disposeCount).toBe(0);
    releaseTranscribe();

    await expect(transcribing).resolves.toMatchObject({ text: 'probe' });
    await disposing;
    expect(disposeCount).toBe(1);

    await pipeline.dispose();
    await runtime.dispose();
  });

  it('waits for active built-in handle transcription before disposing the model', async () => {
    let releaseTranscribe!: () => void;
    const transcribeGate = new Promise<void>((resolve) => {
      releaseTranscribe = resolve;
    });
    let transcribeStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      transcribeStarted = resolve;
    });
    let disposeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => undefined,
      onDisposeModel: () => {
        disposeCount += 1;
      },
      onTranscribe: async () => {
        transcribeStarted();
        await transcribeGate;
      },
    });
    const loaded = await loadBuiltInSpeechModel({ ...REQUEST, runtime });

    const transcribing = loaded.transcribe(new Float32Array(16000));
    await started;
    let disposed = false;
    const disposing = loaded.dispose().then(() => {
      disposed = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(disposed).toBe(false);
    expect(disposeCount).toBe(0);
    releaseTranscribe();

    await expect(transcribing).resolves.toMatchObject({ text: 'probe' });
    await disposing;
    expect(disposeCount).toBe(1);

    await runtime.dispose();
  });

  it('propagates one-shot load cancellation into an in-flight transcription', async () => {
    let releaseTranscribe!: () => void;
    const transcribeGate = new Promise<void>((resolve) => {
      releaseTranscribe = resolve;
    });
    let transcribeStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      transcribeStarted = resolve;
    });
    const controller = new AbortController();
    const runtime = createProbeRuntime({
      onCreateModel: async () => undefined,
      onDisposeModel: () => undefined,
      onTranscribe: async (signal) => {
        transcribeStarted();
        await transcribeGate;
        if (signal?.aborted) {
          throw new PipelineAbortedError('probe-transcribe');
        }
      },
    });

    const transcribing = transcribeSpeech(new Float32Array(16000), {
      ...REQUEST,
      runtime,
      signal: controller.signal,
    });
    await started;
    controller.abort();
    releaseTranscribe();

    await expect(transcribing).rejects.toBeInstanceOf(PipelineAbortedError);
    await runtime.dispose();
  });

  it('propagates one-shot batch cancellation into an in-flight transcription', async () => {
    let releaseBatch!: () => void;
    const batchGate = new Promise<void>((resolve) => {
      releaseBatch = resolve;
    });
    let batchStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      batchStarted = resolve;
    });
    const controller = new AbortController();
    const runtime = createProbeRuntime({
      onCreateModel: async () => undefined,
      onDisposeModel: () => undefined,
      onTranscribeBatch: async (signal) => {
        batchStarted();
        await batchGate;
        if (signal?.aborted) {
          throw new PipelineAbortedError('probe-transcribe-batch');
        }
      },
    });

    const transcribing = transcribeSpeechBatch([new Float32Array(16000)], {
      ...REQUEST,
      runtime,
      signal: controller.signal,
    });
    await started;
    controller.abort();
    releaseBatch();

    await expect(transcribing).rejects.toBeInstanceOf(PipelineAbortedError);
    await runtime.dispose();
  });

  it('reuses the cached model after cancelling an in-flight transcription', async () => {
    let releaseTranscribe!: () => void;
    const transcribeGate = new Promise<void>((resolve) => {
      releaseTranscribe = resolve;
    });
    let transcribeStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      transcribeStarted = resolve;
    });
    const controller = new AbortController();
    let createCount = 0;
    let disposeCount = 0;
    let transcribeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => {
        createCount += 1;
      },
      onDisposeModel: () => {
        disposeCount += 1;
      },
      onTranscribe: async (signal) => {
        transcribeCount += 1;
        transcribeStarted();
        await transcribeGate;
        if (signal?.aborted) {
          throw new PipelineAbortedError('probe-transcribe');
        }
      },
    });
    const pipeline = createSpeechPipeline({ runtime, cacheModels: true });

    const transcribing = pipeline.transcribe(new Float32Array(16000), {
      ...REQUEST,
      cacheKey: 'cancel-reuse-probe',
      signal: controller.signal,
    });
    await started;
    controller.abort();
    releaseTranscribe();

    try {
      await expect(transcribing).rejects.toBeInstanceOf(PipelineAbortedError);
      const result = await pipeline.transcribe(new Float32Array(16000), {
        ...REQUEST,
        cacheKey: 'cancel-reuse-probe',
        signal: new AbortController().signal,
      });
      expect(result.text).toBe('probe');
      expect(transcribeCount).toBe(2);
      expect(createCount).toBe(1);
      expect(disposeCount).toBe(0);
    } finally {
      await pipeline.dispose();
      await runtime.dispose();
    }
    expect(disposeCount).toBe(1);
  });

  it('propagates pipeline cancellation into an in-flight batch transcription', async () => {
    let releaseBatch!: () => void;
    const batchGate = new Promise<void>((resolve) => {
      releaseBatch = resolve;
    });
    let batchStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      batchStarted = resolve;
    });
    const controller = new AbortController();
    const runtime = createProbeRuntime({
      onCreateModel: async () => undefined,
      onDisposeModel: () => undefined,
      onTranscribeBatch: async (signal) => {
        batchStarted();
        await batchGate;
        if (signal?.aborted) {
          throw new PipelineAbortedError('probe-transcribe-batch');
        }
      },
    });
    const pipeline = createSpeechPipeline({ runtime, cacheModels: true });

    const transcribing = pipeline.transcribeBatch([new Float32Array(16000)], {
      ...REQUEST,
      signal: controller.signal,
    });
    await started;
    controller.abort();
    releaseBatch();

    await expect(transcribing).rejects.toBeInstanceOf(PipelineAbortedError);
    await pipeline.dispose();
    await runtime.dispose();
  });

  it('waits for active exposed-session transcription before disposing the loaded handle', async () => {
    let releaseTranscribe!: () => void;
    const transcribeGate = new Promise<void>((resolve) => {
      releaseTranscribe = resolve;
    });
    let transcribeStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      transcribeStarted = resolve;
    });
    let disposeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => undefined,
      onDisposeModel: () => {
        disposeCount += 1;
      },
      onTranscribe: async () => {
        transcribeStarted();
        await transcribeGate;
      },
    });
    const loaded = await loadSpeechModel({ ...REQUEST, runtime });

    const transcribing = loaded.session.transcribe(new Float32Array(16000));
    await started;
    let disposed = false;
    const disposing = loaded.dispose().then(() => {
      disposed = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(disposed).toBe(false);
    expect(disposeCount).toBe(0);
    releaseTranscribe();

    await expect(transcribing).resolves.toMatchObject({ text: 'probe' });
    await disposing;
    expect(disposeCount).toBe(1);

    await runtime.dispose();
  });

  it('waits for active loaded-handle batch transcription before disposing the model', async () => {
    let releaseBatch!: () => void;
    const batchGate = new Promise<void>((resolve) => {
      releaseBatch = resolve;
    });
    let batchStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      batchStarted = resolve;
    });
    let disposeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => undefined,
      onDisposeModel: () => {
        disposeCount += 1;
      },
      onTranscribeBatch: async () => {
        batchStarted();
        await batchGate;
      },
    });
    const loaded = await loadSpeechModel({ ...REQUEST, runtime });
    expect(loaded.supportsBatch).toBe(true);

    const transcribing = loaded.transcribeBatch([new Float32Array(16000)]);
    await started;
    let disposed = false;
    const disposing = loaded.dispose().then(() => {
      disposed = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(disposed).toBe(false);
    expect(disposeCount).toBe(0);
    releaseBatch();

    await expect(transcribing).resolves.toHaveLength(1);
    await disposing;
    expect(disposeCount).toBe(1);

    await runtime.dispose();
  });

  it('makes direct cached-handle disposal wait for its active transcription', async () => {
    let releaseTranscribe!: () => void;
    const transcribeGate = new Promise<void>((resolve) => {
      releaseTranscribe = resolve;
    });
    let transcribeStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      transcribeStarted = resolve;
    });
    let disposeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => undefined,
      onDisposeModel: () => {
        disposeCount += 1;
      },
      onTranscribe: async () => {
        transcribeStarted();
        await transcribeGate;
      },
    });
    const pipeline = createSpeechPipeline({ runtime });
    const loaded = await pipeline.loadModel(REQUEST);

    const transcribing = loaded.transcribe(new Float32Array(16000));
    await started;
    let disposed = false;
    const disposing = loaded.dispose().then(() => {
      disposed = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(disposed).toBe(false);
    expect(disposeCount).toBe(0);
    releaseTranscribe();

    await expect(transcribing).resolves.toMatchObject({ text: 'probe' });
    await disposing;
    expect(disposeCount).toBe(1);
    expect(pipeline.listLoadedModels()).toEqual([]);

    await pipeline.dispose();
    await runtime.dispose();
  });

  it('does not let a queued transcription recreate a model after disposeModel begins', async () => {
    let releaseCreate!: () => void;
    let createStarted!: () => void;
    const createGate = new Promise<void>((resolve) => {
      releaseCreate = resolve;
    });
    const started = new Promise<void>((resolve) => {
      createStarted = resolve;
    });
    let disposeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => {
        createStarted();
        await createGate;
      },
      onDisposeModel: () => {
        disposeCount += 1;
      },
    });
    const pipeline = createSpeechPipeline({ runtime });

    const transcribing = pipeline.transcribe(new Float32Array(16000), REQUEST);
    await started;
    const disposing = pipeline.disposeModel(REQUEST);
    releaseCreate();

    await expect(transcribing).rejects.toThrow('disposed');
    await disposing;
    expect(disposeCount).toBe(1);
    expect(pipeline.listLoadedModels()).toEqual([]);

    await pipeline.dispose();
    await runtime.dispose();
  });

  it('waits for active pipeline transcription before flushAllModels disposes the handle', async () => {
    let releaseTranscribe!: () => void;
    const transcribeGate = new Promise<void>((resolve) => {
      releaseTranscribe = resolve;
    });
    let transcribeStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      transcribeStarted = resolve;
    });
    let disposeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => undefined,
      onDisposeModel: () => {
        disposeCount += 1;
      },
      onTranscribe: async () => {
        transcribeStarted();
        await transcribeGate;
      },
    });
    const pipeline = createSpeechPipeline({ runtime });
    await pipeline.loadModel(REQUEST);

    const transcribing = pipeline.transcribe(new Float32Array(16000), REQUEST);
    await started;
    let flushed = false;
    const flushing = pipeline.flushAllModels().then(() => {
      flushed = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(flushed).toBe(false);
    expect(disposeCount).toBe(0);
    releaseTranscribe();

    await expect(transcribing).resolves.toMatchObject({ text: 'probe' });
    await flushing;
    expect(disposeCount).toBe(1);
    expect(pipeline.listLoadedModels()).toEqual([]);

    await pipeline.dispose();
    await runtime.dispose();
  });

  it('waits for active pipeline transcription before dispose shuts down the runtime', async () => {
    let releaseTranscribe!: () => void;
    const transcribeGate = new Promise<void>((resolve) => {
      releaseTranscribe = resolve;
    });
    let transcribeStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      transcribeStarted = resolve;
    });
    let disposeCount = 0;
    const runtime = createProbeRuntime({
      onCreateModel: async () => undefined,
      onDisposeModel: () => {
        disposeCount += 1;
      },
      onTranscribe: async () => {
        transcribeStarted();
        await transcribeGate;
      },
    });
    const pipeline = createSpeechPipeline({ runtime });
    await pipeline.loadModel(REQUEST);

    const transcribing = pipeline.transcribe(new Float32Array(16000), REQUEST);
    await started;
    let disposed = false;
    const disposing = pipeline.dispose().then(() => {
      disposed = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(disposed).toBe(false);
    expect(disposeCount).toBe(0);
    releaseTranscribe();

    await expect(transcribing).resolves.toMatchObject({ text: 'probe' });
    await disposing;
    expect(disposeCount).toBe(1);

    await runtime.dispose();
  });
});
