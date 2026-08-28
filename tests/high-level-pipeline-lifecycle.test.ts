import {
  createSpeechPipeline,
  createSpeechRuntime,
  type BackendCapabilities,
  type ExecutionBackend,
} from '@asrjs/speech-recognition';
import { describe, expect, it } from 'vitest';

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

function createProbeRuntime(options: {
  onCreateModel: () => Promise<void>;
  onDisposeModel: () => void;
  onTranscribe?: () => Promise<void>;
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
          return {
            async transcribe() {
              await options.onTranscribe?.();
              return {
                text: 'probe',
                warnings: [],
                meta: { detailLevel: 'text' as const, isFinal: true },
              };
            },
            dispose() {},
          };
        },
        dispose() {
          options.onDisposeModel();
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

describe('SpeechPipeline lifecycle races', () => {
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
