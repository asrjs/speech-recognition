import {
  createSpeechPipeline,
  createSpeechRuntime,
  createLoadedSpeechModelHandle,
  loadSpeechModel,
  transcribeSpeech,
  transcribeSpeechFromMonoPcm,
  PipelineAbortedError,
  type BackendCapabilities,
  type ExecutionBackend,
  type BuiltInSpeechModelHandle,
} from '@asrjs/speech-recognition';
import {
  registerBuiltInModelFamilies,
  registerBuiltInPresets,
} from '@asrjs/speech-recognition/builtins';
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
        dispose() {
          return undefined;
        },
      };
    },
  };
}

function createRuntime() {
  const runtime = createSpeechRuntime();
  runtime.registerBackend(
    createStaticBackend({
      id: 'wasm',
      displayName: 'WASM',
      available: true,
      priority: 60,
      environments: ['browser', 'node'],
      acceleration: ['cpu'],
      supportedPrecisions: ['fp32', 'int8'],
      supportsFp16: false,
      supportsInt8: true,
      supportsSharedArrayBuffer: true,
      requiresSharedArrayBuffer: false,
      fallbackSuitable: true,
      notes: [],
    }),
  );
  registerBuiltInModelFamilies(runtime);
  registerBuiltInPresets(runtime, {
    useManifestSources: false,
  });
  return runtime;
}

describe('high-level model-agnostic APIs', () => {
  it('transcribeSpeech runs one-shot load + transcribe + dispose automatically', async () => {
    const runtime = createRuntime();

    const result = await transcribeSpeech(new Float32Array(16000), {
      runtime,
      modelId: 'parakeet-tdt-0.6b-v3',
      backend: 'wasm',
      transcribeOptions: {
        detail: 'segments',
        responseFlavor: 'canonical+native',
      },
    });

    expect(result.canonical.text.length).toBeGreaterThan(0);
    expect(result.native?.warnings?.[0]?.code).toBe('nemo-tdt.stubbed-decoder');

    await runtime.dispose();
  });

  it('transcribeSpeechFromMonoPcm accepts explicit sample rate without manual wrappers', async () => {
    const runtime = createRuntime();

    const result = await transcribeSpeechFromMonoPcm(new Float32Array(16000), 16000, {
      runtime,
      modelId: 'parakeet-tdt-0.6b-v3',
      backend: 'wasm',
      transcribeOptions: {
        responseFlavor: 'canonical',
      },
    });

    expect(result.text.length).toBeGreaterThan(0);

    await runtime.dispose();
  });

  it('loaded model handles can transcribe raw mono PCM directly', async () => {
    const runtime = createRuntime();
    const loaded = await loadSpeechModel({
      runtime,
      modelId: 'parakeet-tdt-0.6b-v3',
      backend: 'wasm',
    });

    const result = await loaded.transcribeMonoPcm(new Float32Array(16000), 16000, {
      responseFlavor: 'canonical',
    });

    expect(result.text.length).toBeGreaterThan(0);
    expect(loaded.supportsBatch).toBe(false);
    expect(loaded.supportsStreaming).toBe(false);
    await expect(loaded.transcribeBatch([new Float32Array(16000)])).rejects.toMatchObject({
      code: 'not-implemented-speech-feature',
    });
    await expect(loaded.createStreamingTranscriber()).rejects.toMatchObject({
      code: 'not-implemented-speech-feature',
    });

    await loaded.dispose();
    await runtime.dispose();
  });

  it('tracks high-level streaming transcribers and disposes them before the model', async () => {
    const runtime = createSpeechRuntime();
    const backend = createStaticBackend({
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
    });
    let transcriberDisposeCount = 0;
    let modelDisposeCount = 0;
    let transcriberDisposed = false;
    let holdPush = false;
    let releasePush!: () => void;
    const pushGate = new Promise<void>((resolve) => {
      releasePush = resolve;
    });
    let releaseTranscriberDispose!: () => void;
    const transcriberDisposeGate = new Promise<void>((resolve) => {
      releaseTranscriberDispose = resolve;
    });
    const partial = {
      kind: 'partial' as const,
      revision: 0,
      text: '',
      committedText: '',
      previewText: '',
      warnings: [],
      meta: { detailLevel: 'text' as const, isFinal: false },
    };
    const streaming = {
      async pushAudio() {
        if (holdPush) {
          await pushGate;
        }
        if (transcriberDisposed) throw new Error('stream disposed');
        return partial;
      },
      async flush() {
        return partial;
      },
      async finalize() {
        return partial;
      },
      reset() {},
      getState() {
        return {
          revision: 0,
          bufferedDurationSeconds: 0,
          committedText: '',
          previewText: '',
          isFinalized: false,
        };
      },
      async dispose() {
        if (!transcriberDisposed) {
          await transcriberDisposeGate;
          transcriberDisposed = true;
          transcriberDisposeCount += 1;
        }
      },
    };
    const session = {
      async transcribe() {
        return {
          text: 'single',
          warnings: [],
          meta: { detailLevel: 'text' as const, isFinal: true },
        };
      },
      dispose() {},
    };
    const model = {
      info: {
        family: 'streaming-test',
        modelId: 'streaming-test-model',
        classification: { ecosystem: 'test', task: 'asr' },
      },
      backend,
      async createSession() {
        return session;
      },
      async createStreamingTranscriber() {
        return streaming;
      },
      async dispose() {
        expect(transcriberDisposed).toBe(true);
        modelDisposeCount += 1;
      },
    };
    const loaded = createLoadedSpeechModelHandle({
      runtime,
      model,
      session,
      async transcribe() {
        return session.transcribe();
      },
      async dispose() {
        await model.dispose();
      },
    } satisfies BuiltInSpeechModelHandle);

    expect(loaded.supportsStreaming).toBe(true);
    const transcriber = await loaded.createStreamingTranscriber();
    await expect(transcriber.pushAudio(new Float32Array(16000))).resolves.toMatchObject({
      kind: 'partial',
    });

    holdPush = true;
    const pushing = transcriber.pushAudio(new Float32Array(16000));
    await Promise.resolve();
    const disposingHandle = loaded.dispose();
    await Promise.resolve();
    expect(modelDisposeCount).toBe(0);
    releasePush();
    await Promise.resolve();
    expect(modelDisposeCount).toBe(0);
    releaseTranscriberDispose();
    await Promise.all([pushing, disposingHandle]);
    expect(transcriberDisposeCount).toBe(1);
    expect(modelDisposeCount).toBe(1);
    await expect(transcriber.pushAudio(new Float32Array(16000))).rejects.toThrow(
      'Streaming transcriber is disposed',
    );
    await expect(loaded.createStreamingTranscriber()).rejects.toThrow('disposed');

    await transcriber.dispose?.();
    expect(transcriberDisposeCount).toBe(1);
    await runtime.dispose();
  });

  it('loaded model handles preserve canonical/native flavors for batch-capable sessions', async () => {
    const runtime = createSpeechRuntime();
    const backend = createStaticBackend({
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
    });
    const nativeTranscript = { utteranceText: 'batch', isFinal: true };
    let batchCalls = 0;
    const session = {
      async transcribe() {
        return {
          text: 'single',
          warnings: [],
          meta: { detailLevel: 'text' as const, isFinal: true },
        };
      },
      async transcribeBatch(
        inputs: readonly unknown[],
        options?: { readonly responseFlavor?: string },
      ) {
        batchCalls += 1;
        if (options?.responseFlavor === 'native') {
          return inputs.map(() => nativeTranscript);
        }
        if (options?.responseFlavor === 'canonical+native') {
          return inputs.map(() => ({
            canonical: {
              text: 'batch',
              warnings: [],
              meta: { detailLevel: 'text' as const, isFinal: true },
            },
            native: nativeTranscript,
          }));
        }
        return inputs.map(() => ({
          text: 'batch',
          warnings: [],
          meta: { detailLevel: 'text' as const, isFinal: true },
        }));
      },
      dispose() {},
    };
    const model = {
      info: {
        family: 'batch-test',
        modelId: 'batch-test-model',
        classification: { ecosystem: 'test', task: 'asr' },
        inference: {
          sampleRate: 16000,
          maxInputDurationSec: 30,
          autoWindowThresholdSec: 30,
          supportsWordTimestamps: false,
          supportsSegmentTimestamps: false,
          defaultSegmentationStrategy: 'none' as const,
          defaultMergeStrategy: 'concat' as const,
        },
      },
      backend,
      async createSession() {
        return session;
      },
      dispose() {},
    };
    const loaded = createLoadedSpeechModelHandle({
      runtime,
      model,
      session,
      async transcribe(input: unknown, options?: { readonly responseFlavor?: string }) {
        return session.transcribe(input, options);
      },
      async dispose() {},
    } satisfies BuiltInSpeechModelHandle);

    expect(loaded.supportsBatch).toBe(true);
    expect(await loaded.transcribeBatch([])).toEqual([]);
    expect(await loaded.session.transcribeBatch?.([])).toEqual([]);
    expect(batchCalls).toBe(0);
    const canonical = await loaded.transcribeBatch(
      [new Float32Array(16000), new Float32Array(8000)],
      {
        detail: 'text',
      },
    );
    expect(canonical.map((item) => item.text)).toEqual(['batch', 'batch']);
    const native = await loaded.transcribeBatch([new Float32Array(16000)], {
      responseFlavor: 'native',
    });
    expect(native[0]).toEqual(nativeTranscript);
    const envelope = await loaded.transcribeBatch([new Float32Array(16000)], {
      responseFlavor: 'canonical+native',
    });
    expect(envelope[0]).toMatchObject({
      canonical: { text: 'batch' },
      native: nativeTranscript,
    });
    await expect(loaded.transcribeBatch([new Float32Array(31 * 16_000)])).rejects.toMatchObject({
      code: 'not-implemented-speech-feature',
      details: { inputIndex: 0 },
    });

    await loaded.dispose();
    await runtime.dispose();
  });

  it('speech pipeline batches through cached and uncached model handles', async () => {
    const runtime = createSpeechRuntime();
    const backend = createStaticBackend({
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
    });
    let batchCalls = 0;
    let modelCreates = 0;
    const nativeTranscript = { utteranceText: 'pipeline-batch', isFinal: true };
    const model = {
      info: {
        family: 'batch-test',
        modelId: 'batch-test-model',
        classification: { ecosystem: 'test', task: 'asr' },
        inference: {
          sampleRate: 16000,
          maxInputDurationSec: 30,
          autoWindowThresholdSec: 30,
          supportsWordTimestamps: false,
          supportsSegmentTimestamps: false,
          defaultSegmentationStrategy: 'none' as const,
          defaultMergeStrategy: 'concat' as const,
        },
      },
      backend,
      async createSession() {
        return {
          async transcribe() {
            return {
              text: 'single',
              warnings: [],
              meta: { detailLevel: 'text' as const, isFinal: true },
            };
          },
          async transcribeBatch(
            inputs: readonly unknown[],
            options?: { readonly responseFlavor?: string },
          ) {
            batchCalls += 1;
            if (options?.responseFlavor === 'native') return inputs.map(() => nativeTranscript);
            return inputs.map(() => ({
              text: 'pipeline-batch',
              warnings: [],
              meta: { detailLevel: 'text' as const, isFinal: true },
            }));
          },
          dispose() {},
        };
      },
      dispose() {},
    };
    runtime.registerBackend(backend);
    runtime.registerModelFamily({
      family: 'batch-test',
      supports(modelId: string) {
        return modelId === 'batch-test-model';
      },
      async createModel() {
        modelCreates += 1;
        return model;
      },
    });

    const pipeline = createSpeechPipeline({ runtime, cacheModels: true });
    const request = { family: 'batch-test', modelId: 'batch-test-model', backend: 'test' };
    const first = await pipeline.transcribeBatch(
      [new Float32Array(16000), new Float32Array(8000)],
      request,
    );
    const second = await pipeline.transcribeBatch([new Float32Array(4000)], request);

    expect(first.map((item) => item.text)).toEqual(['pipeline-batch', 'pipeline-batch']);
    expect(second[0]?.text).toBe('pipeline-batch');
    expect(batchCalls).toBe(2);
    expect(modelCreates).toBe(1);
    expect(pipeline.listLoadedModels()).toHaveLength(1);

    await pipeline.dispose();

    const uncachedPipeline = createSpeechPipeline({ runtime, cacheModels: false });
    const uncached = await uncachedPipeline.transcribeBatch([new Float32Array(16000)], request);
    expect(uncached[0]?.text).toBe('pipeline-batch');
    expect(batchCalls).toBe(3);
    expect(modelCreates).toBe(2);
    expect(uncachedPipeline.listLoadedModels()).toHaveLength(0);
    await uncachedPipeline.dispose();
    await runtime.dispose();
  });

  it('loaded model handles window long raw mono PCM using model inference limits', async () => {
    const runtime = createSpeechRuntime();
    const backend = createStaticBackend({
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
    });
    const calls: number[] = [];
    const session = {
      async transcribe(input: { readonly durationSeconds: number }) {
        calls.push(input.durationSeconds);
        return {
          text: 'window',
          warnings: [],
          meta: {
            detailLevel: 'words' as const,
            isFinal: true,
            durationSeconds: input.durationSeconds,
          },
          words: [{ index: 0, text: 'window', startTime: 0, endTime: 0.5 }],
        };
      },
      dispose() {
        return undefined;
      },
    };
    const model = {
      info: {
        family: 'test-family',
        modelId: 'test-long-audio',
        classification: { ecosystem: 'test', task: 'asr' },
        inference: {
          sampleRate: 16_000,
          maxInputDurationSec: 2,
          recommendedWindowDurationSec: 2,
          minWindowDurationSec: 1,
          maxWindowDurationSec: 2,
          autoWindowThresholdSec: 2,
          defaultOverlapSec: 0.5,
          supportsWordTimestamps: true,
          supportsSegmentTimestamps: true,
          defaultSegmentationStrategy: 'word-punctuation' as const,
          defaultMergeStrategy: 'word-dedupe' as const,
        },
      },
      backend,
      async createSession() {
        return session;
      },
      dispose() {
        return undefined;
      },
    };
    const builtInHandle = {
      runtime,
      model,
      session,
      async transcribe(
        input: { readonly durationSeconds: number },
        options?: { readonly responseFlavor?: string },
      ) {
        return session.transcribe(input, options);
      },
      async dispose() {
        return undefined;
      },
    } satisfies BuiltInSpeechModelHandle;
    const loaded = createLoadedSpeechModelHandle(builtInHandle);

    const transcript = await loaded.transcribeMonoPcm(new Float32Array(4 * 16_000), 16_000);

    expect(calls.length).toBeGreaterThan(1);
    expect(calls.every((duration) => duration <= 2)).toBe(true);
    expect(transcript.text).toContain('window');

    await loaded.dispose();
    await runtime.dispose();
  });

  it('speech pipeline caches and reuses loaded models across transcriptions', async () => {
    const runtime = createRuntime();
    const pipeline = createSpeechPipeline({
      runtime,
      useManifestSources: false,
    });

    const handleA = await pipeline.loadModel({
      modelId: 'parakeet-tdt-0.6b-v3',
      backend: 'wasm',
    });
    const handleB = await pipeline.loadModel({
      modelId: 'parakeet-tdt-0.6b-v3',
      backend: 'wasm',
    });
    const transcript = await pipeline.transcribe(new Float32Array(16000), {
      modelId: 'parakeet-tdt-0.6b-v3',
      backend: 'wasm',
      transcribeOptions: {
        responseFlavor: 'canonical',
      },
    });

    expect(handleA).toBe(handleB);
    expect(transcript.text.length).toBeGreaterThan(0);
    expect(pipeline.listLoadedModels().length).toBe(1);

    await pipeline.dispose();
    await runtime.dispose();
  });

  it('speech pipeline transcribes explicit-rate raw mono PCM directly', async () => {
    const runtime = createRuntime();
    const pipeline = createSpeechPipeline({
      runtime,
      useManifestSources: false,
    });

    const transcript = await pipeline.transcribeMonoPcm(new Float32Array(16000), 16000, {
      modelId: 'parakeet-tdt-0.6b-v3',
      backend: 'wasm',
      transcribeOptions: {
        responseFlavor: 'canonical',
      },
    });

    expect(transcript.text.length).toBeGreaterThan(0);
    expect(pipeline.listLoadedModels().length).toBe(1);

    await pipeline.dispose();
    await runtime.dispose();
  });

  it('supports forceReload when callers need to refresh cached model state', async () => {
    const runtime = createRuntime();
    const pipeline = createSpeechPipeline({
      runtime,
      useManifestSources: false,
    });

    const handleA = await pipeline.loadModel({
      modelId: 'parakeet-tdt-0.6b-v3',
      backend: 'wasm',
    });
    const handleB = await pipeline.loadModel({
      modelId: 'parakeet-tdt-0.6b-v3',
      backend: 'wasm',
      forceReload: true,
    });

    expect(handleA).not.toBe(handleB);
    expect(pipeline.listLoadedModels().length).toBe(1);

    await pipeline.dispose();
    await runtime.dispose();
  });

  it('treats non-serializable load requests as uncached and still transcribes safely', async () => {
    const runtime = createRuntime();
    const pipeline = createSpeechPipeline({
      runtime,
      useManifestSources: false,
    });

    const transcript = await pipeline.transcribe(new Float32Array(16000), {
      family: 'nemo-tdt',
      modelId: 'nemo-fastconformer-tdt-scaffold',
      backend: 'wasm',
      classification: {
        ecosystem: 'nemo',
        encoder: 'fastconformer',
        decoder: 'tdt',
        topology: 'tdt',
        task: 'asr',
      },
      options: {
        // A function makes the request non-serializable for automatic cache keys.
        marker() {
          return 'uncacheable';
        },
      } as unknown,
      transcribeOptions: {
        responseFlavor: 'canonical',
      },
    });

    expect(transcript.text.length).toBeGreaterThan(0);
    expect(pipeline.listLoadedModels().length).toBe(0);

    await pipeline.dispose();
    await runtime.dispose();
  });

  it('honors options.signal on loadSpeechModel before creating a session', async () => {
    const progressEvents: Array<{ phase: string; aborted?: boolean }> = [];
    await expect(
      loadSpeechModel({
        modelId: 'parakeet-tdt-0.6b-v3',
        backend: 'wasm',
        signal: { aborted: true },
        onProgress(event) {
          progressEvents.push({ phase: event.phase, aborted: event.aborted });
        },
      }),
    ).rejects.toBeInstanceOf(PipelineAbortedError);
    expect(progressEvents.map((event) => event.phase)).toEqual(['cancelled']);
    expect(progressEvents.at(-1)?.aborted).toBe(true);
    expect(progressEvents.some((event) => event.phase === 'ready')).toBe(false);
  });
});
