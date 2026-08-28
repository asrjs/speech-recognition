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
import { registerBuiltInModelFamilies, registerBuiltInPresets } from '@asrjs/speech-recognition/builtins';
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

    await loaded.dispose();
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
