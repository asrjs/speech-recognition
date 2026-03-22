import {
  createSpeechPipeline,
  createSpeechRuntime,
  transcribeSpeech,
  type BackendCapabilities,
  type ExecutionBackend,
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
});
