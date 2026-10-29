import { createSpeechRuntime, loadSpeechModel } from '@asrjs/speech-recognition';
import type {
  BackendCapabilities,
  ExecutionBackend,
  RuntimeProgressEvent,
  TranscriptionProgressEvent,
} from '@asrjs/speech-recognition';
import {
  loadBuiltInSpeechModel,
  registerBuiltInModelFamilies,
  registerBuiltInPresets,
} from '@asrjs/speech-recognition/builtins';
import { createCanaryPresetFactory } from '@asrjs/speech-recognition/presets/canary';
import { createParakeetPresetFactory } from '@asrjs/speech-recognition/presets/parakeet';
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
        dispose() {
          return undefined;
        },
      };
    },
  };
}

describe('loadBuiltInSpeechModel', () => {
  it('automatically resolves the approved Parakeet profile from probed backend precisions', async () => {
    const runtime = createSpeechRuntime();
    const cap = (id: 'webgpu' | 'wasm', available: boolean, supportedPrecisions: ('fp32' | 'fp16' | 'int8')[]) => ({
      id,
      displayName: id,
      available,
      priority: id === 'webgpu' ? 100 : 60,
      environments: ['browser', 'node'] as const,
      acceleration: id === 'webgpu' ? ['gpu'] as const : ['cpu'] as const,
      supportedPrecisions,
      supportsFp16: supportedPrecisions.includes('fp16'),
      supportsInt8: supportedPrecisions.includes('int8'),
      supportsSharedArrayBuffer: true,
      requiresSharedArrayBuffer: false,
      fallbackSuitable: true,
      notes: [],
    });
    runtime.registerBackend(createStaticBackend(cap('webgpu', true, ['fp32', 'fp16', 'int8'])));
    runtime.registerBackend(createStaticBackend(cap('wasm', true, ['fp32', 'int8'])));
    registerBuiltInModelFamilies(runtime);
    registerBuiltInPresets(runtime, { useManifestSources: false });
    const sentinel = new Error('load intercepted');
    let request: any;
    vi.spyOn(runtime, 'loadModel').mockImplementation(async (input: any) => {
      request = input;
      throw sentinel;
    });

    await expect(loadBuiltInSpeechModel({ runtime, modelId: 'parakeet-tdt-0.6b-v3' })).rejects.toBe(sentinel);
    expect(request.options.source).toMatchObject({
      revision: 'f88260fa0777fe0868dda6df85d1a98f012a4a7a',
      encoderBackend: 'webgpu', decoderBackend: 'wasm',
      encoderQuant: 'fp16', decoderQuant: 'int8',
    });
  });

  it('keeps the pinned manifest revision when an explicit backend is requested', async () => {
    const runtime = createSpeechRuntime();
    const cap = (id: 'webgpu' | 'wasm', available: boolean, supportedPrecisions: ('fp32' | 'fp16' | 'int8')[]) => ({
      id, displayName: id, available, priority: id === 'webgpu' ? 100 : 60,
      environments: ['browser', 'node'] as const,
      acceleration: id === 'webgpu' ? ['gpu'] as const : ['cpu'] as const,
      supportedPrecisions, supportsFp16: supportedPrecisions.includes('fp16'),
      supportsInt8: supportedPrecisions.includes('int8'), supportsSharedArrayBuffer: true,
      requiresSharedArrayBuffer: false, fallbackSuitable: true, notes: [],
    });
    runtime.registerBackend(createStaticBackend(cap('webgpu', true, ['fp32', 'fp16', 'int8'])));
    runtime.registerBackend(createStaticBackend(cap('wasm', true, ['fp32', 'int8'])));
    registerBuiltInModelFamilies(runtime);
    registerBuiltInPresets(runtime, { useManifestSources: false });
    const sentinel = new Error('load intercepted');
    let request: any;
    vi.spyOn(runtime, 'loadModel').mockImplementation(async (input: any) => {
      request = input;
      throw sentinel;
    });

    await expect(
      loadBuiltInSpeechModel({ runtime, modelId: 'parakeet-tdt-0.6b-v3', backend: 'wasm' }),
    ).rejects.toBe(sentinel);
    expect(request.backend).toBe('wasm');
    expect(request.options.source).toMatchObject({
      revision: 'f88260fa0777fe0868dda6df85d1a98f012a4a7a',
      encoderBackend: 'wasm', decoderBackend: 'wasm',
    });

    await expect(
      loadBuiltInSpeechModel({ runtime, modelId: 'parakeet-tdt-0.6b-v3', backend: 'webgpu-strict' }),
    ).rejects.toBe(sentinel);
    expect(request.backend).toBe('webgpu');
    // The v3 decoder is declared WASM-only, so the strict plan is unsupported;
    // the revision must still stay pinned while components degrade.
    expect(request.options.source).toMatchObject({
      revision: 'f88260fa0777fe0868dda6df85d1a98f012a4a7a',
      encoderBackend: 'webgpu', decoderBackend: 'wasm',
    });
  });

  it('falls back when WebGPU is absent and keeps direct artifact requests untouched', async () => {
    const runtime = createSpeechRuntime();
    const cap = (id: 'webgpu' | 'wasm', available: boolean, supportedPrecisions: ('fp32' | 'fp16' | 'int8')[]) => ({
      id, displayName: id, available, priority: id === 'webgpu' ? 100 : 60,
      environments: ['browser', 'node'] as const,
      acceleration: id === 'webgpu' ? ['gpu'] as const : ['cpu'] as const,
      supportedPrecisions, supportsFp16: supportedPrecisions.includes('fp16'),
      supportsInt8: supportedPrecisions.includes('int8'), supportsSharedArrayBuffer: true,
      requiresSharedArrayBuffer: false, fallbackSuitable: true, notes: [],
    });
    runtime.registerBackend(createStaticBackend(cap('webgpu', false, [])));
    runtime.registerBackend(createStaticBackend(cap('wasm', true, ['fp32', 'int8'])));
    registerBuiltInModelFamilies(runtime);
    registerBuiltInPresets(runtime, { useManifestSources: false });
    const sentinel = new Error('load intercepted');
    let request: any;
    vi.spyOn(runtime, 'loadModel').mockImplementation(async (input: any) => {
      request = input;
      throw sentinel;
    });

    await expect(loadBuiltInSpeechModel({ runtime, modelId: 'parakeet-tdt-0.6b-v3' })).rejects.toBe(sentinel);
    expect(request.options.source).toMatchObject({ encoderBackend: 'wasm', decoderBackend: 'wasm' });
    expect(request.options.source.encoderQuant).toBe('fp32');

    await expect(loadBuiltInSpeechModel({
      runtime,
      modelId: 'parakeet-tdt-0.6b-v3',
      useManifestSources: false,
    })).rejects.toBe(sentinel);
    expect(request.options?.source).toBeUndefined();

    const directOptions = { source: { kind: 'direct', encoderUrl: 'local/encoder.onnx' } };
    await expect(loadBuiltInSpeechModel({
      runtime,
      modelId: 'parakeet-tdt-0.6b-v3',
      options: directOptions as never,
    })).rejects.toBe(sentinel);
    expect(request.options).toEqual(directOptions);
  });

  it('registers verified Whisper manifest sources by default and preserves the opt-out', async () => {
    const runtime = createSpeechRuntime();
    registerBuiltInPresets(runtime);

    const whisper = runtime.listPresets().find((preset) => preset.preset === 'whisper');
    expect(whisper).toBeDefined();
    const resolved = await whisper!.resolveModelRequest(
      { modelId: 'openai/whisper-base' },
      {} as never,
    );
    expect(resolved.options?.source?.kind).toBe('huggingface');

    const optedOutRuntime = createSpeechRuntime();
    registerBuiltInPresets(optedOutRuntime, { useManifestSources: false });
    const optedOutWhisper = optedOutRuntime
      .listPresets()
      .find((preset) => preset.preset === 'whisper');
    const optedOutResolved = await optedOutWhisper!.resolveModelRequest(
      { modelId: 'openai/whisper-base' },
      {} as never,
    );
    expect(optedOutResolved.options?.source).toBeUndefined();
  });

  it('infers the Parakeet preset from modelId and returns a ready session-backed handle', async () => {
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
    runtime.registerBackend(
      createStaticBackend({
        id: 'wasm',
        displayName: 'WASM',
        available: true,
        priority: 60,
        environments: ['browser'],
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
    runtime.registerPreset(
      createParakeetPresetFactory({
        useManifestSource: false,
      }),
    );

    const progressEvents: RuntimeProgressEvent[] = [];
    const loaded = await loadSpeechModel({
      runtime,
      modelId: 'parakeet-tdt-0.6b-v3',
      backend: 'wasm',
      useManifestSources: false,
      onProgress(event) {
        progressEvents.push(event);
      },
    });
    const transcriptionEvents: TranscriptionProgressEvent[] = [];

    const result = await loaded.transcribe(new Float32Array(16000), {
      detail: 'segments',
      responseFlavor: 'canonical+native',
      onProgress(event) {
        transcriptionEvents.push(event);
      },
    });

    expect(loaded.model.info.preset).toBe('parakeet');
    expect(loaded.model.info.family).toBe('nemo-tdt');
    expect(result.canonical.text.length).toBeGreaterThan(0);
    expect(result.native?.warnings?.[0]?.code).toBe('nemo-tdt.stubbed-decoder');
    expect(progressEvents.map((event) => event.phase)).toEqual([
      'resolve:start',
      'resolve:complete',
      'model-load:start',
      'model-load:complete',
      'session-create:start',
      'session-create:complete',
      'ready',
    ]);
    expect(transcriptionEvents.map((event) => event.stage)).toEqual(['start', 'complete']);
    expect(transcriptionEvents[1]?.metrics?.totalMs).toBeGreaterThan(0);

    await loaded.dispose();
  });

  it('infers the Canary preset from modelId and returns a ready session-backed handle', async () => {
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
    runtime.registerPreset(
      createCanaryPresetFactory({
        useManifestSource: false,
      }),
    );

    const loaded = await loadSpeechModel({
      runtime,
      modelId: 'nvidia/canary-180m-flash',
      backend: 'wasm',
    });

    expect(loaded.model.info.preset).toBe('canary');
    expect(loaded.model.info.family).toBe('nemo-aed');

    await loaded.dispose();
  });

  it('supports explicit family loading when the consumer wants the technical implementation directly', async () => {
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

    const loaded = await loadBuiltInSpeechModel({
      runtime,
      family: 'nemo-tdt',
      modelId: 'nemo-fastconformer-tdt-scaffold',
      backend: 'wasm',
      classification: {
        ecosystem: 'nemo',
        encoder: 'fastconformer',
        decoder: 'tdt',
        task: 'asr',
      },
    });

    expect(loaded.model.info.family).toBe('nemo-tdt');
    expect(loaded.model.info.preset).toBeUndefined();

    await loaded.dispose();
  });

  it('forwards runtime model-load events through onProgress when it creates its own runtime', async () => {
    const progressEvents: RuntimeProgressEvent[] = [];

    const loaded = await loadBuiltInSpeechModel({
      family: 'nemo-tdt',
      modelId: 'nemo-fastconformer-tdt-scaffold',
      backend: 'wasm',
      classification: {
        ecosystem: 'nemo',
        encoder: 'fastconformer',
        decoder: 'tdt',
        task: 'asr',
      },
      onProgress(event) {
        progressEvents.push(event);
      },
    });

    expect(progressEvents.map((event) => event.phase)).toEqual([
      'resolve:start',
      'resolve:complete',
      'model-load:start',
      'model-load:complete',
      'session-create:start',
      'session-create:complete',
      'ready',
    ]);

    await loaded.dispose();
  });

  it('treats webgpu-hybrid as a backend preference instead of a literal backend id', async () => {
    const runtime = createSpeechRuntime();
    runtime.registerBackend(
      createStaticBackend({
        id: 'webgpu',
        displayName: 'WebGPU',
        available: true,
        priority: 100,
        environments: ['browser'],
        acceleration: ['gpu'],
        supportedPrecisions: ['fp32', 'fp16', 'int8'],
        supportsFp16: true,
        supportsInt8: true,
        supportsSharedArrayBuffer: true,
        requiresSharedArrayBuffer: false,
        fallbackSuitable: true,
        notes: [],
      }),
    );
    runtime.registerBackend(
      createStaticBackend({
        id: 'wasm',
        displayName: 'WASM',
        available: true,
        priority: 60,
        environments: ['browser'],
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

    const loaded = await loadBuiltInSpeechModel({
      runtime,
      modelId: 'parakeet-tdt-0.6b-v3',
      backend: 'webgpu-hybrid',
      useManifestSources: false,
    });

    expect(loaded.model.backend.id).toBe('webgpu');

    await loaded.dispose();
  });
});
