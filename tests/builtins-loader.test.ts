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
import { createParakeetPresetFactory } from '@asrjs/speech-recognition/presets/parakeet';
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

describe('loadBuiltInSpeechModel', () => {
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
    registerBuiltInModelFamilies(runtime);
    registerBuiltInPresets(runtime, {
      useManifestSources: false,
    });

    const loaded = await loadBuiltInSpeechModel({
      runtime,
      modelId: 'parakeet-tdt-0.6b-v3',
      backend: 'webgpu-hybrid',
    });

    expect(loaded.model.backend.id).toBe('webgpu');

    await loaded.dispose();
  });
});
