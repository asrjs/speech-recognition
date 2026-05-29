import { loadSpeechModel, type BackendCapabilities, type ExecutionBackend } from '@asrjs/speech-recognition';
import { registerBuiltInModelFamilies, registerBuiltInPresets } from '@asrjs/speech-recognition/builtins';
import { createSpeechRuntime } from '@asrjs/speech-recognition';
import {
  createWav2Vec2ModelFamily,
  type Wav2Vec2ModelOptions,
  type Wav2Vec2NativeTranscript,
  type Wav2Vec2TranscriptionOptions,
} from '@asrjs/speech-recognition/models/wav2vec2';
import {
  createWav2Vec2PresetFactory,
  resolveWav2Vec2PresetManifest,
} from '@asrjs/speech-recognition/presets/wav2vec2';
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
  return runtime;
}

describe('Wav2Vec2 model family', () => {
  it('matches Wav2Vec2 model IDs and CTC classification', () => {
    const family = createWav2Vec2ModelFamily();

    expect(family.family).toBe('wav2vec2');
    expect(family.supports('facebook/wav2vec2-base-960h')).toBe(true);
    expect(family.supports('wav2vec2-base-960h')).toBe(true);
    expect(family.supports('openai/whisper-base')).toBe(false);
    expect(family.matchesClassification?.({ ecosystem: 'meta', topology: 'ctc' })).toBe(true);
    expect(family.matchesClassification?.({ ecosystem: 'openai', topology: 'seq2seq' })).toBe(false);
  });

  it('loads through the technical family and maps native Wav2Vec2 output to canonical output', async () => {
    const runtime = createRuntime();
    runtime.registerModelFamily(createWav2Vec2ModelFamily());

    const loaded = await loadSpeechModel<
      Wav2Vec2ModelOptions,
      Wav2Vec2TranscriptionOptions,
      Wav2Vec2NativeTranscript
    >({
      runtime,
      family: 'wav2vec2',
      modelId: 'facebook/wav2vec2-base-960h',
      backend: 'wasm',
    });

    const result = await loaded.transcribe(new Float32Array(16000), {
      detail: 'detailed',
      responseFlavor: 'canonical+native',
      returnTokenIds: true,
      returnConfidence: true,
    });

    expect(loaded.model.info.family).toBe('wav2vec2');
    expect(loaded.model.info.classification).toMatchObject({
      ecosystem: 'meta',
      processor: 'wav2vec2-conv',
      encoder: 'wav2vec2-conformer',
      decoder: 'ctc',
      topology: 'ctc',
      task: 'asr',
    });
    expect(loaded.model.info.nativeOutputName).toBe('Wav2Vec2NativeTranscript');
    expect(result.canonical.text).toBe('wav2vec2 ctc scaffold');
    expect(result.canonical.meta).toMatchObject({
      modelFamily: 'wav2vec2',
      modelId: 'facebook/wav2vec2-base-960h',
      backendId: 'wasm',
      language: 'en',
      nativeAvailable: true,
    });
    expect(result.canonical.words?.map((word) => word.text)).toEqual(['wav2vec2', 'ctc', 'scaffold']);
    expect(result.native?.warnings?.[0]?.code).toBe('wav2vec2.stubbed-decoder');

    await loaded.dispose();
    await runtime.dispose();
  });
});

describe('Wav2Vec2 preset', () => {
  it('resolves the base-960h preset by canonical ID and aliases', () => {
    const manifest = resolveWav2Vec2PresetManifest('facebook/wav2vec2-base-960h');

    expect(manifest?.preset).toBe('wav2vec2');
    expect(manifest?.modelId).toBe('facebook/wav2vec2-base-960h');
    expect(manifest?.config).toMatchObject({
      sampleRate: 16000,
      outputStride: 320,
      vocabularySize: 32,
      ctcBlankId: 0,
      languages: ['en'],
    });
    expect(resolveWav2Vec2PresetManifest('wav2vec2-base-960h')).toBe(manifest);
    expect(resolveWav2Vec2PresetManifest('wav2vec2')).toBe(manifest);
    expect(resolveWav2Vec2PresetManifest('unknown/model')).toBeUndefined();
  });

  it('loads through the branded preset without forcing an unpublished manifest source', async () => {
    const runtime = createRuntime();
    registerBuiltInModelFamilies(runtime);
    runtime.registerPreset(
      createWav2Vec2PresetFactory({
        useManifestSource: false,
      }),
    );

    const loaded = await loadSpeechModel<
      Wav2Vec2ModelOptions,
      Wav2Vec2TranscriptionOptions,
      Wav2Vec2NativeTranscript
    >({
      runtime,
      preset: 'wav2vec2',
      modelId: 'wav2vec2-base-960h',
      backend: 'wasm',
    });

    const result = await loaded.transcribe(new Float32Array(16000), {
      responseFlavor: 'canonical+native',
    });

    expect(loaded.model.info.preset).toBe('wav2vec2');
    expect(loaded.model.info.family).toBe('wav2vec2');
    expect(loaded.model.loadOptions?.source).toBeUndefined();
    expect(result.canonical.text).toBe('wav2vec2 ctc scaffold');
    expect(result.native?.warnings?.[0]?.code).toBe('wav2vec2.stubbed-decoder');

    await loaded.dispose();
    await runtime.dispose();
  });

  it('is registered by the built-in runtime helpers', async () => {
    const runtime = createRuntime();
    registerBuiltInModelFamilies(runtime);
    registerBuiltInPresets(runtime, {
      useManifestSources: false,
    });

    const loaded = await loadSpeechModel<
      Wav2Vec2ModelOptions,
      Wav2Vec2TranscriptionOptions,
      Wav2Vec2NativeTranscript
    >({
      runtime,
      modelId: 'facebook/wav2vec2-base-960h',
      backend: 'wasm',
    });

    expect(loaded.model.info.preset).toBe('wav2vec2');
    expect(loaded.model.info.family).toBe('wav2vec2');

    await loaded.dispose();
    await runtime.dispose();
  });
});
