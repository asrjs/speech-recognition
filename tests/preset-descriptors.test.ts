import {
  buildBuiltInHubLoadOptions,
  buildBuiltInTranscriptionOptions,
  detectBuiltInModelQuantizationsFromFiles,
  getBuiltInModelDescriptor,
  getBuiltInModelLanguageName,
  listBuiltInModelDescriptors,
  listBuiltInModelOptions,
  resolveBuiltInModelComponentBackends,
  type BuiltInModelDescriptor,
} from '@asrjs/speech-recognition/presets';
import { describe, expect, expectTypeOf, it } from 'vitest';

describe('built-in preset descriptors', () => {
  it('lists built-in models with consolidated metadata', () => {
    const descriptors = listBuiltInModelDescriptors();
    const options = listBuiltInModelOptions();

    expectTypeOf(descriptors).toEqualTypeOf<readonly BuiltInModelDescriptor[]>();
    expect(descriptors.length).toBeGreaterThan(0);
    expect(descriptors.map((descriptor) => descriptor.modelId)).toEqual(
      expect.arrayContaining([
        'parakeet-tdt-0.6b-v2',
        'parakeet-tdt-0.6b-v3',
        'nvidia/canary-180m-flash',
        'google/medasr',
        'facebook/wav2vec2-base-960h',
      ]),
    );
    expect(options.find((option) => option.key === 'nvidia/canary-180m-flash')?.preset).toBe(
      'canary',
    );
  });

  it('exposes Canary capabilities, controls, and loading defaults', () => {
    const descriptor = getBuiltInModelDescriptor('canary-180m-flash');

    expect(descriptor?.preset).toBe('canary');
    expect(descriptor?.capabilities.supportsTranslation).toBe(true);
    expect(descriptor?.capabilities.supportsPunctuationCapitalization).toBe(true);
    expect(descriptor?.loading.encoderArtifactBaseName).toBe('encoder-model');
    expect(descriptor?.loading.decoderArtifactBaseName).toBe('decoder-model');
    expect(descriptor?.loading.availableEncoderBackends).toEqual(['webgpu', 'wasm']);
    expect(descriptor?.loading.availableDecoderBackends).toEqual(['webgpu', 'wasm']);
    expect(descriptor?.loading.defaultEncoderBackend).toBe('webgpu');
    expect(descriptor?.loading.defaultDecoderBackend).toBe('wasm');
    expect(descriptor?.loading.availableEncoderQuantizations).toEqual(['fp16', 'int8', 'fp32']);
    expect(descriptor?.loading.availableDecoderQuantizations).toEqual(['fp16', 'int8', 'fp32']);
    expect(descriptor?.loading.defaultEncoderQuantization).toBe('fp32');
    expect(descriptor?.loading.defaultDecoderQuantization).toBe('int8');
    expect(descriptor?.controls.map((control) => control.key)).toEqual([
      'task',
      'targetLanguage',
      'punctuate',
      'timestamps',
    ]);
  });

  it('exposes split quantization metadata for Parakeet too', () => {
    const descriptor = getBuiltInModelDescriptor('parakeet-tdt-0.6b-v2');

    expect(descriptor?.loading.availableEncoderBackends).toEqual(['webgpu', 'wasm']);
    expect(descriptor?.loading.availableDecoderBackends).toEqual(['wasm']);
    expect(descriptor?.loading.defaultEncoderBackend).toBe('webgpu');
    expect(descriptor?.loading.defaultDecoderBackend).toBe('wasm');
    expect(descriptor?.loading.availableEncoderQuantizations).toEqual(['fp16', 'int8', 'fp32']);
    expect(descriptor?.loading.availableDecoderQuantizations).toEqual(['fp16', 'int8', 'fp32']);
    expect(descriptor?.loading.defaultEncoderQuantization).toBe('fp16');
    expect(descriptor?.loading.defaultDecoderQuantization).toBe('int8');
    expect(descriptor?.loading.defaultRevision).toBe('main');
  });

  it('builds preset-aware hub load requests from a single generic helper', () => {
    const parakeet = buildBuiltInHubLoadOptions({
      modelId: 'parakeet-tdt-0.6b-v3',
      revision: 'feat/fp16-canonical-v3',
      backend: 'webgpu-hybrid',
      encoderQuant: 'fp16',
      decoderQuant: 'int8',
      preprocessorName: 'nemo128',
      preprocessorBackend: 'js',
    });
    const canary = buildBuiltInHubLoadOptions({
      modelId: 'nvidia/canary-180m-flash',
      backend: 'wasm',
      encoderQuant: 'fp32',
      decoderQuant: 'int8',
      preprocessorBackend: 'js',
    });

    expect(parakeet).toMatchObject({
      preset: 'parakeet',
      modelId: 'parakeet-tdt-0.6b-v3',
      options: {
        source: {
          repoId: 'ysdede/parakeet-tdt-0.6b-v3-onnx',
          encoderQuant: 'fp16',
          decoderQuant: 'int8',
          preprocessorBackend: 'js',
        },
      },
    });
    expect(canary).toMatchObject({
      preset: 'canary',
      modelId: 'nvidia/canary-180m-flash',
      options: {
        source: {
          repoId: 'ysdede/canary-180m-flash-onnx',
          encoderBackend: 'wasm',
          decoderBackend: 'wasm',
          encoderQuant: 'fp32',
          decoderQuant: 'int8',
          preprocessorBackend: 'js',
        },
      },
    });
  });

  it('builds preset-aware transcription options with Canary prompt semantics', () => {
    const canaryAsr = buildBuiltInTranscriptionOptions('nvidia/canary-180m-flash', {
      sourceLanguage: 'de',
      task: 'asr',
      punctuate: false,
      timestamps: true,
      enableProfiling: true,
    });
    const canaryTranslate = buildBuiltInTranscriptionOptions('nvidia/canary-180m-flash', {
      sourceLanguage: 'fr',
      targetLanguage: 'en',
      task: 'translation',
      punctuate: true,
      timestamps: false,
    });
    const parakeet = buildBuiltInTranscriptionOptions('parakeet-tdt-0.6b-v2', {
      frameStride: 2,
      timestamps: true,
      returnConfidences: true,
      enableProfiling: true,
    });

    expect(canaryAsr).toEqual({
      sourceLanguage: 'de',
      targetLanguage: 'de',
      task: 'asr',
      pnc: 'no',
      timestamp: 'yes',
      enableProfiling: true,
    });
    expect(canaryTranslate).toEqual({
      sourceLanguage: 'fr',
      targetLanguage: 'en',
      task: 'translation',
      pnc: 'yes',
      timestamp: 'no',
      enableProfiling: undefined,
    });
    expect(parakeet).toEqual({
      returnTimestamps: true,
      returnConfidences: true,
      frameStride: 2,
      enableProfiling: true,
    });
  });

  it('exposes Wav2Vec2 CTC metadata with raw-waveform loading defaults', () => {
    const descriptor = getBuiltInModelDescriptor('wav2vec2-base-960h');

    expect(descriptor?.preset).toBe('wav2vec2');
    expect(descriptor?.modelId).toBe('facebook/wav2vec2-base-960h');
    expect(descriptor?.classification).toMatchObject({
      ecosystem: 'meta',
      processor: 'wav2vec2-conv',
      encoder: 'wav2vec2-conformer',
      decoder: 'ctc',
      topology: 'ctc',
      family: 'wav2vec2',
      task: 'asr',
    });
    expect(descriptor?.capabilities).toMatchObject({
      supportedTasks: ['asr'],
      supportsWordTimestamps: true,
      supportsSegmentTimestamps: true,
      supportsTranslation: false,
    });
    expect(descriptor?.inference).toMatchObject({
      sampleRate: 16000,
      supportsWordTimestamps: true,
      supportsTokenTimestamps: true,
      defaultSegmentationStrategy: 'ctc-frame',
      defaultMergeStrategy: 'ctc-collapse',
    });
    expect(descriptor?.loading).toMatchObject({
      supportsHubSource: false,
      supportsLocalSource: true,
      defaultEncoderBackend: 'wasm',
      defaultDecoderBackend: 'wasm',
      encoderArtifactBaseName: 'model',
      decoderArtifactBaseName: 'model',
    });
  });

  it('resolves human-readable language names across preset families', () => {
    expect(getBuiltInModelLanguageName('zh')).toBe('Chinese');
    expect(getBuiltInModelLanguageName('fr')).toBe('French');
    expect(getBuiltInModelLanguageName('auto')).toBe('Auto-detect');
  });

  it('resolves model-specific component backend defaults and restrictions', () => {
    expect(resolveBuiltInModelComponentBackends('parakeet-tdt-0.6b-v2')).toEqual({
      encoderBackend: 'webgpu',
      decoderBackend: 'wasm',
    });

    expect(
      resolveBuiltInModelComponentBackends('parakeet-tdt-0.6b-v2', {
        encoderBackend: 'wasm',
        decoderBackend: 'wasm',
      }),
    ).toEqual({
      encoderBackend: 'wasm',
      decoderBackend: 'wasm',
    });

    expect(
      resolveBuiltInModelComponentBackends('parakeet-tdt-0.6b-v2', {
        backend: 'webgpu-hybrid',
      }),
    ).toEqual({
      encoderBackend: 'webgpu',
      decoderBackend: 'wasm',
    });

    expect(
      resolveBuiltInModelComponentBackends('nvidia/canary-180m-flash', {
        backend: 'webgpu-strict',
      }),
    ).toEqual({
      encoderBackend: 'webgpu',
      decoderBackend: 'webgpu',
    });

    expect(() => resolveBuiltInModelComponentBackends('unknown-model-id')).toThrow(
      'Unknown built-in model "unknown-model-id".',
    );
  });

  it('detects quantization variants from repo filenames using built-in artifact names', () => {
    const detected = detectBuiltInModelQuantizationsFromFiles('nvidia/canary-180m-flash', [
      'encoder-model.onnx',
      'encoder-model.fp16.onnx',
      'decoder-model.int8.onnx',
      'decoder-model.onnx',
      'tokenizer.json',
    ]);

    expect(detected.encoderArtifactBaseName).toBe('encoder-model');
    expect(detected.decoderArtifactBaseName).toBe('decoder-model');
    expect(detected.encoder).toEqual(['fp16', 'fp32']);
    expect(detected.decoder).toEqual(['int8', 'fp32']);
  });
});
