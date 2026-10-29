import { describe, expect, it } from 'vitest';
import {
  listSpeechModels,
  listSpeechModelOptions,
  getSpeechModelDescriptor,
  buildSpeechModelLoadOptions,
  resolveSpeechModelComponentBackends,
  getSpeechModelLanguageName,
} from '../src/index.js';
import { createBuiltInSpeechRuntime } from '../src/runtime/builtins.js';
import type { BuiltInPresetId } from '../src/presets/descriptors.js';

const EXPECTED_PRESET_IDS: readonly BuiltInPresetId[] = [
  'parakeet',
  'canary',
  'medasr',
  'nemotron',
  'whisper',
  'wav2vec2',
  'wav2vec2-tr',
  'moonshine',
  'qwen3-asr',
  'parakeet-ctc',
  'omnilingual-ctc',
  'gigaam-ctc',
  'gigaam-rnnt',
  'sensevoice',
  'x-asr',
];

const EXPECTED_MODEL_FAMILIES = [
  'lasr-ctc',
  'gigaam-ctc',
  'parakeet-ctc',
  'omnilingual-ctc',
  'gigaam-rnnt',
  'nemo-aed',
  'nemo-rnnt',
  'nemo-tdt',
  'nemotron-rnnt',
  'whisper-seq2seq',
  'qwen-asr',
  'sensevoice',
  'x-asr',
  'wav2vec2',
  'moonshine',
  'qwen3-asr',
];

describe('Catalog & Preset Resolution', () => {
  const models = listSpeechModels();
  const options = listSpeechModelOptions();

  it('lists all built-in catalog models with required metadata', () => {
    expect(models.length).toBeGreaterThanOrEqual(17);

    for (const model of models) {
      expect(model.modelId).toBeTruthy();
      expect(model.preset).toBeTruthy();
      expect(EXPECTED_PRESET_IDS).toContain(model.preset);
      expect(model.displayName).toBeTruthy();
      expect(model.description).toBeTruthy();
      expect(model.capabilities.supportedTasks).toContain('asr');
      expect(Array.isArray(model.languages)).toBe(true);
      expect(model.languages.length).toBeGreaterThan(0);
    }
  });

  it('resolves every model descriptor by modelId and aliases', () => {
    for (const model of models) {
      const resolved = getSpeechModelDescriptor(model.modelId);
      expect(resolved).not.toBeNull();
      expect(resolved?.modelId).toBe(model.modelId);
      expect(resolved?.preset).toBe(model.preset);

      if (model.aliases) {
        for (const alias of model.aliases) {
          const aliasResolved = getSpeechModelDescriptor(alias);
          expect(aliasResolved).not.toBeNull();
          expect(aliasResolved?.modelId).toBe(model.modelId);
        }
      }
    }
    const streamingMoonshine = getSpeechModelDescriptor('moonshine-streaming-tiny');
    expect(streamingMoonshine).toMatchObject({
      modelId: 'moonshine-ai/moonshine-streaming-tiny',
      status: 'experimental',
      loading: {
        repoId: 'ysdede/moonshine-streaming-tiny-onnx',
        supportsHubSource: true,
        supportsLocalSource: true,
        supportsSplitBackends: false,
        defaultRevision: 'cb347a0760c2d7d6cf9083eb770376ba108e5b31',
      },
      warmup: { mode: 'expected-text' },
    });
    expect(streamingMoonshine?.loading.availableEncoderBackends).toEqual(['wasm', 'webgpu']);
    expect(streamingMoonshine?.loading.availableDecoderBackends).toEqual(['wasm', 'webgpu']);
    expect(streamingMoonshine?.capabilities.supportsWordTimestamps).toBe(false);
    expect(streamingMoonshine?.capabilities.supportsSegmentTimestamps).toBe(false);
    expect(streamingMoonshine?.inference).toMatchObject({
      supportsWordTimestamps: false,
      supportsTokenTimestamps: false,
      supportsSegmentTimestamps: false,
      supportsConfidence: false,
    });
  });

  it('validates capabilities and inference limits for each catalog model', () => {
    for (const model of models) {
      expect(model.capabilities).toBeDefined();
      expect(model.capabilities.supportedTasks.length).toBeGreaterThan(0);
      expect(typeof model.capabilities.supportsTranslation).toBe('boolean');
      expect(typeof model.capabilities.supportsWordTimestamps).toBe('boolean');

      expect(model.inference).toBeDefined();
      expect(model.inference.sampleRate).toBe(16000);
      expect(model.inference.maxInputDurationSec).toBeGreaterThan(0);
      expect(model.inference.recommendedWindowDurationSec).toBeGreaterThan(0);
      expect(model.inference.minWindowDurationSec).toBeGreaterThan(0);
    }
  });

  it('validates backend loading configuration and constraints', () => {
    for (const model of models) {
      const { loading } = model;
      expect(loading).toBeDefined();

      expect(loading.availableEncoderBackends.length).toBeGreaterThan(0);
      for (const backend of loading.availableEncoderBackends) {
        expect(['webgpu', 'wasm']).toContain(backend);
      }
      expect(loading.availableEncoderBackends).toContain(loading.defaultEncoderBackend);

      expect(loading.availableDecoderBackends.length).toBeGreaterThan(0);
      for (const backend of loading.availableDecoderBackends) {
        expect(['webgpu', 'wasm']).toContain(backend);
      }
      expect(loading.availableDecoderBackends).toContain(loading.defaultDecoderBackend);

      expect(loading.availableEncoderQuantizations.length).toBeGreaterThan(0);
      expect(loading.availableDecoderQuantizations.length).toBeGreaterThan(0);
      expect(loading.availableEncoderQuantizations).toContain(loading.defaultEncoderQuantization);
      expect(loading.availableDecoderQuantizations).toContain(loading.defaultDecoderQuantization);
    }
  });

  it('builds valid load options and resolves component backends for each model', () => {
    for (const model of models) {
      if (model.loading.supportsHubSource) {
        const loadOptions = buildSpeechModelLoadOptions({
          modelId: model.modelId,
        });

        expect(loadOptions.preset).toBe(model.preset);
        expect(loadOptions.modelId).toBe(model.modelId);
      } else {
        expect(model.loading.supportsLocalSource).toBe(true);
      }

      const componentBackends = resolveSpeechModelComponentBackends(model.modelId);
      expect(['webgpu', 'wasm']).toContain(componentBackends.encoderBackend);
      expect(['webgpu', 'wasm']).toContain(componentBackends.decoderBackend);
    }
  });

  it('registers all 15 presets and 16 families in createBuiltInSpeechRuntime', () => {
    const runtime = createBuiltInSpeechRuntime();

    const registeredPresetNames = runtime.listPresets().map((p) => p.preset);
    for (const presetId of EXPECTED_PRESET_IDS) {
      expect(registeredPresetNames).toContain(presetId);
    }

    const registeredFamilyNames = runtime.listModelFamilies().map((f) => f.family);
    for (const family of EXPECTED_MODEL_FAMILIES) {
      expect(registeredFamilyNames).toContain(family);
    }

    const registeredBackends = runtime.listBackends().map((b) => b.id);
    expect(registeredBackends).toEqual(['webgpu', 'wasm']);
  });

  it('matches all models in listSpeechModels with listSpeechModelOptions', () => {
    expect(options.length).toBe(models.length);
    const optionKeys = new Set(options.map((opt) => opt.key));
    for (const model of models) {
      expect(optionKeys.has(model.modelId)).toBe(true);
    }
  });

  it('resolves common language labels', () => {
    expect(getSpeechModelLanguageName('en')).toBe('English');
    expect(getSpeechModelLanguageName('de')).toBe('German');
    expect(getSpeechModelLanguageName('es')).toBe('Spanish');
    expect(getSpeechModelLanguageName('fr')).toBe('French');
    expect(getSpeechModelLanguageName('auto')).toBe('Auto-detect');
  });
});
