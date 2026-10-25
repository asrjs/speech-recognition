import {
  CANARY_1B_V2_LANGUAGES,
  CANARY_1B_V2_VOCABULARY_SIZE,
  DEFAULT_MODEL,
  getLanguageName,
  getModelConfig,
  getModelKeyFromRepoId,
  listModels,
  resolveCanaryArtifactSource,
  resolveCanaryPresetManifest,
} from '@asrjs/speech-recognition/presets/canary';
import { getBuiltInModelDescriptor } from '@asrjs/speech-recognition/presets';
import { describe, expect, it } from 'vitest';

describe('Canary catalog', () => {
  it('advertises only registered sources and prompt-format controls', () => {
    const legacy = getBuiltInModelDescriptor('nvidia/canary-1b');
    const v2 = getBuiltInModelDescriptor('nvidia/canary-1b-v2');
    const flash = getBuiltInModelDescriptor(DEFAULT_MODEL);
    expect(legacy?.loading).toMatchObject({ supportsHubSource: false, supportsLocalSource: true });
    expect(v2?.loading).toMatchObject({ supportsHubSource: false, supportsLocalSource: true });
    expect(flash?.loading.supportsHubSource).toBe(true);
    expect(legacy?.capabilities.supportsInverseTextNormalization).toBe(false);
    expect(legacy?.controls.some((control) => control.key === 'timestamps')).toBe(false);
    for (const descriptor of [v2, flash]) {
      expect(descriptor?.capabilities.supportsInverseTextNormalization).toBe(true);
      expect(descriptor?.controls.some((control) => control.key === 'timestamps')).toBe(true);
    }
  });

  it('lists available Canary models', () => {
    const models = listModels();

    expect(models.length).toBeGreaterThan(0);
    expect(models).toContain(DEFAULT_MODEL);
  });

  it('retrieves model configuration by model key and repository id', () => {
    const byKey = getModelConfig(DEFAULT_MODEL);
    const byRepoId = getModelConfig('ysdede/canary-180m-flash-onnx');

    expect(byKey).not.toBeNull();
    expect(byRepoId).toBe(byKey);
    expect(byKey?.repoId).toBe('ysdede/canary-180m-flash-onnx');
    expect(byKey?.languages).toEqual(['en', 'de', 'es', 'fr']);
    expect(byKey?.defaultSourceLanguage).toBe('en');
    expect(byKey?.defaultTargetLanguage).toBe('en');
  });

  it('keeps Canary 1B v2 metadata aligned with its checkpoint and model card', () => {
    const config = getModelConfig('nvidia/canary-1b-v2');
    const manifest = resolveCanaryPresetManifest('nvidia/canary-1b-v2');
    const descriptor = getBuiltInModelDescriptor('nvidia/canary-1b-v2');
    const flashDescriptor = getBuiltInModelDescriptor(DEFAULT_MODEL);

    expect(config).toMatchObject({
      languages: CANARY_1B_V2_LANGUAGES,
      vocabSize: CANARY_1B_V2_VOCABULARY_SIZE,
      featuresSize: 128,
      encoderLayers: 32,
      decoderLayers: 8,
    });
    expect(CANARY_1B_V2_LANGUAGES).toHaveLength(25);
    expect(manifest?.config).toMatchObject({
      languages: CANARY_1B_V2_LANGUAGES,
      vocabularySize: CANARY_1B_V2_VOCABULARY_SIZE,
      melBins: 128,
    });
    expect(manifest?.source).toBeUndefined();
    expect(manifest?.docs).toBeUndefined();
    expect(resolveCanaryArtifactSource('nvidia/canary-1b-v2')).toBeUndefined();
    expect(descriptor?.languages).toEqual(CANARY_1B_V2_LANGUAGES);
    expect(descriptor?.description).toBe(manifest?.description);
    expect(descriptor?.docs).toBeUndefined();
    expect(flashDescriptor?.docs?.modelCardUrl).toBe('https://huggingface.co/nvidia/canary-180m-flash');
    expect(flashDescriptor?.capabilities.supportsWordTimestamps).toBe(true);
  });

  it('treats unknown and prototype-like keys as missing', () => {
    expect(getModelConfig('invalid-model-key')).toBeNull();
    expect(getModelConfig('toString')).toBeNull();
    expect(getModelConfig('__proto__')).toBeNull();
    expect(getModelKeyFromRepoId('invalid-repo-id')).toBeNull();
  });

  it('resolves model keys and artifact sources from known repository ids', () => {
    expect(getModelKeyFromRepoId('ysdede/canary-180m-flash-onnx')).toBe(DEFAULT_MODEL);

    const source = resolveCanaryArtifactSource(DEFAULT_MODEL);
    expect(source?.kind).toBe('huggingface');
    expect(source?.repoId).toBe('ysdede/canary-180m-flash-onnx');
    expect(source?.preprocessorBackend).toBe('js');
    expect(resolveCanaryArtifactSource('unknown-model')).toBeUndefined();
  });

  it('resolves language names case-insensitively and falls back to the code', () => {
    expect(getLanguageName('en')).toBe('English');
    expect(getLanguageName('EN')).toBe('English');
    expect(getLanguageName('De')).toBe('German');
    expect(getLanguageName('xx')).toBe('xx');
  });
});
