import {
  DEFAULT_MODEL,
  getLanguageName,
  getModelConfig,
  getModelKeyFromRepoId,
  listModels,
  resolveCanaryArtifactSource,
} from '@asrjs/speech-recognition/presets/canary';
import { describe, expect, it } from 'vitest';

describe('Canary catalog', () => {
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
