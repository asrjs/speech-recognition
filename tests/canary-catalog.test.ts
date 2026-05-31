import {
  getLanguageName,
  getModelConfig,
  getModelKeyFromRepoId,
  listModels,
} from '@asrjs/speech-recognition/presets/canary';
import { describe, expect, it } from 'vitest';

describe('Canary catalog', () => {
  it('exposes a list of available Canary models', () => {
    const models = listModels();
    expect(models.length).toBeGreaterThan(0);
    expect(models).toContain('nvidia/canary-180m-flash');
  });

  it('retrieves model configuration by model key', () => {
    const config = getModelConfig('nvidia/canary-180m-flash');
    expect(config).not.toBeNull();
    expect(config?.repoId).toBe('ysdede/canary-180m-flash-onnx');
    expect(config?.languages).toContain('en');
    expect(config?.defaultSourceLanguage).toBe('en');
    expect(config?.defaultTargetLanguage).toBe('en');
    expect(config?.featuresSize).toBe(128);
  });

  it('retrieves model configuration by repository ID', () => {
    const config = getModelConfig('ysdede/canary-180m-flash-onnx');
    expect(config).not.toBeNull();
    expect(config?.repoId).toBe('ysdede/canary-180m-flash-onnx');
    expect(config?.languages).toContain('fr');
  });

  it('returns null for unknown model configurations', () => {
    expect(getModelConfig('invalid-model-key')).toBeNull();
    expect(getModelConfig('invalid-repo-id')).toBeNull();
    expect(getModelConfig('toString')).toBeNull();
    expect(getModelConfig('__proto__')).toBeNull();
  });

  it('resolves model keys from repository IDs', () => {
    expect(getModelKeyFromRepoId('ysdede/canary-180m-flash-onnx')).toBe('nvidia/canary-180m-flash');
  });

  it('returns null when resolving unknown repository IDs', () => {
    expect(getModelKeyFromRepoId('invalid-repo-id')).toBeNull();
    expect(getModelKeyFromRepoId('toString')).toBeNull();
    expect(getModelKeyFromRepoId('__proto__')).toBeNull();
  });

  it('resolves language names case-insensitively and falls back to the code', () => {
    expect(getLanguageName('en')).toBe('English');
    expect(getLanguageName('EN')).toBe('English');
    expect(getLanguageName('De')).toBe('German');
    expect(getLanguageName('fr')).toBe('French');
    expect(getLanguageName('xx')).toBe('xx');
    expect(getLanguageName('unknown')).toBe('unknown');
  });
});
