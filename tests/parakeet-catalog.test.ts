import { describe, expect, expectTypeOf, it } from 'vitest';
import {
  getModelConfig,
  getModelKeyFromRepoId,
  MODELS,
  type ParakeetModelConfig,
} from '../src/presets/parakeet/catalog.js';

describe('Parakeet catalog', () => {
  it('resolves model keys from repository ids', () => {
    expect(getModelKeyFromRepoId('ysdede/parakeet-tdt-0.6b-v2-onnx')).toBe(
      'parakeet-tdt-0.6b-v2',
    );
    expect(getModelKeyFromRepoId('ysdede/parakeet-tdt-0.6b-v3-onnx')).toBe(
      'parakeet-tdt-0.6b-v3',
    );
    expect(getModelKeyFromRepoId('ysdede/parakeet-realtime-eou-120m-v1-onnx')).toBe(
      'parakeet-realtime-eou-120m-v1',
    );
    expect(getModelKeyFromRepoId('unknown/repo-id')).toBeNull();
    expect(getModelKeyFromRepoId('')).toBeNull();
  });

  it('returns the v2 TDT configuration by model key and repository id', () => {
    const config = getModelConfig('parakeet-tdt-0.6b-v2');

    expectTypeOf(config).toEqualTypeOf<ParakeetModelConfig | null>();
    expect(getModelConfig('ysdede/parakeet-tdt-0.6b-v2-onnx')).toBe(config);
    expect(config?.repoId).toBe('ysdede/parakeet-tdt-0.6b-v2-onnx');
    expect(config?.languages).toEqual(['en']);
    expect(config?.topology).toBe('tdt');
    expect(config?.supportsWordTimestamps).toBe(true);
    expect(config?.defaultRevision).toBe('main');
    expect(config?.cacheKeyFallbackRevisions).toEqual(['feat/fp16-canonical-v2']);
  });

  it('returns the v3 multilingual TDT configuration', () => {
    const config = getModelConfig('parakeet-tdt-0.6b-v3');

    expect(config?.repoId).toBe('ysdede/parakeet-tdt-0.6b-v3-onnx');
    expect(config?.languages).toContain('ja');
    expect(config?.vocabSize).toBe(8193);
    expect(config?.topology).toBe('tdt');
    expect(config?.supportsWordTimestamps).toBe(true);
    expect(config?.defaultRevision).toBe('main');
    expect(config?.cacheKeyFallbackRevisions).toEqual(['feat/fp16-canonical-v3']);
  });

  it('returns the realtime EOU RNNT configuration', () => {
    const config = getModelConfig('parakeet-realtime-eou-120m-v1');

    expect(config?.repoId).toBe('ysdede/parakeet-realtime-eou-120m-v1-onnx');
    expect(config?.topology).toBe('rnnt');
    expect(config?.supportsWordTimestamps).toBe(false);
    expect(config?.defaultRevision).toBe('6d6be8e9113b4aa8ae7b4d5dfb655795c084d0c6');
    expect(config?.warmupRequiredKeywordGroups).toEqual([
      ['boy', 'there'],
      ['pink', 'salmon'],
    ]);
  });

  it('treats unknown and prototype-like keys as missing', () => {
    expect(getModelConfig('unknown-model')).toBeNull();
    expect(getModelConfig('ysdede/unknown-repo-onnx')).toBeNull();
    expect(getModelConfig('toString')).toBeNull();
    expect(getModelConfig('__proto__')).toBeNull();
    expect(getModelConfig('hasOwnProperty')).toBeNull();
  });

  it('returns catalog object references for known model keys', () => {
    expect(getModelConfig('parakeet-tdt-0.6b-v2')).toBe(MODELS['parakeet-tdt-0.6b-v2']);
  });
});
