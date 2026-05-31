import { describe, expect, it, expectTypeOf } from 'vitest';
import {
  getModelConfig,
  getModelKeyFromRepoId,
  MODELS,
  type ParakeetModelConfig,
} from '../src/presets/parakeet/catalog.js';

describe('Parakeet Catalog', () => {
  describe('getModelKeyFromRepoId', () => {
    it('resolves parakeet-tdt-0.6b-v2 from its repoId', () => {
      expect(getModelKeyFromRepoId('ysdede/parakeet-tdt-0.6b-v2-onnx')).toBe('parakeet-tdt-0.6b-v2');
    });

    it('resolves parakeet-tdt-0.6b-v3 from its repoId', () => {
      expect(getModelKeyFromRepoId('ysdede/parakeet-tdt-0.6b-v3-onnx')).toBe('parakeet-tdt-0.6b-v3');
    });

    it('resolves parakeet-realtime-eou-120m-v1 from its repoId', () => {
      expect(getModelKeyFromRepoId('ysdede/parakeet-realtime-eou-120m-v1-onnx')).toBe('parakeet-realtime-eou-120m-v1');
    });

    it('returns null for unknown repository IDs', () => {
      expect(getModelKeyFromRepoId('unknown/repo-id')).toBeNull();
      expect(getModelKeyFromRepoId('')).toBeNull();
    });
  });

  describe('getModelConfig', () => {
    it('returns correct configuration for parakeet-tdt-0.6b-v2 using model key', () => {
      const config = getModelConfig('parakeet-tdt-0.6b-v2');
      expectTypeOf(config).toEqualTypeOf<ParakeetModelConfig | null>();
      expect(config).not.toBeNull();
      expect(config?.repoId).toBe('ysdede/parakeet-tdt-0.6b-v2-onnx');
      expect(config?.displayName).toBe('Parakeet TDT 0.6B v2 (English)');
      expect(config?.languages).toEqual(['en']);
      expect(config?.defaultLanguage).toBe('en');
      expect(config?.vocabSize).toBe(1025);
      expect(config?.featuresSize).toBe(128);
      expect(config?.preprocessor).toBe('nemo128');
      expect(config?.subsampling).toBe(8);
      expect(config?.predHidden).toBe(640);
      expect(config?.predLayers).toBe(2);
      expect(config?.topology).toBe('tdt');
      expect(config?.supportsWordTimestamps).toBe(true);
      expect(config?.defaultRevision).toBe('main');
      expect(config?.cacheKeyFallbackRevisions).toEqual(['feat/fp16-canonical-v2']);
    });

    it('returns correct configuration for parakeet-tdt-0.6b-v2 using repoId', () => {
      const config = getModelConfig('ysdede/parakeet-tdt-0.6b-v2-onnx');
      expect(config).not.toBeNull();
      expect(config?.vocabSize).toBe(1025);
      expect(config?.topology).toBe('tdt');
    });

    it('returns correct configuration for parakeet-tdt-0.6b-v3 using model key', () => {
      const config = getModelConfig('parakeet-tdt-0.6b-v3');
      expect(config).not.toBeNull();
      expect(config?.repoId).toBe('ysdede/parakeet-tdt-0.6b-v3-onnx');
      expect(config?.displayName).toBe('Parakeet TDT 0.6B v3 (Multilingual)');
      expect(config?.languages).toEqual(['en', 'fr', 'de', 'es', 'it', 'pt', 'nl', 'pl', 'ru', 'uk', 'ja', 'ko', 'zh']);
      expect(config?.defaultLanguage).toBe('en');
      expect(config?.vocabSize).toBe(8193);
      expect(config?.featuresSize).toBe(128);
      expect(config?.preprocessor).toBe('nemo128');
      expect(config?.subsampling).toBe(8);
      expect(config?.predHidden).toBe(640);
      expect(config?.predLayers).toBe(2);
      expect(config?.topology).toBe('tdt');
      expect(config?.supportsWordTimestamps).toBe(true);
      expect(config?.defaultRevision).toBe('main');
      expect(config?.cacheKeyFallbackRevisions).toEqual(['feat/fp16-canonical-v3']);
    });

    it('returns correct configuration for parakeet-tdt-0.6b-v3 using repoId', () => {
      const config = getModelConfig('ysdede/parakeet-tdt-0.6b-v3-onnx');
      expect(config).not.toBeNull();
      expect(config?.vocabSize).toBe(8193);
      expect(config?.topology).toBe('tdt');
    });

    it('returns correct configuration for parakeet-realtime-eou-120m-v1 using model key', () => {
      const config = getModelConfig('parakeet-realtime-eou-120m-v1');
      expect(config).not.toBeNull();
      expect(config?.repoId).toBe('ysdede/parakeet-realtime-eou-120m-v1-onnx');
      expect(config?.displayName).toBe('Parakeet Realtime EOU 120M v1 (English)');
      expect(config?.languages).toEqual(['en']);
      expect(config?.defaultLanguage).toBe('en');
      expect(config?.vocabSize).toBe(1026);
      expect(config?.featuresSize).toBe(128);
      expect(config?.preprocessor).toBe('nemo128');
      expect(config?.subsampling).toBe(8);
      expect(config?.predHidden).toBe(640);
      expect(config?.predLayers).toBe(1);
      expect(config?.topology).toBe('rnnt');
      expect(config?.supportsWordTimestamps).toBe(false);
      expect(config?.defaultRevision).toBe('6d6be8e9113b4aa8ae7b4d5dfb655795c084d0c6');
      expect(config?.warmupExpectedTexts).toEqual([
        'the boy was there when the sun rose',
        'the boy was there when the sun rose a rod is used to catch pink salmon',
      ]);
      expect(config?.warmupRequiredKeywordGroups).toEqual([
        ['boy', 'there'],
        ['pink', 'salmon'],
      ]);
    });

    it('returns correct configuration for parakeet-realtime-eou-120m-v1 using repoId', () => {
      const config = getModelConfig('ysdede/parakeet-realtime-eou-120m-v1-onnx');
      expect(config).not.toBeNull();
      expect(config?.vocabSize).toBe(1026);
      expect(config?.topology).toBe('rnnt');
    });

    it('returns null for unknown models', () => {
      expect(getModelConfig('unknown-model')).toBeNull();
      expect(getModelConfig('ysdede/unknown-repo-onnx')).toBeNull();
    });

    it('handles prototype pollution safely', () => {
      expect(getModelConfig('toString')).toBeNull();
      expect(getModelConfig('__proto__')).toBeNull();
      expect(getModelConfig('hasOwnProperty')).toBeNull();
    });

    it('returns the exact reference to the MODELS configuration object', () => {
      const config = getModelConfig('parakeet-tdt-0.6b-v2');
      expect(config).toBe(MODELS['parakeet-tdt-0.6b-v2']);
    });
  });
});
