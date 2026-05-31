import { describe, expect, it } from 'vitest';

import { createMedAsrPresetFactory } from '../src/presets/medasr/factory.js';

describe('createMedAsrPresetFactory', () => {
  it('returns a factory with the correct preset identifier', () => {
    const factory = createMedAsrPresetFactory();

    expect(factory.preset).toBe('medasr');
  });

  describe('supports', () => {
    it('returns true when no modelId is provided', () => {
      const factory = createMedAsrPresetFactory();

      expect(factory.supports()).toBe(true);
    });

    it('returns true for known MedASR model IDs', () => {
      const factory = createMedAsrPresetFactory();

      expect(factory.supports('google/medasr')).toBe(true);
      expect(factory.supports('medasr')).toBe(true);
      expect(factory.supports('google-medasr')).toBe(true);
    });

    it('returns false for unknown model IDs', () => {
      const factory = createMedAsrPresetFactory();

      expect(factory.supports('unknown/model')).toBe(false);
    });
  });

  describe('resolveModelRequest', () => {
    it('resolves a basic request with default modelId', async () => {
      const factory = createMedAsrPresetFactory();
      const request = await factory.resolveModelRequest({}, {} as any);

      expect(request.family).toBe('lasr-ctc');
      expect(request.modelId).toBe('google/medasr');
      expect(request.resolvedPreset).toBe('medasr');

      expect(request.classification?.family).toBe('medasr');
    });

    it('merges request options with manifest defaults', async () => {
      const factory = createMedAsrPresetFactory();
      const request = await factory.resolveModelRequest(
        {
          modelId: 'medasr',
          options: {
            config: {
              featureHopSeconds: 0.05, // Override
            },
          },
        },
        {} as any,
      );

      // Verify override
      expect(request.options?.config?.featureHopSeconds).toBe(0.05);

      // Verify inherited from manifest
      expect(request.options?.config?.languages).toEqual(['en']);
      expect(request.options?.source?.kind).toBe('huggingface');
      expect(request.options?.source?.repoId).toBe('ysdede/medasr-onnx');
    });

    it('handles requests for non-existent manifests gracefully', async () => {
      const factory = createMedAsrPresetFactory();
      const request = await factory.resolveModelRequest(
        {
          modelId: 'unknown/model',
        },
        {} as any,
      );

      expect(request.modelId).toBe('unknown/model');
      expect(request.options?.config).toEqual({});
      expect(request.options?.source).toBeUndefined();
    });
  });
});
