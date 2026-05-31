import { describe, expect, it } from 'vitest';
import { createWhisperPresetFactory } from '../src/presets/whisper/factory.js';

describe('createWhisperPresetFactory', () => {
  it('returns a factory with the correct preset name', () => {
    const factory = createWhisperPresetFactory();
    expect(factory.preset).toBe('whisper');
  });

  describe('supports', () => {
    it('returns true when no modelId is provided', () => {
      const factory = createWhisperPresetFactory();
      expect(factory.supports()).toBe(true);
    });

    it('returns true for known whisper model IDs', () => {
      const factory = createWhisperPresetFactory();
      expect(factory.supports('whisper-base')).toBe(true);
      expect(factory.supports('onnx-community/whisper-tiny')).toBe(true);
    });

    it('returns false for unknown model IDs', () => {
      const factory = createWhisperPresetFactory();
      expect(factory.supports('unknown-model')).toBe(false);
      expect(factory.supports('parakeet-tdt')).toBe(false);
    });
  });

  describe('resolveModelRequest', () => {
    it('resolves a request with defaults when no modelId is provided', async () => {
      const factory = createWhisperPresetFactory();
      const request = await factory.resolveModelRequest({ preset: 'whisper' }, {} as any);

      expect(request.family).toBe('whisper-seq2seq');
      expect(request.modelId).toBe('onnx-community/whisper-base');
      expect(request.resolvedPreset).toBe('whisper');
      expect(request.classification).toEqual({ family: 'whisper' });
      expect(request.options?.config?.melBins).toBe(80); // Base config melBins
    });

    it('resolves a request for a specific model ID', async () => {
      const factory = createWhisperPresetFactory();
      const request = await factory.resolveModelRequest({ preset: 'whisper', modelId: 'whisper-large-v3-turbo' }, {} as any);

      expect(request.modelId).toBe('whisper-large-v3-turbo');
      expect(request.options?.config?.melBins).toBe(128); // Large config melBins
    });

    it('merges classification and options from the request', async () => {
      const factory = createWhisperPresetFactory();
      const request = await factory.resolveModelRequest({
        preset: 'whisper',
        classification: { ecosystem: 'custom' },
        options: { config: { maxSourcePositions: 1000 } }
      }, {} as any);

      expect(request.classification).toEqual({ family: 'whisper', ecosystem: 'custom' });
      expect(request.options?.config?.maxSourcePositions).toBe(1000);
    });

    it('resolves manifest source when useManifestSource is true', async () => {
      const factory = createWhisperPresetFactory({ useManifestSource: true });
      const request = await factory.resolveModelRequest({ preset: 'whisper' }, {} as any);

      expect(request.options?.source).toBeDefined();
      expect(request.options?.source?.kind).toBe('huggingface');
    });

    it('does not resolve manifest source when useManifestSource is false', async () => {
      const factory = createWhisperPresetFactory({ useManifestSource: false });
      const request = await factory.resolveModelRequest({ preset: 'whisper' }, {} as any);

      expect(request.options?.source).toBeUndefined();
    });

    it('prefers source from request options over manifest source', async () => {
      const factory = createWhisperPresetFactory({ useManifestSource: true });
      const customSource = { kind: 'url' as const, url: 'http://example.com' };
      const request = await factory.resolveModelRequest({
        preset: 'whisper',
        options: { source: customSource }
      }, {} as any);

      expect(request.options?.source).toBe(customSource);
    });
  });
});
