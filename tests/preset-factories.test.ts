import { describe, expect, it } from 'vitest';
import { createCanaryPresetFactory, DEFAULT_MODEL } from '../src/presets/canary/index.js';
import { createMedAsrPresetFactory } from '../src/presets/medasr/factory.js';
import { createWhisperPresetFactory } from '../src/presets/whisper/factory.js';
import type { ResolveModelRequestContext } from '../src/types/index.js';

const context = {} as ResolveModelRequestContext;

describe('preset factories', () => {
  it('resolves Canary requests with defaults, manifest sources, and user overrides', async () => {
    const plain = createCanaryPresetFactory();

    expect(plain.preset).toBe('canary');
    expect(plain.supports()).toBe(true);
    expect(plain.supports(DEFAULT_MODEL)).toBe(true);
    expect(plain.supports('unknown/model')).toBe(false);

    const defaultRequest = await plain.resolveModelRequest({}, context);
    expect(defaultRequest.family).toBe('nemo-aed');
    expect(defaultRequest.modelId).toBe(DEFAULT_MODEL);
    expect(defaultRequest.resolvedPreset).toBe('canary');
    expect(defaultRequest.classification?.family).toBe('canary');
    expect(defaultRequest.options?.source).toBeUndefined();

    const withSource = await createCanaryPresetFactory({ useManifestSource: true }).resolveModelRequest(
      {},
      context,
    );
    expect(withSource.options?.source?.kind).toBe('huggingface');

    const overridden = await plain.resolveModelRequest(
      {
        modelId: DEFAULT_MODEL,
        classification: { task: 'custom-task' },
        options: { config: { featuresSize: 256 } },
      },
      context,
    );
    expect(overridden.classification?.task).toBe('custom-task');
    expect(overridden.options?.config?.featuresSize).toBe(256);
  });

  it('resolves MedASR requests with manifest defaults and overrides', async () => {
    const factory = createMedAsrPresetFactory();

    expect(factory.preset).toBe('medasr');
    expect(factory.supports()).toBe(true);
    expect(factory.supports('google/medasr')).toBe(true);
    expect(factory.supports('unknown/model')).toBe(false);

    const request = await factory.resolveModelRequest(
      {
        modelId: 'medasr',
        options: { config: { featureHopSeconds: 0.05 } },
      },
      context,
    );

    expect(request.family).toBe('lasr-ctc');
    expect(request.modelId).toBe('medasr');
    expect(request.resolvedPreset).toBe('medasr');
    expect(request.classification?.family).toBe('medasr');
    expect(request.options?.config?.featureHopSeconds).toBe(0.05);
    expect(request.options?.config?.languages).toEqual(['en']);
    expect(request.options?.source?.kind).toBe('huggingface');
  });

  it('resolves Whisper requests with optional manifest sources', async () => {
    const factory = createWhisperPresetFactory();

    expect(factory.preset).toBe('whisper');
    expect(factory.supports()).toBe(true);
    expect(factory.supports('onnx-community/whisper-tiny')).toBe(true);
    expect(factory.supports('unknown-model')).toBe(false);

    const defaultRequest = await factory.resolveModelRequest({ preset: 'whisper' }, context);
    expect(defaultRequest.family).toBe('whisper-seq2seq');
    expect(defaultRequest.modelId).toBe('onnx-community/whisper-base');
    expect(defaultRequest.resolvedPreset).toBe('whisper');
    expect(defaultRequest.classification).toEqual({ family: 'whisper' });
    expect(defaultRequest.options?.source).toBeUndefined();

    const withSource = await createWhisperPresetFactory({
      useManifestSource: true,
    }).resolveModelRequest({ preset: 'whisper' }, context);
    expect(withSource.options?.source?.kind).toBe('huggingface');

    const customSource = { kind: 'url' as const, url: 'https://example.com/model.onnx' };
    const overridden = await createWhisperPresetFactory({
      useManifestSource: true,
    }).resolveModelRequest(
      {
        preset: 'whisper',
        classification: { ecosystem: 'custom' },
        options: {
          source: customSource,
          config: { maxSourcePositions: 1000 },
        },
      },
      context,
    );
    expect(overridden.classification).toEqual({ family: 'whisper', ecosystem: 'custom' });
    expect(overridden.options?.source).toBe(customSource);
    expect(overridden.options?.config?.maxSourcePositions).toBe(1000);
  });
});
