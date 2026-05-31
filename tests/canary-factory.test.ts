import { describe, expect, it } from 'vitest';
import { createCanaryPresetFactory } from '../src/presets/canary/factory.js';
import { DEFAULT_MODEL } from '../src/presets/canary/manifest.js';
import type { ResolveModelRequestContext } from '../src/types/index.js';

describe('createCanaryPresetFactory', () => {
  it('creates a factory with the canary preset id', () => {
    const factory = createCanaryPresetFactory();
    expect(factory.preset).toBe('canary');
  });

  it('supports the default models and undefined model ids', () => {
    const factory = createCanaryPresetFactory();
    expect(factory.supports()).toBe(true);
    expect(factory.supports(DEFAULT_MODEL)).toBe(true);
  });

  it('does not support unknown models', () => {
    const factory = createCanaryPresetFactory();
    expect(factory.supports('unknown/model')).toBe(false);
  });

  it('resolves model requests with default options', async () => {
    const factory = createCanaryPresetFactory();
    const request = await factory.resolveModelRequest({}, {} as ResolveModelRequestContext);

    expect(request.family).toBe('nemo-aed');
    expect(request.modelId).toBe(DEFAULT_MODEL);
    expect(request.resolvedPreset).toBe('canary');
    expect(request.classification?.family).toBe('canary');
    expect(request.options?.source).toBeUndefined();
  });

  it('resolves model requests including the manifest source when requested', async () => {
    const factory = createCanaryPresetFactory({ useManifestSource: true });
    const request = await factory.resolveModelRequest({}, {} as ResolveModelRequestContext);

    expect(request.options?.source).toBeDefined();
    if (request.options?.source?.kind === 'huggingface') {
      expect(request.options.source.repoId).toBe('ysdede/canary-180m-flash-onnx');
    }
  });

  it('preserves and merges user overrides in the model request', async () => {
    const factory = createCanaryPresetFactory();
    const request = await factory.resolveModelRequest(
      {
        modelId: DEFAULT_MODEL,
        classification: { task: 'custom-task' },
        options: {
          config: { featuresSize: 256 } as any,
        },
      },
      {} as ResolveModelRequestContext,
    );

    expect(request.modelId).toBe(DEFAULT_MODEL);
    expect(request.classification?.task).toBe('custom-task');
    expect(request.classification?.family).toBe('canary');
    expect((request.options?.config as any)?.featuresSize).toBe(256);
  });
});
