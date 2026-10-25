import { describe, expect, it } from 'vitest';
import { createCanaryPresetFactory, DEFAULT_MODEL } from '../src/presets/canary/index.js';
import { createMedAsrPresetFactory } from '../src/presets/medasr/factory.js';
import { createWhisperPresetFactory } from '../src/presets/whisper/factory.js';
import { createGigaAmCtcPresetFactory } from '../src/presets/gigaam-ctc-local.js';
import { createGigaAmRnntPresetFactory } from '../src/presets/gigaam-rnnt-local.js';
import { createSenseVoicePresetFactory } from '../src/presets/sensevoice-local.js';
import { createXAsrPresetFactory } from '../src/presets/x-asr-local.js';
import { buildSpeechModelLoadOptions } from '../src/runtime/catalog.js';
import type { ResolveModelRequestContext } from '../src/types/index.js';

const context = {} as ResolveModelRequestContext;

describe('preset factories', () => {
  it.each(['nvidia/canary-1b', 'nvidia/canary-1b-v2'])('keeps %s source requirements observable through the catalog loader', async (modelId) => {
    const request = buildSpeechModelLoadOptions({ modelId });
    expect(request.options?.source).toBeUndefined();
    await expect(createCanaryPresetFactory({ useManifestSource: true }).resolveModelRequest(
      request, context,
    )).rejects.toThrow(/No verified Canary ONNX artifact source/);
  });
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
    expect(withSource.options?.source?.kind === 'huggingface' && withSource.options.source.repoId)
      .toBe('ysdede/canary-180m-flash-onnx');

    const strictFactory = createCanaryPresetFactory({ useManifestSource: true });
    const customSource = {
      kind: 'direct' as const,
      artifacts: {
        encoderUrl: 'https://example.test/encoder.onnx',
        decoderUrl: 'https://example.test/decoder.onnx',
        tokenizerUrl: 'https://example.test/tokenizer.json',
      },
    };
    for (const modelId of ['nvidia/canary-1b', 'nvidia/canary-1b-v2']) {
      await expect(strictFactory.resolveModelRequest({ modelId }, context)).rejects.toThrow(
        /No verified Canary ONNX artifact source.*compatible adapter\/export/,
      );
      const overridden = await strictFactory.resolveModelRequest(
        { modelId, options: { source: customSource } },
        context,
      );
      expect(overridden.options?.source).toBe(customSource);
    }

    const optedOut = await createCanaryPresetFactory({ useManifestSource: false }).resolveModelRequest(
      { modelId: 'nvidia/canary-1b-v2' },
      context,
    );
    expect(optedOut.options?.source).toBeUndefined();

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

    const strictFactory = createWhisperPresetFactory({ useManifestSource: true });
    await expect(
      strictFactory.resolveModelRequest({ modelId: 'openai/whisper-large-v3' }, context),
    ).rejects.toThrow(/No verified Whisper artifact source is registered/);

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

  it('keeps the new official GigaAM and SenseVoice presets local-direct only', async () => {
    const cases = [
      { factory: createGigaAmCtcPresetFactory(), modelId: 'gigaam-multilingual-ctc', family: 'gigaam-ctc' },
      { factory: createGigaAmRnntPresetFactory(), modelId: 'gigaam-v3-e2e-rnnt', family: 'gigaam-rnnt' },
      { factory: createSenseVoicePresetFactory(), modelId: 'FunAudioLLM/SenseVoiceSmall', family: 'sensevoice' },
      { factory: createXAsrPresetFactory(), modelId: 'x-asr-zh-en-160ms', family: 'x-asr' },
    ];

    for (const { factory, modelId, family } of cases) {
      expect(factory.supports(modelId)).toBe(true);
      await expect(factory.resolveModelRequest({ modelId }, context)).rejects.toThrow(/local official ONNX artifacts/);
      await expect(factory.resolveModelRequest({
        modelId,
        options: { source: { kind: 'huggingface', repoId: 'unverified/model' } },
      } as never, context)).rejects.toThrow(/requires local official ONNX artifacts/);
      const source = { kind: 'direct' as const, artifacts: {} };
      const resolved = await factory.resolveModelRequest({ modelId, options: { source } } as never, context);
      expect(resolved.family).toBe(family);
      expect(resolved.modelId).toBe(modelId);
      expect(resolved.options?.source).toBe(source);
      if (family === 'x-asr') expect(resolved.options?.config).toMatchObject({ chunkMs: 160 });
    }
  });
});
