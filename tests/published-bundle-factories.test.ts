import { describe, expect, it } from 'vitest';
import { createMoonshinePresetFactory } from '../src/presets/moonshine-factory.js';
import { createQwen3AsrPresetFactory } from '../src/presets/qwen3-asr-factory.js';
import type { ResolveModelRequestContext } from '../src/types/index.js';

const context = {} as ResolveModelRequestContext;

describe('published ONNX bundle preset routing', () => {
  it('maps the registered Qwen repository to its pinned three-graph direct contract', async () => {
    const factory = createQwen3AsrPresetFactory();
    const resolved = await factory.resolveModelRequest(
      { modelId: 'Qwen/Qwen3-ASR-0.6B-hf' },
      context,
    );
    const source = resolved.options?.source;
    expect(source?.kind).toBe('direct');
    if (source?.kind !== 'direct') throw new Error('Expected direct Qwen graph source.');
    expect(source.artifacts).toMatchObject({
      encoderUrl: expect.stringContaining(
        '/ac0bedcc0f1614c0b3982940c14bdb169ff2365d/audio-encoder-dynamic.onnx',
      ),
      decoderUrl: expect.stringContaining('/decoder-prefill-fp16.onnx'),
      decoderStepUrl: expect.stringContaining('/decoder-step-fp16.onnx'),
      tokenizerUrl: expect.stringContaining('/tokenizer/tokenizer.json'),
      decoderPrefillDataUrl: expect.stringContaining('/decoder-fp16.onnx.data'),
      decoderStepDataUrl: expect.stringContaining('/decoder-fp16.onnx.data'),
      decoderPrefillDataPath: 'decoder-fp16.onnx.data',
      decoderStepDataPath: 'decoder-fp16.onnx.data',
    });
  });

  it('preserves explicit Qwen direct graphs and rejects unregistered Hub folders', async () => {
    const factory = createQwen3AsrPresetFactory();
    const direct = {
      kind: 'direct' as const,
      artifacts: {
        encoderUrl: 'local:encoder',
        decoderUrl: 'local:decoder',
        tokenizerUrl: 'local:tokenizer',
      },
    };
    const resolved = await factory.resolveModelRequest({ options: { source: direct } }, context);
    expect(resolved.options?.source).toBe(direct);
    await expect(
      factory.resolveModelRequest(
        {
          options: { source: { kind: 'huggingface', repoId: 'Qwen/Qwen3-ASR-0.6B-hf' } },
        },
        context,
      ),
    ).rejects.toThrow(/registered ONNX Hub bundle/);
    await expect(
      factory.resolveModelRequest(
        {
          options: {
            source: { kind: 'huggingface', repoId: 'ysdede/qwen3-asr-0.6b-onnx', revision: 'main' },
          },
        },
        context,
      ),
    ).rejects.toThrow(/registered pinned revision/);
    await expect(
      factory.resolveModelRequest(
        {
          options: {
            source: {
              kind: 'huggingface',
              repoId: 'ysdede/qwen3-asr-0.6b-onnx',
              decoderQuant: 'fp32',
            },
          },
        },
        context,
      ),
    ).rejects.toThrow(/decoder precision.*fp16/);
    await expect(
      createQwen3AsrPresetFactory({ useManifestSource: false }).resolveModelRequest({}, context),
    ).rejects.toThrow(/explicit direct graph source/);
    await expect(
      factory.resolveModelRequest(
        {
          options: {
            source: {
              kind: 'huggingface',
              repoId: 'ysdede/qwen3-asr-0.6b-onnx',
              decoderQuant: 'fp32',
            },
          },
        },
        context,
      ),
    ).rejects.toThrow(/decoder precision.*fp16/);
    await expect(
      createQwen3AsrPresetFactory({ useManifestSource: false }).resolveModelRequest({}, context),
    ).rejects.toThrow(/explicit direct graph source/);
  });

  it('maps only the streaming Tiny catalog id to the four-graph Moonshine bundle', async () => {
    const factory = createMoonshinePresetFactory();
    const resolved = await factory.resolveModelRequest(
      { modelId: 'moonshine-ai/moonshine-streaming-tiny' },
      context,
    );
    expect(resolved.options).toMatchObject({
      encoderUrl: expect.stringContaining('/encoder_model.onnx'),
      tokenizerUrl: expect.stringContaining('/tokenizer.json'),
      generationUrl: expect.stringContaining('/generation_config.json'),
      streaming: {
        encoderDataUrl: expect.stringContaining('/encoder_model.onnx.data'),
        chunkedEncoderUrl: expect.stringContaining('/encoder_chunked_model.onnx'),
        chunkedEncoderDataUrl: expect.stringContaining('/encoder_chunked_model.onnx.data'),
        prefillUrl: expect.stringContaining('/decoder_prefill_model.onnx'),
        prefillDataUrl: expect.stringContaining('/decoder_prefill_model.onnx.data'),
        stepUrl: expect.stringContaining('/decoder_step_model.onnx'),
        stepDataUrl: expect.stringContaining('/decoder_step_model.onnx.data'),
      },
    });
    await expect(
      factory.resolveModelRequest({ modelId: 'moonshine-tiny' }, context),
    ).rejects.toThrow(/only the registered streaming Tiny bundle/);
  });
});
