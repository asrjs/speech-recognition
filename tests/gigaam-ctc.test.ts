import { describe, expect, it } from 'vitest';

import { GigaAmJsPreprocessor, GigaAmTokenizer, OrtGigaAmCtcExecutor } from '../src/models/gigaam-ctc/index.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from '../src/models/lasr-ctc/ort.js';
import { createBuiltInSpeechRuntime } from '../src/runtime/builtins.js';
import { loadSpeechModel } from '../src/runtime/load.js';

describe('GigaAM Multilingual CTC contract', () => {
  it('uses the published graph feed names and mel-major layout', async () => {
    class Tensor<TData extends ArrayBufferView = ArrayBufferView> implements OrtTensorLike<TData> {
      constructor(readonly type: string, readonly data: TData, readonly dims: readonly number[]) {}
    }
    const feeds: Record<string, unknown> = {};
    const session: OrtSessionLike = {
      async run(input) {
        Object.assign(feeds, input);
        const logits = new Float32Array(99 * 71).fill(-10);
        for (let frame = 0; frame < 99; frame += 1) logits[frame * 71 + 2] = 10;
        return { log_probs: new Tensor('float32', logits, [1, 99, 71]) };
      },
    };
    const ort: OrtModuleLike = {
      env: { wasm: {} },
      Tensor,
      InferenceSession: { create: async () => session },
    };
    const executor = new OrtGigaAmCtcExecutor('gigaam-test', 'wasm', {
      ecosystem: 'gigaam', architecture: 'gigaam-ctc', processorArchitecture: 'gigaam-fbank',
      encoderArchitecture: 'gigaam-conformer', decoderArchitecture: 'ctc', sampleRate: 16000,
      rawStride: 4, nMels: 64, featureHopSeconds: 0.01, vocabularySize: 71,
      languages: ['ru', 'en', 'kk', 'ky', 'uz'], tokenizer: { kind: 'sentencepiece', blankTokenId: 70 },
      nFft: 320, winLength: 320, hopLength: 160, featureLayout: 'mel-major',
    }, undefined);
    (executor as unknown as { loadStatePromise: Promise<unknown> }).loadStatePromise = Promise.resolve({
      ort, session, tokenizer: GigaAmTokenizer.fromText("▁ 0\na 2\n<blk> 70\n"), warnings: [],
    });

    const result = await executor.transcribe({ sampleRate: 16000, numberOfChannels: 1, numberOfFrames: 16000, durationSeconds: 1, channels: [new Float32Array(16000)] });
    const featureTensor = feeds.features as { dims: readonly number[] };
    const lengthTensor = feeds.feature_lengths as { data: BigInt64Array };
    expect(featureTensor.dims).toEqual([1, 64, 99]);
    expect(lengthTensor.data[0]).toBe(99n);
    expect(result.utteranceText).toBe('a');
  });

  it('decodes FP16 logits before CTC argmax', async () => {
    class Tensor<TData extends ArrayBufferView = ArrayBufferView> implements OrtTensorLike<TData> {
      constructor(readonly type: string, readonly data: TData, readonly dims: readonly number[]) {}
    }
    const session: OrtSessionLike = {
      async run() {
        const logits = new Uint16Array(99 * 71).fill(0xbc00); // -1
        logits[2] = 0x4900; // 10
        return { log_probs: new Tensor('float16', logits, [1, 99, 71]) };
      },
    };
    const executor = new OrtGigaAmCtcExecutor('gigaam-fp16-test', 'wasm', {
      ecosystem: 'gigaam', architecture: 'gigaam-ctc', processorArchitecture: 'gigaam-fbank', encoderArchitecture: 'gigaam-conformer', decoderArchitecture: 'ctc', sampleRate: 16000, rawStride: 4, nMels: 64, featureHopSeconds: 0.01, vocabularySize: 71, languages: ['ru'], tokenizer: { kind: 'sentencepiece', blankTokenId: 70 }, nFft: 320, winLength: 320, hopLength: 160, featureLayout: 'mel-major',
    }, undefined);
    (executor as unknown as { loadStatePromise: Promise<unknown> }).loadStatePromise = Promise.resolve({ ort: { env: { wasm: {} }, Tensor, InferenceSession: { create: async () => session } }, session, tokenizer: GigaAmTokenizer.fromText("▁ 0\na 2\n<blk> 70\n"), warnings: [] });

    await expect(executor.transcribe({ sampleRate: 16000, numberOfChannels: 1, numberOfFrames: 16000, durationSeconds: 1, channels: [new Float32Array(16000)] })).resolves.toMatchObject({ utteranceText: 'a' });
  });

  it('is discoverable but remains artifact-gated', async () => {
    const runtime = createBuiltInSpeechRuntime({ useManifestSources: false });
    expect(runtime.listModelFamilies().find((family) => family.family === 'gigaam-ctc')?.supports('gigaam-multilingual-ctc')).toBe(true);
    await expect(loadSpeechModel({ family: 'gigaam-ctc', modelId: 'gigaam-multilingual-ctc', backend: 'wasm' })).rejects.toThrow(/No GigaAM artifact source/);
    await expect(loadSpeechModel({ modelId: 'gigaam-multilingual-ctc', backend: 'wasm' })).rejects.toThrow(/No GigaAM artifact source/);
  });

  it('uses the published 64-bin, 320/320/160 feature geometry', () => {
    const processor = new GigaAmJsPreprocessor();
    const result = processor.process(new Float32Array(16000));

    expect(result.featureSize).toBe(64);
    expect(result.frameCount).toBe(99);
    expect(result.features.length).toBe(64 * 99);
  });

  it('decodes the character vocabulary and final CTC blank', () => {
    const tokenizer = GigaAmTokenizer.fromText("▁ 0\na 2\n' 1\n<blk> 70\n");

    expect(tokenizer.blankId).toBe(70);
    expect(tokenizer.decode([0, 2, 1, 70])).toBe("a'");
  });
});
