import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';
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

  it('packs float16 feature feeds when the official graph declares tensor(float16)', async () => {
    class Tensor<TData extends ArrayBufferView = ArrayBufferView> implements OrtTensorLike<TData> {
      constructor(readonly type: string, readonly data: TData, readonly dims: readonly number[]) {}
    }
    const feeds: Record<string, OrtTensorLike> = {};
    const session: OrtSessionLike = {
      inputMetadata: { features: { type: 'tensor(float16)' } },
      async run(input) {
        Object.assign(feeds, input);
        const logits = new Float32Array(99 * 71).fill(-10);
        for (let frame = 0; frame < 99; frame += 1) logits[frame * 71 + 70] = 10;
        logits[2] = 10;
        logits[70] = -10;
        return { log_probs: new Tensor('float32', logits, [1, 99, 71]) };
      },
    };
    const ort: OrtModuleLike = {
      env: { wasm: {} },
      Tensor,
      InferenceSession: { create: async () => session },
    };
    const executor = new OrtGigaAmCtcExecutor('gigaam-fp16-features', 'wasm', {
      ecosystem: 'gigaam', architecture: 'gigaam-ctc', processorArchitecture: 'gigaam-fbank',
      encoderArchitecture: 'gigaam-conformer', decoderArchitecture: 'ctc', sampleRate: 16000,
      rawStride: 4, nMels: 64, featureHopSeconds: 0.01, vocabularySize: 71,
      languages: ['ru'], tokenizer: { kind: 'sentencepiece', blankTokenId: 70 },
      nFft: 320, winLength: 320, hopLength: 160, featureLayout: 'mel-major',
    }, undefined);
    (executor as unknown as { loadStatePromise: Promise<unknown> }).loadStatePromise = Promise.resolve({
      ort, session, tokenizer: GigaAmTokenizer.fromText("▁ 0\na 2\n<blk> 70\n"), warnings: [],
    });
    await executor.transcribe({
      sampleRate: 16000, numberOfChannels: 1, numberOfFrames: 16000, durationSeconds: 1,
      channels: [new Float32Array(16000)],
    });
    expect(feeds.features?.type).toBe('float16');
    expect(feeds.features?.data).toBeInstanceOf(Uint16Array);
  });

  it('decodes FP16 logits before CTC argmax', async () => {
    class Tensor<TData extends ArrayBufferView = ArrayBufferView> implements OrtTensorLike<TData> {
      constructor(readonly type: string, readonly data: TData, readonly dims: readonly number[]) {}
    }
    const session: OrtSessionLike = {
      async run() {
        const logits = new Uint16Array(99 * 71).fill(0x5140); // 70 as fp16 is not this; use -1 then set blank
        logits.fill(0xbc00); // -1
        for (let frame = 0; frame < 99; frame += 1) logits[frame * 71 + 70] = 0x4900; // blank
        logits[2] = 0x4900; // 10
        logits[70] = 0xbc00;
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

  it('loads a local file:// vocabulary', async () => {
    const filePath = path.join(os.tmpdir(), 'gigaam-vocab-file-url.txt');
    fs.writeFileSync(filePath, "  0\na 1\n<blk> 2\n");
    const tokenizer = await GigaAmTokenizer.fromUrl(pathToFileURL(filePath).href);
    expect(tokenizer.blankId).toBe(2);
    expect(tokenizer.decode([1, 0, 1])).toBe('a a');
  });

  it('decodes the character vocabulary and final CTC blank', () => {
    const tokenizer = GigaAmTokenizer.fromText("▁ 0\na 2\n' 1\n<blk> 70\n");

    expect(tokenizer.blankId).toBe(70);
    expect(tokenizer.decode([0, 2, 1, 70])).toBe(" a'");
  });

  it('treats official character vocab blank as len(vocab)', () => {
    const tokenizer = GigaAmTokenizer.fromVocabulary([' ', 'a', 'b']);
    expect(tokenizer.blankId).toBe(3);
    expect(tokenizer.decode([1, 0, 2, 3])).toBe('a b');
  });

  it('packs mixed-length audio into one padded CTC graph call', async () => {
    class Tensor<TData extends ArrayBufferView = ArrayBufferView> implements OrtTensorLike<TData> {
      constructor(readonly type: string, readonly data: TData, readonly dims: readonly number[]) {}
    }
    let feeds: Record<string, OrtTensorLike> = {};
    const session: OrtSessionLike = {
      async run(input) {
        feeds = input as Record<string, OrtTensorLike>;
        const logits = new Float32Array(2 * 25 * 71).fill(-10);
        for (let frame = 0; frame < 25; frame += 1) {
          logits[frame * 71 + 70] = 10;
          logits[25 * 71 + frame * 71 + 70] = 10;
        }
        logits[2] = 20;
        logits[70] = -10;
        logits[25 * 71 + 1] = 20;
        logits[25 * 71 + 70] = -10;
        return { log_probs: new Tensor('float32', logits, [2, 25, 71]) };
      },
    };
    const ort: OrtModuleLike = { env: { wasm: {} }, Tensor, InferenceSession: { create: async () => session } };
    const executor = new OrtGigaAmCtcExecutor('gigaam-batch-test', 'wasm', {
      ecosystem: 'gigaam', architecture: 'gigaam-ctc', processorArchitecture: 'gigaam-fbank', encoderArchitecture: 'gigaam-conformer', decoderArchitecture: 'ctc', sampleRate: 16000, rawStride: 4, nMels: 64, featureHopSeconds: 0.01, vocabularySize: 71, languages: ['ru'], tokenizer: { kind: 'sentencepiece', blankTokenId: 70 }, nFft: 320, winLength: 320, hopLength: 160, featureLayout: 'mel-major',
    }, undefined);
    (executor as unknown as { loadStatePromise: Promise<unknown> }).loadStatePromise = Promise.resolve({ ort, session, tokenizer: GigaAmTokenizer.fromText("▁ 0\na 2\n' 1\n<blk> 70\n"), warnings: [] });

    const audio = (frames: number) => ({ sampleRate: 16000, numberOfChannels: 1, numberOfFrames: frames, durationSeconds: frames / 16000, channels: [new Float32Array(frames)] });
    const result = await executor.transcribeBatch([audio(16000), audio(8000)]);

    expect((feeds.features?.dims)).toEqual([2, 64, 99]);
    expect(Array.from((feeds.feature_lengths?.data as BigInt64Array))).toEqual([99n, 49n]);
    expect(result.map((item) => item.utteranceText)).toEqual(['a', "'"]);
  });

  it('returns an empty batch without loading or invoking the graph', async () => {
    let runCount = 0;
    const session: OrtSessionLike = {
      async run() {
        runCount += 1;
        return {};
      },
    };
    const ort: OrtModuleLike = { env: { wasm: {} }, InferenceSession: { create: async () => session } };
    const executor = new OrtGigaAmCtcExecutor('gigaam-empty-batch-test', 'wasm', {
      ecosystem: 'gigaam', architecture: 'gigaam-ctc', processorArchitecture: 'gigaam-fbank', encoderArchitecture: 'gigaam-conformer', decoderArchitecture: 'ctc', sampleRate: 16000, rawStride: 4, nMels: 64, featureHopSeconds: 0.01, vocabularySize: 71, languages: ['ru'], tokenizer: { kind: 'sentencepiece', blankTokenId: 70 }, nFft: 320, winLength: 320, hopLength: 160, featureLayout: 'mel-major',
    }, undefined);
    (executor as unknown as { loadStatePromise: Promise<unknown> }).loadStatePromise = Promise.resolve({ ort, session, tokenizer: GigaAmTokenizer.fromText("▁ 0\na 2\n' 1\n<blk> 70\n"), warnings: [] });

    await expect(executor.transcribeBatch([])).resolves.toEqual([]);
    expect(runCount).toBe(0);
  });
});
