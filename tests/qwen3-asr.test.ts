import { describe, expect, it } from 'vitest';
import { PcmAudioBuffer } from '../src/audio/index.js';
import {
  DEFAULT_QWEN3_ASR_CLASSIFICATION,
  DEFAULT_QWEN3_ASR_CONFIG,
  OrtQwen3AsrExecutor,
  Qwen3AsrFeatureProcessor,
  Qwen3AsrTokenizer,
  createQwen3AsrModelFamily,
  getQwenAudioTokenCount,
  parseOfficialQwen3AsrConfig,
  parseQwen3AsrConfig,
  resolveOfficialQwen3AsrDirectArtifacts,
  applyOfficialQwen3AsrGraphDefaults,
  type QwenOrtModuleLike,
  type QwenOrtSessionLike,
  type QwenOrtTensorLike,
  type Qwen3AsrExecutor,
} from '../src/models/qwen-asr/index.js';
import type { SpeechModelFactoryContext } from '../src/types/index.js';

describe('Qwen3-ASR feature contract', () => {
  it('pads the 128-bin frontend to an 800-frame graph window and preserves the valid mask', () => {
    const processor = new Qwen3AsrFeatureProcessor(DEFAULT_QWEN3_ASR_CONFIG);
    const audio = PcmAudioBuffer.fromMono(new Float32Array(8000), 16000);
    const result = processor.process(audio);

    expect(result.nMels).toBe(128);
    expect(result.frameCount).toBe(800);
    expect(result.features.length).toBe(128 * 800);
    expect(result.validFrameCount).toBe(50);
    expect(Array.from(result.inputFeaturesMask.slice(0, 50)).every((value) => value === 1)).toBe(
      true,
    );
    expect(Array.from(result.inputFeaturesMask.slice(50)).every((value) => value === 0)).toBe(true);
  });

  it('resamples non-16 kHz audio before computing valid frames', () => {
    const processor = new Qwen3AsrFeatureProcessor(DEFAULT_QWEN3_ASR_CONFIG);
    const result = processor.process(PcmAudioBuffer.fromMono(new Float32Array(4000), 8000));

    expect(result.sampleRate).toBe(16000);
    expect(result.durationSeconds).toBeCloseTo(0.5, 6);
    expect(result.validFrameCount).toBe(50);
  });

  it('matches the upstream placeholder-length formula at graph boundaries', () => {
    expect(getQwenAudioTokenCount(50)).toBe(7);
    expect(getQwenAudioTokenCount(100)).toBe(13);
    expect(getQwenAudioTokenCount(1050)).toBe(137);
    expect(getQwenAudioTokenCount(1100)).toBe(143);
    expect(getQwenAudioTokenCount(0)).toBe(0);
  });

  it('pads leftover frames to the next 100-frame chunk for official graphs', () => {
    const processor = new Qwen3AsrFeatureProcessor(parseOfficialQwen3AsrConfig());
    const result = processor.process(PcmAudioBuffer.fromMono(new Float32Array(168000), 16000));
    expect(result.validFrameCount).toBe(1050);
    expect(result.frameCount).toBe(1100);
    expect(getQwenAudioTokenCount(result.validFrameCount)).toBe(137);
  });
});

describe('Qwen3-ASR ByteLevel BPE tokenizer', () => {
  it('keeps chat/audio special tokens atomic while decoding byte-level text', () => {
    const tokenizer = Qwen3AsrTokenizer.fromJson(
      JSON.stringify({
        model: {
          vocab: { h: 0, i: 1, '!': 2 },
          merges: [],
        },
        added_tokens: [
          { id: 10, content: '<|audio_pad|>', special: true },
          { id: 11, content: '<|im_start|>', special: true },
          { id: 12, content: '<asr_text>', special: true },
        ],
      }),
    );

    expect(tokenizer.encode('<|im_start|>hi<asr_text>')).toEqual([11, 0, 1, 12]);
    expect(tokenizer.decode([0, 1])).toBe('hi');
    expect(tokenizer.decode([11, 0, 1], { skipSpecialTokens: true })).toBe('hi');
    expect(tokenizer.getTokenId('<|audio_pad|>')).toBe(10);
    expect(tokenizer.isSpecialTokenId(10)).toBe(true);
  });
});

describe('Qwen3-ASR model family boundary', () => {
  const backend = {
    id: 'wasm',
    displayName: 'test wasm',
    probeCapabilities: async () => ({
      id: 'wasm',
      displayName: 'test wasm',
      available: true,
      priority: 1,
      environments: ['node'] as const,
      acceleration: ['cpu'] as const,
      supportedPrecisions: ['fp32'] as const,
      supportsFp16: false,
      supportsInt8: true,
      supportsSharedArrayBuffer: false,
      requiresSharedArrayBuffer: false,
      fallbackSuitable: true,
      notes: [],
    }),
    createExecutionContext: async () => ({
      backendId: 'wasm',
      capabilities: {} as never,
      dispose() {},
    }),
  };

  it('recognizes Qwen ASR ids without claiming generic Qwen text models', () => {
    const family = createQwen3AsrModelFamily();
    expect(family.supports('Qwen/Qwen3-ASR-0.6B-hf')).toBe(true);
    expect(family.supports('Qwen/Qwen3-0.6B')).toBe(false);
    expect(family.classification).toMatchObject(DEFAULT_QWEN3_ASR_CLASSIFICATION);
  });

  it('requires a real artifact source instead of returning a scaffold transcript', async () => {
    const family = createQwen3AsrModelFamily();
    const model = await family.createModel(
      { family: 'qwen-asr', modelId: 'Qwen/Qwen3-ASR-0.6B-hf' },
      {
        runtime: {} as SpeechModelFactoryContext['runtime'],
        backend,
        hooks: {},
      },
    );
    const session = await model.createSession();
    await expect(session.transcribe(new Float32Array(1600))).rejects.toThrow(
      /No Qwen3-ASR artifact source/,
    );
    await model.dispose();
  });

  it('maps an injected reference executor through the canonical transcript contract', async () => {
    const executor: Qwen3AsrExecutor = {
      async transcribe() {
        return {
          utteranceText: 'merhaba',
          language: 'Turkish',
          isFinal: true,
          segments: [{ index: 0, text: 'merhaba', startTime: 0, endTime: 0.5 }],
        };
      },
      dispose() {},
    };
    const family = createQwen3AsrModelFamily({ dependencies: { executor } });
    const model = await family.createModel(
      { family: 'qwen-asr', modelId: 'Qwen/Qwen3-ASR-0.6B-hf' },
      {
        runtime: {} as SpeechModelFactoryContext['runtime'],
        backend,
        hooks: {},
      },
    );
    const session = await model.createSession();
    const result = await session.transcribe(new Float32Array(8000), { detail: 'segments' });

    expect(result.text).toBe('merhaba');
    expect(result.meta.modelFamily).toBe('qwen-asr');
    expect(result.meta.language).toBe('Turkish');
    expect(result.segments?.[0]?.text).toBe('merhaba');
    await model.dispose();
  });
});

describe('Qwen3-ASR ONNX prefill and KV-cache contract', () => {
  class FakeTensor<
    TData extends ArrayBufferView = ArrayBufferView,
  > implements QwenOrtTensorLike<TData> {
    readonly location = 'cpu';

    constructor(
      readonly type: string,
      readonly data: TData,
      readonly dims: readonly number[],
    ) {}

    dispose(): void {}
  }

  it('keeps the audio prompt in prefill and advances one-token KV steps', async () => {
    const config = parseQwen3AsrConfig('test', {
      languages: ['Turkish'],
      tokenizer: { kind: 'bpe', vocabSize: 5, eosTokenId: 4, padTokenId: 14 },
      graph: {
        ...DEFAULT_QWEN3_ASR_CONFIG.graph,
        numLayers: 1,
        numKvHeads: 1,
        headDim: 2,
        hiddenSize: 2,
        vocabularySize: 5,
        eosTokenIds: [4],
        audioPadTokenId: 12,
        audioStartTokenId: 10,
        audioEndTokenId: 11,
        imStartTokenId: 13,
        imEndTokenId: 14,
      },
    });
    const tokenizer = Qwen3AsrTokenizer.fromJson(
      JSON.stringify({
        model: { vocab: { unused: 0, h: 1, i: 2, '!': 3 }, merges: [] },
        added_tokens: [
          { id: 10, content: '<|audio_start|>', special: true },
          { id: 11, content: '<|audio_end|>', special: true },
          { id: 12, content: '<|audio_pad|>', special: true },
          { id: 13, content: '<|im_start|>', special: true },
          { id: 14, content: '<|im_end|>', special: true },
        ],
      }),
    );
    class FakeOrtTensor<TData extends ArrayBufferView = ArrayBufferView> extends FakeTensor<TData> {
      constructor(
        type: 'float16' | 'float32' | 'int32' | 'int64' | 'bool',
        data: TData,
        dims: readonly number[],
      ) {
        super(type, data, dims);
      }
    }
    const Tensor = FakeOrtTensor as unknown as QwenOrtModuleLike['Tensor'];
    const ort = {
      env: { wasm: {} },
      Tensor,
      InferenceSession: {
        async create(): Promise<QwenOrtSessionLike> {
          throw new Error('The test must inject both sessions.');
        },
      },
    } as unknown as QwenOrtModuleLike;
    const encoderSession: QwenOrtSessionLike = {
      async run(): Promise<Record<string, QwenOrtTensorLike>> {
        return {
          audio_embeddings: new FakeTensor(
            'float16',
            new Uint16Array([0x3c00, 0x3c00, 0, 0]),
            [1, 2, 2],
          ),
          audio_token_mask: new FakeTensor('bool', new Uint8Array([1, 0]), [1, 2]),
        };
      },
    };
    const decoderFeeds: Record<string, unknown>[] = [];
    let decoderCall = 0;
    const decoderSession: QwenOrtSessionLike = {
      async run(feeds): Promise<Record<string, QwenOrtTensorLike>> {
        decoderFeeds.push(feeds);
        const sequence = Number((feeds.input_ids as FakeTensor).dims[1]);
        const nextToken = decoderCall === 0 ? 1 : decoderCall === 1 ? 2 : decoderCall === 2 ? 3 : 4;
        const logits = new Float32Array(sequence * 5);
        logits[(sequence - 1) * 5 + nextToken] = 10;
        const totalSequence = Number((feeds.attention_mask as FakeTensor).dims[3]);
        decoderCall += 1;
        return {
          logits: new FakeTensor('float32', logits, [1, sequence, 5]),
          'present.0.key': new FakeTensor('float16', new Uint16Array(totalSequence * 2), [
            1,
            1,
            totalSequence,
            2,
          ]),
          'present.0.value': new FakeTensor('float16', new Uint16Array(totalSequence * 2), [
            1,
            1,
            totalSequence,
            2,
          ]),
        };
      },
    };
    const executor = new OrtQwen3AsrExecutor(
      'Qwen/Qwen3-ASR-0.6B-hf',
      config,
      'wasm',
      {
        source: {
          kind: 'direct',
          artifacts: { encoderUrl: 'encoder', decoderUrl: 'decoder', tokenizerUrl: 'tokenizer' },
        },
      },
      {
        ort,
        tokenizer,
        featureProcessor: new Qwen3AsrFeatureProcessor(config),
        encoderSession,
        decoderSession,
      },
    );

    const result = await executor.transcribe(
      PcmAudioBuffer.fromMono(new Float32Array(8000), 16000),
      {
        language: 'tr',
        maxNewTokens: 4,
      },
      {
        modelId: 'Qwen/Qwen3-ASR-0.6B-hf',
        classification: DEFAULT_QWEN3_ASR_CLASSIFICATION,
        config,
      },
    );

    expect(result.utteranceText).toBe('hi!');
    expect(result.language).toBe('Turkish');
    expect(decoderCall).toBe(4);
    expect(decoderFeeds[0]?.input_ids).toMatchObject({ dims: [1, expect.any(Number)] });
    expect((decoderFeeds[0]?.audio_mask as FakeTensor).dims).toEqual([1, expect.any(Number), 1]);
    expect((decoderFeeds[1]?.input_ids as FakeTensor).dims).toEqual([1, 1]);
    expect((decoderFeeds[1]?.attention_mask as FakeTensor).dims[3]).toBeGreaterThan(1);
    expect(result.metrics?.decoderStepCount).toBe(3);
    await executor.dispose();
  });
});

describe('Qwen3-ASR official stacked artifact defaults', () => {
  it('defaults official encoder URLs to the dynamic graph, not static T=1100', () => {
    const artifacts = resolveOfficialQwen3AsrDirectArtifacts({
      baseUrl: '/qwen3-asr-official',
    });
    expect(artifacts.encoderUrl).toBe('/qwen3-asr-official/audio-encoder-dynamic.onnx');
    expect(artifacts.decoderStepUrl).toBe('/qwen3-asr-official/decoder-step.onnx');
    expect(
      resolveOfficialQwen3AsrDirectArtifacts({
        baseUrl: '/qwen3-asr-official',
        encoder: 'static-t1100',
      }).encoderUrl,
    ).toBe('/qwen3-asr-official/audio-encoder-static-t1100.onnx');
  });

  it('applies pad-to-100 stacked graph defaults when decoder-step artifacts are present', () => {
    const config = applyOfficialQwen3AsrGraphDefaults(DEFAULT_QWEN3_ASR_CONFIG, {
      kind: 'direct',
      artifacts: {
        encoderUrl: '/qwen3-asr-official/audio-encoder-dynamic.onnx',
        decoderUrl: '/qwen3-asr-official/decoder-prefill.onnx',
        decoderStepUrl: '/qwen3-asr-official/decoder-step.onnx',
        tokenizerUrl: '/qwen3-asr-official/tokenizer/tokenizer.json',
      },
    });
    expect(config.graph.kvLayout).toBe('stacked');
    expect(config.graph.audioFramesMultiple).toBe(100);
    expect(config.graph.pastSeedLength).toBe(0);
    expect(parseOfficialQwen3AsrConfig().graph.audioWindowFrames).not.toBe(1100);
  });
});
