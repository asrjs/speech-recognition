import { describe, expect, it } from 'vitest';
import { WhisperOnnxExecutor } from '../src/models/whisper-seq2seq/executor.js';
import { WhisperTokenizer } from '../src/models/whisper-seq2seq/tokenizer.js';

function createTokenizer(): WhisperTokenizer {
  return new WhisperTokenizer({
    model: {
      type: 'BPE',
      vocab: { '!': 0 },
      merges: [],
    },
    added_tokens: [
      { id: 50257, content: '<|endoftext|>', special: true },
      { id: 50258, content: '<|startoftranscript|>', special: true },
      { id: 50259, content: '<|en|>', special: true },
      { id: 50359, content: '<|translate|>', special: true },
      { id: 50360, content: '<|transcribe|>', special: true },
      { id: 50363, content: '<|notimestamps|>', special: true },
    ],
  });
}

class FakeTensor {
  readonly type: string;
  readonly data: ArrayBufferView;
  readonly dims: readonly number[];

  constructor(type: string, data: ArrayBufferView, dims: readonly number[]) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}

function createExecutor(): WhisperOnnxExecutor {
  return new WhisperOnnxExecutor(
    'whisper-merged-test',
    { ecosystem: 'openai', family: 'whisper-seq2seq', task: 'transcribe' },
    {
      ecosystem: 'openai',
      architecture: 'whisper-seq2seq',
      processorArchitecture: 'whisper-mel',
      encoderArchitecture: 'whisper-transformer',
      decoderArchitecture: 'transformer-decoder',
      sampleRate: 16000,
      melBins: 80,
      maxSourcePositions: 3000,
      maxTargetPositions: 448,
      vocabularySize: 51865,
      languages: ['en'],
      tokenizer: { kind: 'tiktoken', vocabSize: 51865 },
    },
    'wasm',
    undefined,
  );
}

describe('merged Whisper alignment boundaries', () => {
  it('keeps finalized merged word timestamps inside the input clip', async () => {
    const executor = createExecutor() as unknown as {
      finalizeWordTimestamps: (
        alignedWords: readonly {
          readonly index: number;
          readonly text: string;
          readonly startTime: number;
          readonly endTime: number;
        }[],
        tokens: readonly unknown[],
        tokenizerValue: unknown,
        language: string,
        options: Record<string, never>,
        audio: { readonly durationSeconds: number },
        warnings: unknown[],
      ) => Promise<readonly { readonly startTime: number; readonly endTime: number }[]>;
    };

    const words = await executor.finalizeWordTimestamps(
      [
        { index: 0, text: 'overlap', startTime: 5.9, endTime: 6.8 },
        { index: 1, text: 'padding', startTime: 6.8, endTime: 8.0 },
      ],
      [],
      { getTokenId: () => undefined },
      'en',
      {},
      { durationSeconds: 6 },
      [],
    );

    expect(words).toEqual([{ index: 0, text: 'overlap', startTime: 5.9, endTime: 6 }]);
  });

  it('keeps timestamp-token segments inside the input clip', () => {
    const executor = createExecutor() as unknown as {
      buildSegments: (
        tokens: readonly { readonly id: number; readonly confidence?: number }[],
        tokenizerValue: unknown,
        noTimestamps: boolean,
        durationSeconds: number,
      ) => readonly { readonly startTime: number; readonly endTime: number }[];
    };
    const tokenizer = {
      isTimestampTokenId: (id: number) => id >= 50_000,
      timestampTokenIdToSeconds: (id: number) => (id === 50_000 ? 5.9 : 8),
      decode: () => 'overlap padding',
    };

    const segments = executor.buildSegments(
      [
        { id: 50_000, confidence: 0.9 },
        { id: 11, confidence: 0.9 },
        { id: 12, confidence: 0.9 },
        { id: 50_001, confidence: 0.9 },
      ],
      tokenizer,
      false,
      6,
    );

    expect(segments).toEqual([
      { index: 0, text: 'overlap padding', startTime: 5.9, endTime: 6, confidence: 0.9 },
    ]);
  });

  it('uses the reference prompt, causal logit rows, and declared cache inputs', async () => {
    const tokenizer = createTokenizer();
    const feedsSeen: Record<string, FakeTensor> = {};
    const vocabSize = 8;
    const forcedLength = 7;
    const logits = new Float32Array(forcedLength * vocabSize);
    for (let row = 0; row < forcedLength; row++) {
      logits[row * vocabSize] = 100 + row;
    }

    const decoderSession = {
      inputNames: [
        'input_ids',
        'encoder_hidden_states',
        'past_key_values.0.decoder.key',
        'past_key_values.0.decoder.value',
        'past_key_values.0.encoder.key',
        'past_key_values.0.encoder.value',
        'use_cache_branch',
      ],
      inputMetadata: [{ name: 'encoder_hidden_states', type: 'float16' }],
      run: async (feeds: Record<string, FakeTensor>) => {
        Object.assign(feedsSeen, feeds);
        return {
          logits: new FakeTensor('float32', logits, [1, forcedLength, vocabSize]),
          'cross_attentions.0': new FakeTensor(
            'float32',
            new Float32Array(1 * 1 * forcedLength * 4),
            [1, 1, forcedLength, 4],
          ),
        };
      },
    };
    const ort = {
      env: { wasm: {} },
      Tensor: FakeTensor,
      InferenceSession: { create: async () => decoderSession },
    };
    const loaded = {
      ort,
      tokenizer,
      encoderSession: decoderSession,
      decoderSession,
      generationConfig: { alignmentHeads: [] },
      modelConfig: {
        medianFilterWidth: 7,
        decoderLayers: 1,
        decoderAttentionHeads: 1,
        dModel: 4,
        headDim: 4,
      },
      warnings: [],
      isSplitGraph: false,
    } as never;
    const encoderHiddenStates = new FakeTensor('float32', new Float32Array(4), [1, 2, 2]);

    const alignment = await (
      createExecutor() as unknown as {
        runForcedAlignment: (
          loadedState: unknown,
          encoder: unknown,
          language: string,
          textTokens: number[],
          task: 'transcribe' | 'translate',
        ) => Promise<{ readonly logitsForText: Float32Array }>;
      }
    ).runForcedAlignment(loaded, encoderHiddenStates, 'en', [11, 12], 'translate');

    const inputIds = feedsSeen.input_ids.data as BigInt64Array;
    expect(Array.from(inputIds, Number)).toEqual([50258, 50259, 50359, 50363, 11, 12, 50257]);
    expect(feedsSeen.encoder_hidden_states.type).toBe('float16');
    expect(feedsSeen['past_key_values.0.decoder.key'].dims).toEqual([1, 1, 0, 4]);
    expect(feedsSeen.use_cache_branch.data).toEqual(new Uint8Array([1]));

    // Row 3 (<|notimestamps|>) predicts text token 11; row 4 predicts token 12.
    expect(alignment.logitsForText[0]).toBe(103);
    expect(alignment.logitsForText[vocabSize]).toBe(104);
  });

  it('does not add cache feeds to a merged decoder that does not declare them', async () => {
    const tokenizer = createTokenizer();
    const feedsSeen: Record<string, FakeTensor> = {};
    const decoderSession = {
      inputNames: ['input_ids', 'encoder_hidden_states'],
      inputMetadata: [{ name: 'encoder_hidden_states', type: 'float32' }],
      run: async (feeds: Record<string, FakeTensor>) => {
        Object.assign(feedsSeen, feeds);
        return {
          logits: new FakeTensor('float32', new Float32Array(4), [1, 2, 2]),
        };
      },
    };
    const ort = {
      env: { wasm: {} },
      Tensor: FakeTensor,
      InferenceSession: { create: async () => decoderSession },
    };
    const loaded = {
      ort,
      tokenizer,
      encoderSession: decoderSession,
      decoderSession,
      generationConfig: { alignmentHeads: [] },
      modelConfig: {
        medianFilterWidth: 7,
        decoderLayers: 1,
        decoderAttentionHeads: 1,
        dModel: 4,
        headDim: 4,
      },
      warnings: [],
      isSplitGraph: false,
    } as never;

    await (
      createExecutor() as unknown as {
        runForcedAlignment: (
          loadedState: unknown,
          encoder: unknown,
          language: string,
          textTokens: number[],
          task: 'transcribe' | 'translate',
        ) => Promise<unknown>;
      }
    ).runForcedAlignment(
      loaded,
      new FakeTensor('float32', new Float32Array(4), [1, 2, 2]),
      'en',
      [11],
      'transcribe',
    );

    expect(Object.keys(feedsSeen)).toEqual(['input_ids', 'encoder_hidden_states']);
  });

  it('does not add cache feeds to the regular merged decode step when absent', async () => {
    const feedsSeen: Record<string, FakeTensor> = {};
    const decoderSession = {
      inputNames: ['input_ids', 'encoder_hidden_states'],
      run: async (feeds: Record<string, FakeTensor>) => {
        Object.assign(feedsSeen, feeds);
        return {
          logits: new FakeTensor('float32', new Float32Array([0, 1]), [1, 1, 2]),
        };
      },
    };
    const loaded = {
      ort: { Tensor: FakeTensor },
      tokenizer: createTokenizer(),
      encoderSession: decoderSession,
      decoderSession,
      modelConfig: {
        medianFilterWidth: 7,
        decoderLayers: 1,
        decoderAttentionHeads: 1,
        dModel: 4,
        headDim: 4,
      },
    } as never;

    await (
      createExecutor() as unknown as {
        runDecoderStep: (
          loadedState: unknown,
          encoder: unknown,
          generatedTokens: readonly number[],
          pastKeyValues: Record<string, never>,
          isFirstStep: boolean,
        ) => Promise<unknown>;
      }
    ).runDecoderStep(
      loaded,
      new FakeTensor('float32', new Float32Array(4), [1, 2, 2]),
      [50258],
      {},
      true,
    );

    expect(Object.keys(feedsSeen)).toEqual(['input_ids', 'encoder_hidden_states']);
  });

  it('uses text rows from cross-attention and crops padded frames to audio duration', async () => {
    const tokenizer = {
      encode: (text: string) =>
        text === 'hello world' ? [11, 12] : text === 'hello' ? [11] : [12],
      decode: (ids: readonly number[]) =>
        ids.length === 2 ? 'hello world' : ids[0] === 11 ? 'hello' : 'world',
      getTokenId: () => undefined,
      isSpecialTokenId: () => false,
      isTimestampTokenId: () => false,
    };
    const attention = new Float32Array(7 * 8);
    // Prompt rows deliberately point at the padded tail. The alignment code
    // must keep the no-timestamps anchor row 3, followed by the causal text
    // prediction rows 4 and 5.
    attention[0 * 8 + 7] = 1;
    attention[1 * 8 + 7] = 1;
    attention[2 * 8 + 7] = 1;
    attention[3 * 8 + 0] = 1;
    attention[4 * 8 + 0] = 1;
    attention[5 * 8 + 1] = 1;
    attention[6 * 8 + 7] = 1;

    const executor = createExecutor() as unknown as {
      runForcedAlignment: () => Promise<{
        readonly crossAttentions: readonly {
          readonly data: Float32Array;
          readonly dims: readonly number[];
        }[];
        readonly logitsForText: Float32Array;
      }>;
      computeAttentionWordTimestamps: (
        loadedState: unknown,
        encoder: unknown,
        tokenizerValue: unknown,
        tokenDetails: readonly unknown[],
        segments: readonly { readonly text: string }[],
        language: string,
        options: { readonly task: 'transcribe' | 'translate' },
        audioDurationSeconds: number,
      ) => Promise<readonly { readonly startTime: number; readonly endTime: number }[] | undefined>;
    };
    executor.runForcedAlignment = async () => ({
      crossAttentions: [{ data: attention, dims: [1, 1, 7, 8] }],
      logitsForText: new Float32Array(0),
    });

    const words = await executor.computeAttentionWordTimestamps(
      {
        generationConfig: { alignmentHeads: [{ layer: 0, head: 0 }] },
        modelConfig: { medianFilterWidth: 1 },
      },
      { dims: [1, 4, 2] },
      tokenizer,
      [],
      [{ text: 'hello world' }],
      'en',
      { task: 'transcribe' },
      0.06,
    );

    expect(words).toHaveLength(2);
    expect(words?.[0]?.startTime).toBeLessThanOrEqual(0.02);
    expect(words?.[1]?.endTime).toBeLessThanOrEqual(0.06);
  });
});
