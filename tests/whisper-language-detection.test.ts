import { describe, expect, it } from 'vitest';
import { selectWhisperLanguageFromLogits, WhisperOnnxExecutor } from '../src/models/whisper-seq2seq/index.js';
import type { TextTokenizer } from '../src/tokenizers/index.js';

function mockTokenizer(tokens: Record<number, string>): Pick<TextTokenizer, 'idsToTokens'> {
  return {
    idsToTokens(ids: readonly number[]): readonly string[] {
      return ids.map((id) => tokens[id] ?? '');
    },
  };
}

describe('selectWhisperLanguageFromLogits', () => {
  it('selects the highest-scoring language token from the final vocab slice', () => {
    const vocabSize = 51865;
    const logits = new Float32Array(vocabSize * 2);
    logits.fill(-10);
    logits[50259] = 100;
    logits[vocabSize + 50259] = 1;
    logits[vocabSize + 50268] = 8;

    const language = selectWhisperLanguageFromLogits(
      mockTokenizer({
        50259: '<|en|>',
        50268: '<|tr|>',
      }),
      logits,
      vocabSize,
    );

    expect(language).toBe('tr');
  });

  it('ignores special tokens that are not language tokens', () => {
    const vocabSize = 51865;
    const logits = new Float32Array(vocabSize);
    logits.fill(-10);
    logits[50259] = 1;
    logits[50268] = 8;

    const language = selectWhisperLanguageFromLogits(
      mockTokenizer({
        50259: '<|en|>',
        50268: '<|not-a-language|>',
      }),
      logits,
      vocabSize,
    );

    expect(language).toBe('en');
  });

  it('returns undefined when tokenizer cannot map language token ids', () => {
    const vocabSize = 51865;
    const logits = new Float32Array(vocabSize);
    logits[50259] = 10;

    expect(selectWhisperLanguageFromLogits({}, logits, vocabSize)).toBeUndefined();
  });

  it('is used by the splitgraph executor language probe', async () => {
    const vocabSize = 51865;
    const logits = new Float32Array(vocabSize * 2);
    logits.fill(-10);
    logits[50259] = 100;
    logits[vocabSize + 50259] = 1;
    logits[vocabSize + 50268] = 9;
    const outputTensor = {
      data: logits,
      dims: [1, 2, vocabSize],
      dispose() {},
    };
    const feedsSeen: any[] = [];
    const executor = new WhisperOnnxExecutor(
      'mock-whisper',
      {},
      {
        ecosystem: 'openai',
        architecture: 'whisper-seq2seq',
        processorArchitecture: 'whisper-mel',
        encoderArchitecture: 'whisper-transformer',
        decoderArchitecture: 'transformer-decoder',
        sampleRate: 16000,
        melBins: 80,
        maxSourcePositions: 1500,
        maxTargetPositions: 448,
        languages: ['auto'],
        tokenizer: { kind: 'tiktoken' },
      },
      'wasm',
      undefined,
    );

    const language = await (executor as any).detectLanguageFromEncoder(
      {
        tokenizer: {
          getTokenId(token: string): number | undefined {
            return token === '<|startoftranscript|>' ? 50258 : undefined;
          },
          idsToTokens(ids: readonly number[]): readonly string[] {
            return ids.map((id) => ({ 50259: '<|en|>', 50268: '<|tr|>' })[id] ?? '');
          },
        },
        ort: {
          Tensor: class Tensor {
            constructor(
              public readonly type: string,
              public readonly data: BigInt64Array,
              public readonly dims: readonly number[],
            ) {}
          },
        },
        decoderInitSession: {
          async run(feeds: Record<string, unknown>) {
            feedsSeen.push(feeds);
            return { logits: outputTensor };
          },
        },
      },
      { data: new Float32Array([1, 2, 3]), dims: [1, 1, 3] },
    );

    expect(language).toBe('tr');
    expect(feedsSeen).toHaveLength(1);
    expect((feedsSeen[0].input_ids as { data: BigInt64Array }).data[0]).toBe(50258n);
    expect(feedsSeen[0].encoder_hidden_states).toEqual({ data: new Float32Array([1, 2, 3]), dims: [1, 1, 3] });
  });
});
