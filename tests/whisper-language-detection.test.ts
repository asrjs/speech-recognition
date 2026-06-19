import { describe, expect, it } from 'vitest';
import { selectWhisperLanguageFromLogits, WhisperOnnxExecutor } from '../src/models/whisper-seq2seq/index.js';
import {
  resolveWhisperLanguageCode,
  resolveWhisperLanguageTokenId,
} from '../src/models/whisper-seq2seq/executor.js';
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

  it('detects language via merged-decoder probe', async () => {
    const vocabSize = 51865;
    const logits = new Float32Array(vocabSize);
    logits.fill(-10);
    logits[50268] = 9; // <|tr|>

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

    const language = await (executor as any).detectLanguageFromMergedDecoder(
      {
        tokenizer: {
          getTokenId(token: string): number | undefined {
            return token === '<|startoftranscript|>' ? 50258 : undefined;
          },
          idsToTokens(ids: readonly number[]): readonly string[] {
            return ids.map((id) => ({ 50259: '<|en|>', 50268: '<|tr|>' }[id] ?? ''));
          },
        },
        decoderSession: {
          async run(_feeds: Record<string, unknown>) {
            return {
              logits: { data: logits, dims: [1, 1, vocabSize] },
              'present.0.decoder.key': { data: new Float32Array([1]), dims: [1, 1, 1, 1] },
              'present.0.decoder.value': { data: new Float32Array([1]), dims: [1, 1, 1, 1] },
            };
          },
          inputNames: ['input_ids', 'encoder_hidden_states', 'use_cache_branch'],
        },
        ort: {
          Tensor: class Tensor {
            constructor(
              public readonly type: string,
              public readonly data: BigInt64Array | Uint8Array | Float32Array,
              public readonly dims: readonly number[],
            ) {}
          },
        },
        modelConfig: { decoderLayers: 1, decoderAttentionHeads: 1, headDim: 1 },
      },
      { data: new Float32Array([1, 2, 3]), dims: [1, 1, 3] },
    );

    expect(language).toBe('tr');
  });

  it('falls back from unresolved auto language to English, not Turkish or <|auto|>', () => {
    const seenTokens: string[] = [];
    const tokenizer = {
      getTokenId(token: string): number | undefined {
        seenTokens.push(token);
        return token === '<|en|>' ? 50259 : undefined;
      },
    };

    const language = resolveWhisperLanguageCode('auto', ['auto']);
    const tokenId = resolveWhisperLanguageTokenId(tokenizer, language);

    expect(language).toBe('en');
    expect(tokenId).toBe(50259);
    expect(seenTokens).toEqual(['<|en|>']);
  });

  it('keeps explicit or detected Turkish language tokens when the tokenizer supports them', () => {
    const tokenizer = {
      getTokenId(token: string): number | undefined {
        return token === '<|tr|>' ? 50268 : token === '<|en|>' ? 50259 : undefined;
      },
    };

    const language = resolveWhisperLanguageCode('tr', ['auto']);

    expect(language).toBe('tr');
    expect(resolveWhisperLanguageTokenId(tokenizer, language)).toBe(50268);
  });
});
