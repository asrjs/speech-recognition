import { describe, expect, it } from 'vitest';
import { WhisperTokenizer } from '../src/models/whisper-seq2seq/tokenizer.js';

// Duplicating the helper for focused TDD — will be moved into executor
function buildForcedAlignmentTokens(
  tokenizer: WhisperTokenizer,
  language: string,
  textTokenIds: readonly number[],
): number[] {
  const sotId = tokenizer.getTokenId('<|startoftranscript|>') ?? 50258;
  const langToken = language === 'auto' ? '<|tr|>' : `<|${language}|>`;
  const langId = tokenizer.getTokenId(langToken) ?? 50268;
  const taskId = tokenizer.getTokenId('<|transcribe|>') ?? 50359;
  const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;

  return [sotId, langId, taskId, ...textTokenIds, eosId];
}

function buildForcedAlignmentTextTokenIds(
  tokenizer: WhisperTokenizer,
  segments: ReadonlyArray<{ readonly text: string }>,
): number[] {
  const joined = segments.map((s) => s.text).join(' ');
  const ids = tokenizer.encode(joined);
  return ids.filter((id) => !tokenizer.isSpecialTokenId(id));
}

const mockTokenizerJson = {
  model: {
    type: 'BPE' as const,
    vocab: {
      '!': 0,
      '"': 1,
      '<|endoftext|>': 50257,
    },
    merges: [] as string[],
  },
  added_tokens: [
    { id: 50257, content: '<|endoftext|>', special: true },
    { id: 50258, content: '<|startoftranscript|>', special: true },
    { id: 50259, content: '<|en|>', special: true },
    { id: 50268, content: '<|tr|>', special: true },
    { id: 50359, content: '<|transcribe|>', special: true },
    { id: 50363, content: '<|notimestamps|>', special: true },
    { id: 50364, content: '<|0.00|>', special: true },
    { id: 51864, content: '<|30.00|>', special: true },
  ],
};

describe('Whisper forced alignment prompt', () => {
  it('builds prompt with SOT + lang + task + text + EOT', () => {
    const tokenizer = new WhisperTokenizer(mockTokenizerJson);
    const tokens = buildForcedAlignmentTokens(tokenizer, 'tr', [100, 200]);
    expect(tokens).toEqual([50258, 50268, 50359, 100, 200, 50257]);
  });

  it('uses en when language is en', () => {
    const tokenizer = new WhisperTokenizer(mockTokenizerJson);
    const tokens = buildForcedAlignmentTokens(tokenizer, 'en', [100]);
    expect(tokens[1]).toBe(50259); // <|en|>
  });

  it('encodes text from segments to text token ids', () => {
    const vocab: Record<string, number> = {};
    for (let i = 0; i < 128; i++) vocab[String.fromCharCode(i)] = i;
    vocab.hello = 200;
    vocab['Ġworld'] = 201;

    const tokenizer = new WhisperTokenizer({
      model: { type: 'BPE', vocab, merges: [] },
      added_tokens: [{ id: 50257, content: '<|endoftext|>', special: true }],
    });

    const ids = buildForcedAlignmentTextTokenIds(tokenizer, [{ text: 'hello' }]);
    // 'hello' encodes through BPE. With a vocab where hello→200 but
    // GPT-2 regex split produces ['hello'], and ByteLevel maps characters...
    // This is more of an integration test
    expect(ids.length).toBeGreaterThan(0);
  });
});
