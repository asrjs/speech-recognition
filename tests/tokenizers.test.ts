import { describe, expect, it } from 'vitest';
import * as tokenizers from '../src/tokenizers.js';

describe('tokenizers exports', () => {
  it('exposes the expected tokenizer classes and types', () => {
    // Verify specific implementations are exported
    expect(tokenizers.StubTextTokenizer).toBeTypeOf('function');
    expect(tokenizers.StubSentencePieceTokenizer).toBeTypeOf('function');
    expect(tokenizers.BPETokenizer).toBeTypeOf('function');
    expect(tokenizers.Tiktokenizer).toBeTypeOf('function');
    expect(tokenizers.UTF8Tokenizer).toBeTypeOf('function');
  });

  it('can instantiate the StubTextTokenizer', () => {
    const tokenizer = new tokenizers.StubTextTokenizer('custom');
    expect(tokenizer.kind).toBe('custom');
    expect(tokenizer.decode([1, 2, 3])).toBe('tok1 tok2 tok3');
    expect(tokenizer.encode('hello world')).toEqual([1, 2]);
  });
});
