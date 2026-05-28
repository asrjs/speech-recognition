import { describe, it, expect, beforeAll } from 'vitest';
import { WhisperTokenizer } from '../src/models/whisper-seq2seq/tokenizer.js';

describe('WhisperTokenizer BPE encode', () => {
  let tokenizer: WhisperTokenizer;

  beforeAll(async () => {
    tokenizer = await WhisperTokenizer.fromUrl('file:///tmp/whisper-tiny-onnx/tokenizer.json');
  });

  it('encodes English text matching reference', () => {
    expect(tokenizer.encode('Hello world')).toEqual([15947, 1002]);
    expect(tokenizer.encode('This is a test')).toEqual([5723, 307, 257, 1500]);
    expect(tokenizer.encode('123')).toEqual([4762, 18]);
    expect(tokenizer.encode('!!!')).toEqual([4589]);
    expect(tokenizer.encode("it's")).toEqual([270, 311]);
  });

  it('encodes Turkish text matching reference', () => {
    expect(tokenizer.encode('Merhaba dünya')).toEqual([45757, 42016, 19378, 7457]);
    expect(tokenizer.encode("Türkiye'de yaşıyorum")).toEqual([51, 1655, 31137, 1116, 68, 16098, 15230]);
  });

  it('round-trips encode/decode for English and Turkish', () => {
    for (const text of ['Hello world', 'Merhaba dünya', "Türkiye'de yaşıyorum"]) {
      const ids = tokenizer.encode(text);
      expect(tokenizer.decode(ids)).toBe(text);
    }
  });

  it('handles special tokens alongside plain text', () => {
    const ids = tokenizer.encode('<|startoftranscript|>Hello world<|endoftext|>');
    expect(ids[0]).toBe(50258); // <|startoftranscript|>
    expect(ids[ids.length - 1]).toBe(50257); // <|endoftext|>
    expect(ids.slice(1, -1)).toEqual([15947, 1002]);
  });
});
