import { describe, expect, it } from 'vitest';

import { parseNemoAedConfig } from '../src/models/nemo-aed/config.js';
import { CanaryTokenizer, type CanaryTokenizerPayload } from '../src/models/nemo-aed/tokenizer.js';

function createTokenizerPayload(): CanaryTokenizerPayload {
  const specialPieces = Array.from({ length: 80 }, (_, index) => `<special:${index}>`);
  return {
    kind: 'canary-aggregate-tokenizer',
    version: 1,
    prompt_format: 'canary2',
    vocab_size: 84,
    langs: ['spl_tokens', 'en'],
    language_codes: ['en'],
    bos_id: 4,
    eos_id: 3,
    pad_id: 2,
    special_tokens: {
      '<pad>': 2,
      '<|endoftext|>': 3,
      '<|startoftranscript|>': 4,
      '<|pnc|>': 5,
      '<|nopnc|>': 6,
      '<|startofcontext|>': 7,
      '<|itn|>': 8,
      '<|noitn|>': 9,
      '<|timestamp|>': 10,
      '<|notimestamp|>': 11,
      '<|diarize|>': 12,
      '<|nodiarize|>': 13,
      '<|emo:undefined|>': 16,
      '<|en|>': 62,
      '<|de|>': 76,
    },
    subtokenizers: {
      spl_tokens: {
        offset: 0,
        size: 80,
        pieces: specialPieces,
      },
      en: {
        offset: 80,
        size: 4,
        pieces: ['\u2581Hello', ',', '\u2581world', '!'],
      },
    },
  };
}

describe('CanaryTokenizer', () => {
  it('builds the default canary2 prompt ids and decodes aggregate pieces', () => {
    const tokenizer = CanaryTokenizer.fromPayload(createTokenizerPayload());
    const config = parseNemoAedConfig('test-canary');

    const prompt = tokenizer.resolvePromptSettings(config, {
      targetLanguage: 'de',
    });

    expect(tokenizer.buildPromptIds(prompt)).toEqual([7, 4, 16, 62, 76, 5, 9, 11, 13]);
    expect(tokenizer.decode([80, 81, 82, 83, 3])).toBe('Hello, world!');
    expect(tokenizer.idsToTokens([80, 81, 82, 83])).toEqual([
      '\u2581Hello',
      ',',
      '\u2581world',
      '!',
    ]);
  });

  it('rejects non-empty decoder context until text-side prompt encoding is implemented', () => {
    const tokenizer = CanaryTokenizer.fromPayload(createTokenizerPayload());
    const config = parseNemoAedConfig('test-canary');
    const prompt = tokenizer.resolvePromptSettings(config, {
      decoderContext: 'seed text',
    });

    expect(() => tokenizer.buildPromptIds(prompt)).toThrow('decoder_context');
  });

  it('accepts NeMo-style aliases for languages, task, pnc, and timestamp toggles', () => {
    const tokenizer = CanaryTokenizer.fromPayload(createTokenizerPayload());
    const config = parseNemoAedConfig('test-canary');

    const prompt = tokenizer.resolvePromptSettings(config, {
      source_lang: 'de',
      task: 'asr',
      pnc: 'no',
      timestamp: 'yes',
    });

    expect(prompt).toMatchObject({
      sourceLanguage: 'de',
      targetLanguage: 'de',
      punctuate: false,
      timestamps: true,
    });
    expect(tokenizer.buildPromptIds(prompt)).toEqual([7, 4, 16, 76, 76, 6, 9, 10, 13]);
  });
});
