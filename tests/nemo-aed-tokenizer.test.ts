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

  it('builds the native legacy canary prompt and keeps task explicit', () => {
    const payload = createTokenizerPayload();
    const specialTokens = {
      '<pad>': 1,
      '<|endoftext|>': 2,
      '<|startoftranscript|>': 3,
      '<|en|>': 4,
      '<|es|>': 5,
      '<|de|>': 6,
      '<|fr|>': 7,
      '<|transcribe|>': 8,
      '<|translate|>': 9,
      '<|nopnc|>': 10,
      '<|pnc|>': 11,
      '<|nospeech|>': 12,
    };
    const pieces = Array.from({ length: 32 }, (_, index) => `<|spltoken${index}|>`);
    for (const [token, id] of Object.entries(specialTokens)) pieces[id] = token;
    const tokenizer = CanaryTokenizer.fromPayload({
      ...payload,
      prompt_format: 'canary',
      langs: ['spl_tokens', 'en', 'de', 'es', 'fr'],
      language_codes: ['en', 'de', 'es', 'fr'],
      bos_id: 3,
      eos_id: 2,
      pad_id: 1,
      special_tokens: specialTokens,
      subtokenizers: {
        spl_tokens: { offset: 0, size: 32, pieces },
        en: { offset: 32, size: 13, pieces: Array.from({ length: 13 }, (_, i) => `en-${i}`) },
        de: { offset: 45, size: 13, pieces: Array.from({ length: 13 }, (_, i) => `de-${i}`) },
        es: { offset: 58, size: 13, pieces: Array.from({ length: 13 }, (_, i) => `es-${i}`) },
        fr: { offset: 71, size: 13, pieces: Array.from({ length: 13 }, (_, i) => `fr-${i}`) },
      },
    });
    const config = parseNemoAedConfig('nvidia/canary-1b', {
      promptFormat: 'canary',
      promptDefaults: [{
        sourceLanguage: 'en', targetLanguage: 'en', task: 'asr', decoderContext: '',
        emotion: '<|emo:undefined|>', punctuate: true, inverseTextNormalization: false,
        timestamps: false, diarize: false,
      }],
    });

    const asr = tokenizer.resolvePromptSettings(config, { task: 'asr' });
    expect(asr.task).toBe('asr');
    expect(tokenizer.buildPromptIds(asr)).toEqual([3, 4, 8, 4, 11]);

    const translation = tokenizer.resolvePromptSettings(config, {
      source_lang: 'en', target_lang: 'de', task: 's2t_translation', pnc: 'no',
    });
    expect(translation.task).toBe('translation');
    expect(tokenizer.buildPromptIds(translation)).toEqual([3, 4, 9, 6, 10]);
  });

  it('rejects Canary 2-only prompt controls for the legacy canary format', () => {
    const payload = createTokenizerPayload();
    const tokenizer = CanaryTokenizer.fromPayload({ ...payload, prompt_format: 'unsupported' });
    const config = parseNemoAedConfig('test-canary');
    expect(() => tokenizer.buildPromptIds(tokenizer.resolvePromptSettings(config))).toThrow(
      /Unsupported Canary prompt format/,
    );
  });
});
