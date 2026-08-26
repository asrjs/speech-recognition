import { describe, expect, it } from 'vitest';
import {
  SenseVoiceJsPreprocessor,
  SenseVoiceTokenizer,
  createSenseVoicePrompt,
  resolveSenseVoiceLanguage,
} from '../src/models/sensevoice/index.js';

describe('SenseVoice prompt contract', () => {
  it('maps supported languages and ITN IDs to the ONNX prompt contract', () => {
    expect(createSenseVoicePrompt({ language: 'en' })).toEqual({
      language: 'en',
      languageId: 4,
      textnorm: 'withitn',
      textnormId: 14,
    });
    expect(createSenseVoicePrompt({ language: 'ja', useItn: false })).toMatchObject({
      languageId: 11,
      textnormId: 15,
    });
  });

  it('falls back safely for unsupported language values', () => {
    expect(resolveSenseVoiceLanguage('tr')).toBe('auto');
    expect(resolveSenseVoiceLanguage(undefined)).toBe('auto');
  });
});

describe('SenseVoice tokenizer and frontend', () => {
  it('decodes SentencePiece pieces while dropping CTC blank and prompt tags', () => {
    const tokenizer = SenseVoiceTokenizer.fromText(
      '<blk> 0\n<|en|> 1\n▁hello 2\n▁world 3\n',
    );
    expect(tokenizer.blankId).toBe(0);
    expect(tokenizer.decode([1, 2, 3, 0])).toBe('hello world');
  });

  it('emits raw 80-bin time-major fbank frames', () => {
    const processor = new SenseVoiceJsPreprocessor();
    const audio = new Float32Array(16000);
    for (let index = 0; index < audio.length; index += 1) {
      audio[index] = Math.sin((2 * Math.PI * 220 * index) / 16000) * 0.1;
    }
    const result = processor.process(audio);
    expect(result.featureSize).toBe(80);
    expect(result.frameCount).toBeGreaterThan(0);
    expect(result.validFrameCount).toBe(result.frameCount);
    expect(result.features.length).toBe(result.frameCount * 80);
  });
});
