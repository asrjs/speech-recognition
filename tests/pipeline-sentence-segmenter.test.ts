import { joinTranscriptWords } from '../src/pipeline/sentence-segmenter.js';
import { describe, expect, it } from 'vitest';

describe('joinTranscriptWords', () => {
  it('joins standard words with spaces', () => {
    const words = [{ text: 'hello' }, { text: 'world' }];
    expect(joinTranscriptWords(words)).toBe('hello world');
  });

  it('attaches punctuation without preceding spaces', () => {
    const words = [{ text: 'Hello' }, { text: ',' }, { text: 'world' }, { text: '!' }];
    expect(joinTranscriptWords(words)).toBe('Hello, world!');
  });

  it('handles empty arrays', () => {
    expect(joinTranscriptWords([])).toBe('');
  });

  it('handles empty strings or undefined text', () => {
    const words = [{ text: 'first' }, { text: '' }, { text: undefined as any }, { text: 'second' }];
    expect(joinTranscriptWords(words)).toBe('first second');
  });
});
