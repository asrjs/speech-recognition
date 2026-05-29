/**
 * Tests for chunking fixed-window + post-processing extras.
 * T2: FixedWindowChunker, T3: dedup/normalize/sentence-boundary.
 */

import { describe, it, expect } from 'vitest';
import { FixedWindowChunker } from '../src/chunking/fixed-window.js';
import { deduplicateWords, normalizeText, buildSentences } from '../src/post-processing/extras.js';

// ---------------------------------------------------------------------------
// FixedWindowChunker
// ---------------------------------------------------------------------------

describe('FixedWindowChunker', () => {
  it('returns single window for short audio', () => {
    const chunker = new FixedWindowChunker();
    const audio = new Float32Array(16000 * 10); // 10s
    const windows = chunker.chunk(audio, 16000);
    expect(windows).toHaveLength(1);
    expect(windows[0]!.durationSeconds).toBeCloseTo(10, 0);
  });

  it('splits long audio into windows with overlap', () => {
    const chunker = new FixedWindowChunker({ windowDurationMs: 30000, hopDurationMs: 28000 });
    const audio = new Float32Array(16000 * 90); // 90s
    const windows = chunker.chunk(audio, 16000);
    // 90s → window0: 0-30s, window1: 28-58s, window2: 56-86s, window3: 84-90s
    expect(windows.length).toBeGreaterThanOrEqual(3);
    // Overlap: window1 start < window0 end
    expect(windows[1]!.startSeconds).toBeLessThan(windows[0]!.endSeconds);
  });
});

// ---------------------------------------------------------------------------
// deduplicateWords
// ---------------------------------------------------------------------------

describe('deduplicateWords', () => {
  it('removes overlapping duplicate words', () => {
    const words = [
      { word: 'hello', start: 0, end: 1, probability: 0.9 },
      { word: 'hello', start: 0.5, end: 1.5, probability: 0.85 },
      { word: 'world', start: 2, end: 3, probability: 0.8 },
    ];
    const result = deduplicateWords(words);
    expect(result).toHaveLength(2);
    // Keeps first 'hello' because probability is higher
    expect(result[0]!.probability).toBe(0.9);
  });

  it('keeps distinct non-overlapping words', () => {
    const words = [
      { word: 'one', start: 0, end: 1, probability: 0.9 },
      { word: 'two', start: 2, end: 3, probability: 0.8 },
    ];
    expect(deduplicateWords(words)).toHaveLength(2);
  });

  it('case-insensitive comparison', () => {
    const words = [
      { word: 'Hello', start: 0, end: 1, probability: 0.9 },
      { word: 'hello', start: 0.5, end: 1.5, probability: 0.85 },
    ];
    expect(deduplicateWords(words)).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// normalizeText
// ---------------------------------------------------------------------------

describe('normalizeText', () => {
  it('collapses multiple spaces', () => {
    expect(normalizeText('hello   world  test')).toBe('hello world test');
  });

  it('trims whitespace', () => {
    expect(normalizeText('  hello  ')).toBe('hello');
  });
});

// ---------------------------------------------------------------------------
// buildSentences
// ---------------------------------------------------------------------------

describe('buildSentences', () => {
  const makeWord = (word: string, start: number, end: number) => ({
    word, start, end, probability: 0.9,
  });

  it('splits at punctuation endings', () => {
    const words = [
      makeWord('Hello', 0, 0.5),
      makeWord('world.', 0.5, 1.0),
      makeWord('How', 1.2, 1.5),
      makeWord('are', 1.5, 1.8),
      makeWord('you?', 1.8, 2.0),
    ];
    const sentences = buildSentences(words);
    expect(sentences).toHaveLength(2);
    expect(sentences[0]!.text).toBe('Hello world.');
    expect(sentences[1]!.text).toBe('How are you?');
  });

  it('splits at 3+ second gaps', () => {
    const words = [
      makeWord('First', 0, 0.5),
      makeWord('sentence', 0.5, 1.0),
      makeWord('Second', 5.0, 5.5),
      makeWord('sentence', 5.5, 6.0),
    ];
    const sentences = buildSentences(words);
    expect(sentences).toHaveLength(2);
  });
});
