/**
 * Tests for standalone post-processing/ module.
 * Phase C: segment-merger moved from whisper-seq2seq.
 */

import { describe, it, expect } from 'vitest';
import { mergeSegments } from '../src/post-processing/index.js';

describe('mergeSegments', () => {
  it('adjusts timestamps by offset', () => {
    const chunks = [{
      timeOffsetSeconds: 10.0,
      segments: [{ id: 0, start: 0.0, end: 5.0, text: 'hello world', words: [
        { start: 0.0, end: 0.5, word: 'hello', probability: 0.9 },
      ]}],
      words: [{ start: 0.0, end: 0.5, word: 'hello', probability: 0.9 }],
    }];
    const result = mergeSegments(chunks);
    expect(result.segments[0]!.start).toBeCloseTo(10.0, 1);
    expect(result.words[0]!.start).toBeCloseTo(10.0, 1);
  });

  it('merges multiple chunks', () => {
    const chunks = [
      { timeOffsetSeconds: 0.0, segments: [{ id: 0, start: 0, end: 5, text: 'first', words: [
        { start: 0, end: 1, word: 'first', probability: 0.9 },
      ]}], words: [{ start: 0, end: 1, word: 'first', probability: 0.9 }] },
      { timeOffsetSeconds: 5.0, segments: [{ id: 0, start: 0, end: 5, text: 'second', words: [
        { start: 0, end: 1, word: 'second', probability: 0.8 },
      ]}], words: [{ start: 0, end: 1, word: 'second', probability: 0.8 }] },
    ];
    const result = mergeSegments(chunks);
    expect(result.segments).toHaveLength(2);
    expect(result.words[1]!.word).toBe('second');
  });

  it('deduplicates overlapping words', () => {
    const chunks = [
      { timeOffsetSeconds: 0.0, segments: [], words: [
        { start: 2, end: 4.8, word: 'world', probability: 0.8 },
      ]},
      { timeOffsetSeconds: 4.5, segments: [], words: [
        { start: 0.1, end: 1, word: 'world', probability: 0.85 },
        { start: 2, end: 4, word: 'goodbye', probability: 0.7 },
      ]},
    ];
    const result = mergeSegments(chunks);
    expect(result.words).toHaveLength(2);
    expect(result.words[0]!.word).toBe('world');
    expect(result.words[0]!.probability).toBe(0.85);
  });

  it('handles empty input', () => {
    expect(mergeSegments([]).segments).toHaveLength(0);
  });
});
