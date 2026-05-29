/**
 * Tests for segment-merger — merge Whisper chunk results with timestamp adjustment.
 * Phase 7: overlap reconciliation + word deduplication.
 */

import { describe, it, expect } from 'vitest';
import { mergeWhisperSegments } from '../src/models/whisper-seq2seq/segment-merger.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Minimal word shape matching what mergeWhisperSegments expects */
interface TestWord {
  start: number;
  end: number;
  word: string;
  probability: number;
}

interface TestSegment {
  id: number;
  start: number;
  end: number;
  text: string;
  words: TestWord[];
}

interface ChunkResult {
  segments: TestSegment[];
  words: TestWord[];
  timeOffsetSeconds: number;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('mergeWhisperSegments', () => {
  it('adjusts timestamps by time offset', () => {
    const chunks: ChunkResult[] = [
      {
        timeOffsetSeconds: 10.0,
        segments: [
          { id: 0, start: 0.0, end: 5.0, text: 'hello world', words: [
            { start: 0.0, end: 0.5, word: 'hello', probability: 0.9 },
            { start: 0.6, end: 1.2, word: 'world', probability: 0.8 },
          ]},
        ],
        words: [
          { start: 0.0, end: 0.5, word: 'hello', probability: 0.9 },
          { start: 0.6, end: 1.2, word: 'world', probability: 0.8 },
        ],
      },
    ];

    const result = mergeWhisperSegments(chunks);
    expect(result.segments[0]!.start).toBeCloseTo(10.0, 1);
    expect(result.segments[0]!.end).toBeCloseTo(15.0, 1);
    expect(result.words[0]!.start).toBeCloseTo(10.0, 1);
  });

  it('merges multiple chunks in order', () => {
    const chunks: ChunkResult[] = [
      {
        timeOffsetSeconds: 0.0,
        segments: [{ id: 0, start: 0, end: 5, text: 'first', words: [
          { start: 0, end: 0.5, word: 'first', probability: 0.9 },
        ]}],
        words: [{ start: 0, end: 0.5, word: 'first', probability: 0.9 }],
      },
      {
        timeOffsetSeconds: 5.0,
        segments: [{ id: 0, start: 0, end: 5, text: 'second', words: [
          { start: 0, end: 0.5, word: 'second', probability: 0.8 },
        ]}],
        words: [{ start: 0, end: 0.5, word: 'second', probability: 0.8 }],
      },
    ];

    const result = mergeWhisperSegments(chunks);
    expect(result.segments).toHaveLength(2);
    expect(result.segments[0]!.start).toBeCloseTo(0, 1);
    expect(result.segments[1]!.start).toBeCloseTo(5, 1);
    expect(result.words).toHaveLength(2);
    expect(result.words[0]!.word).toBe('first');
    expect(result.words[1]!.word).toBe('second');
  });

  it('reassigns sequential segment IDs', () => {
    const chunks: ChunkResult[] = [
      {
        timeOffsetSeconds: 0.0,
        segments: [
          { id: 0, start: 0, end: 2, text: 'a', words: [] },
          { id: 1, start: 2, end: 5, text: 'b', words: [] },
        ],
        words: [],
      },
    ];

    const result = mergeWhisperSegments(chunks);
    expect(result.segments).toHaveLength(2);
    expect(result.segments[0]!.id).toBe(0);
    expect(result.segments[1]!.id).toBe(1);
  });

  it('handles empty chunks', () => {
    const result = mergeWhisperSegments([]);
    expect(result.segments).toHaveLength(0);
    expect(result.words).toHaveLength(0);
  });

  it('handles chunks with no segments', () => {
    const chunks: ChunkResult[] = [
      { timeOffsetSeconds: 0.0, segments: [], words: [] },
    ];
    const result = mergeWhisperSegments(chunks);
    expect(result.segments).toHaveLength(0);
  });

  it('deduplicates words at overlapping boundaries', () => {
    // Chunk 1 (0-5s): "hello world", world ends at 4.8
    // Chunk 2 (4.5-9.5s): "world goodbye", world starts at 4.6 (in absolute)
    // → 'world' overlaps at 4.6-4.8, deduplicate
    const chunks: ChunkResult[] = [
      {
        timeOffsetSeconds: 0.0,
        segments: [{ id: 0, start: 0, end: 5, text: 'hello world', words: [
          { start: 0, end: 1, word: 'hello', probability: 0.9 },
          { start: 2, end: 4.8, word: 'world', probability: 0.8 },
        ]}],
        words: [
          { start: 0, end: 1, word: 'hello', probability: 0.9 },
          { start: 2, end: 4.8, word: 'world', probability: 0.8 },
        ],
      },
      {
        timeOffsetSeconds: 4.5,
        segments: [{ id: 0, start: 0, end: 5, text: 'world goodbye', words: [
          { start: 0.1, end: 1, word: 'world', probability: 0.85 },
          { start: 2, end: 4.9, word: 'goodbye', probability: 0.7 },
        ]}],
        words: [
          { start: 0.1, end: 1, word: 'world', probability: 0.85 },
          { start: 2, end: 4.9, word: 'goodbye', probability: 0.7 },
        ],
      },
    ];

    const result = mergeWhisperSegments(chunks);
    // 'world' should appear only once (deduplicated)
    expect(result.words).toHaveLength(3);
    expect(result.words[0]!.word).toBe('hello');
    expect(result.words[1]!.word).toBe('world');
    expect(result.words[2]!.word).toBe('goodbye');
    // The kept 'world' should have the higher probability
    expect(result.words[1]!.probability).toBe(0.85);
  });
});
