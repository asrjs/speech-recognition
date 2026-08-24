import { describe, expect, it } from 'vitest';
import {
  buildWhisperWordTimestampsFromDtwTokens,
  buildWhisperWordTimestampsFromTokenDetails,
  coalesceWhisperWordTimestamps,
  constrainWhisperWordDurations,
  refineWhisperWordsWithForcedAlignment,
  splitWhisperWordsByPause,
  forcedAlignmentLooksAnchored,
} from '../src/models/whisper-seq2seq/word-timestamps.js';
import { mapWhisperNativeToCanonical } from '../src/models/whisper-seq2seq/mapping.js';
import type { WhisperNativeToken, WhisperNativeTranscript } from '../src/models/whisper-seq2seq/types.js';

const TIMESTAMP_BEGIN = 50_364;

function timestamp(seconds: number): number {
  return TIMESTAMP_BEGIN + Math.round(seconds / 0.02);
}

describe('Whisper word timestamp fallback', () => {
  it('interpolates token times between timestamp tokens and collates words', () => {
    const tokens: WhisperNativeToken[] = [
      { index: 0, id: timestamp(0), text: '<|0.00|>' },
      { index: 1, id: 11, text: ' Hello', confidence: 0.8 },
      { index: 2, id: 12, text: ' world', confidence: 0.6 },
      { index: 3, id: 13, text: '!', confidence: 0.4 },
      { index: 4, id: timestamp(1.2), text: '<|1.20|>' },
    ];

    expect(
      buildWhisperWordTimestampsFromTokenDetails(tokens, {
        timestampBegin: TIMESTAMP_BEGIN,
        language: 'en',
      }),
    ).toEqual([
      { index: 0, text: 'Hello', startTime: 0, endTime: 0.4, tokenIds: [11], confidence: 0.8, tokenIndices: [1] },
      {
        index: 1,
        text: 'world!',
        startTime: 0.4,
        endTime: 1.2,
        tokenIds: [12, 13],
        confidence: 0.5,
        tokenIndices: [2, 3],
      },
    ]);
  });

  it('maps native Whisper words to canonical word detail instead of treating segments as words', () => {
    const native: WhisperNativeTranscript = {
      utteranceText: 'Hello world!',
      isFinal: true,
      language: 'en',
      segments: [{ index: 0, text: 'Hello world!', startTime: 0, endTime: 1.2, confidence: 0.6 }],
      words: [
        { index: 0, text: 'Hello', startTime: 0, endTime: 0.4, confidence: 0.8, tokenIndices: [1] },
        { index: 1, text: 'world!', startTime: 0.4, endTime: 1.2, confidence: 0.5, tokenIndices: [2, 3] },
      ],
    };

    const canonical = mapWhisperNativeToCanonical(
      native,
      { ecosystem: 'openai', family: 'whisper-seq2seq', task: 'transcribe' },
      { detailLevel: 'words' },
    );

    expect(canonical.words).toEqual(native.words);
    expect(canonical.meta.wordCount).toBe(2);
    expect(canonical.segments).toHaveLength(1);
  });

  it('falls back to timestamp-token interpolation when alignment is empty', () => {
    const tokens: WhisperNativeToken[] = [
      { index: 0, id: timestamp(0), text: '<|0.00|>' },
      { index: 1, id: 11, text: ' Hello' },
      { index: 2, id: 12, text: ' world' },
      { index: 3, id: timestamp(1), text: '<|1.00|>' },
    ];

    expect(
      coalesceWhisperWordTimestamps([], tokens, {
        timestampBegin: TIMESTAMP_BEGIN,
        language: 'en',
      }),
    ).toEqual([
      { index: 0, text: 'Hello', startTime: 0, endTime: 0.5, tokenIds: [11], tokenIndices: [1] },
      { index: 1, text: 'world', startTime: 0.5, endTime: 1, tokenIds: [12], tokenIndices: [2] },
    ]);
  });

  it('collates DTW token times into words without re-encoding the transcript', () => {
    expect(
      buildWhisperWordTimestampsFromDtwTokens(
        [
          { id: 11, text: ' Hello', sourceIndex: 1 },
          { id: 12, text: ' world', sourceIndex: 2 },
          { id: 13, text: '!', sourceIndex: 3 },
        ],
        [0.1, 0.4, 0.9, 1.2],
        { language: 'en' },
      ),
    ).toEqual([
      { index: 0, text: 'Hello', startTime: 0.1, endTime: 0.4, tokenIds: [11], tokenIndices: [1] },
      { index: 1, text: 'world!', startTime: 0.4, endTime: 1.2, tokenIds: [12, 13], tokenIndices: [2, 3] },
    ]);
  });
});

describe('Whisper word duration constraints', () => {
  it('clips DTW outlier words longer than twice the median duration', () => {
    const words = constrainWhisperWordDurations([
      { index: 0, text: 'In', startTime: 0, endTime: 0.16 },
      { index: 1, text: 'the', startTime: 0.16, endTime: 0.32 },
      { index: 2, text: 'long', startTime: 0.32, endTime: 2.2 },
      { index: 3, text: 'history', startTime: 2.2, endTime: 4.0 },
      { index: 4, text: 'of', startTime: 4.0, endTime: 4.16 },
    ]);

    expect(words[2]!.endTime - words[2]!.startTime).toBeLessThanOrEqual(0.4);
    expect(words[3]!.endTime - words[3]!.startTime).toBeLessThanOrEqual(0.4);
    expect(words[3]!.startTime).toBeGreaterThanOrEqual(words[2]!.endTime);
    expect(words[3]!.startTime).toBeLessThan(0.8);
    expect(words[4]!.startTime).toBeLessThan(1.2);
  });

  it('does not pull the next phrase across a comma pause', () => {
    const words = constrainWhisperWordDurations([
      { index: 0, text: 'world,', startTime: 0.5, endTime: 0.9 },
      { index: 1, text: 'only', startTime: 4.2, endTime: 4.5 },
    ]);
    expect(words[1]!.startTime).toBe(4.2);
  });

  it('closes turbo DTW holes after a clipped word but keeps the comma pause', () => {
    const words = constrainWhisperWordDurations([
      { index: 0, text: 'In', startTime: 0, endTime: 0.16 },
      { index: 1, text: 'the', startTime: 0.16, endTime: 0.32 },
      { index: 2, text: 'long', startTime: 0.5, endTime: 1.06 },
      { index: 3, text: 'history', startTime: 2.35, endTime: 2.91 },
      { index: 4, text: 'of', startTime: 4.2, endTime: 4.22 },
      { index: 5, text: 'the', startTime: 4.22, endTime: 4.4 },
      { index: 6, text: 'world,', startTime: 4.4, endTime: 4.8 },
      { index: 7, text: 'only', startTime: 5.5, endTime: 5.8 },
    ]);
    expect(words[3]!.startTime).toBeLessThan(1.5);
    expect(words[4]!.startTime).toBeLessThan(2);
    expect(words[4]!.endTime - words[4]!.startTime).toBeGreaterThanOrEqual(0.08);
    expect(words[6]!.text).toBe('world,');
    expect(words[7]!.startTime).toBe(5.5);
  });

  it('grows a 20ms DTW stub by borrowing from the previous word', () => {
    const words = constrainWhisperWordDurations([
      { index: 0, text: 'history', startTime: 0.4, endTime: 1.0 },
      { index: 1, text: 'of', startTime: 1.02, endTime: 1.04 },
      { index: 2, text: 'the', startTime: 1.06, endTime: 1.2 },
    ]);
    expect(words[1]!.endTime - words[1]!.startTime).toBeGreaterThanOrEqual(0.08);
    expect(words[1]!.startTime).toBeGreaterThanOrEqual(words[0]!.endTime);
    expect(words[2]!.startTime).toBeGreaterThanOrEqual(words[1]!.endTime);
  });

  it('closes a half-second DTW hole even when the median word is already long', () => {
    const words = constrainWhisperWordDurations([
      { index: 0, text: 'granted', startTime: 8.6, endTime: 9.0 },
      { index: 1, text: 'the', startTime: 9.02, endTime: 9.16 },
      { index: 2, text: 'role', startTime: 9.7, endTime: 9.98 },
    ]);
    expect(words[2]!.startTime).toBeLessThan(9.3);
    expect(words[2]!.startTime).toBeGreaterThanOrEqual(words[1]!.endTime);
  });
});

describe('Whisper forced-alignment refine', () => {
  it('zips equal-length wav2vec2 times onto Whisper words', () => {
    expect(
      refineWhisperWordsWithForcedAlignment(
        [
          { index: 0, text: 'Hello', startTime: 0, endTime: 1.8, tokenIds: [11] },
          { index: 1, text: 'world', startTime: 1.8, endTime: 4.2, tokenIds: [12] },
        ],
        [
          { text: 'Hello', startTime: 0.12, endTime: 0.4, confidence: 0.9 },
          { text: 'world', startTime: 0.45, endTime: 0.82, confidence: 0.8 },
        ],
      ),
    ).toEqual([
      { index: 0, text: 'Hello', startTime: 0.12, endTime: 0.4, tokenIds: [11], confidence: 0.9 },
      { index: 1, text: 'world', startTime: 0.45, endTime: 0.82, tokenIds: [12], confidence: 0.8 },
    ]);
  });

  it('matches sequential normalized text when counts differ', () => {
    const words = refineWhisperWordsWithForcedAlignment(
      [
        { index: 0, text: 'Hello', startTime: 0, endTime: 1 },
        { index: 1, text: ',', startTime: 1, endTime: 1.1 },
        { index: 2, text: 'world', startTime: 1.1, endTime: 3 },
      ],
      [
        { text: 'hello', startTime: 0.1, endTime: 0.3 },
        { text: 'world', startTime: 0.4, endTime: 0.7 },
      ],
    );

    expect(words[0]!.text).toBe('Hello');
    expect(words[0]!.startTime).toBeCloseTo(0.1, 1);
    expect(words[1]!.text).toBe(',');
    expect(words[2]!.text).toBe('world');
    expect(words[2]!.startTime).toBeGreaterThanOrEqual(words[0]!.endTime);
  });

  it('closes a mid-phrase wav2vec2 hole that is not a punctuation pause', () => {
    const words = refineWhisperWordsWithForcedAlignment(
      [
        { index: 0, text: 'the', startTime: 0.16, endTime: 0.5 },
        { index: 1, text: 'long', startTime: 0.5, endTime: 1.06 },
        { index: 2, text: 'history', startTime: 1.08, endTime: 1.64 },
      ],
      [
        { text: 'the', startTime: 0.22, endTime: 0.42 },
        { text: 'long', startTime: 1.65, endTime: 1.73 },
        { text: 'history', startTime: 1.96, endTime: 2.08 },
      ],
    );

    expect(words[1]!.text).toBe('long');
    expect(words[1]!.startTime).toBeLessThan(0.6);
    expect(words[1]!.startTime).toBeGreaterThanOrEqual(words[0]!.endTime);
    expect(words[2]!.startTime).toBeGreaterThanOrEqual(words[1]!.endTime);
  });

  it('overlays wav2vec2 times even when they start later than DTW', () => {
    const words = refineWhisperWordsWithForcedAlignment(
      [
        { index: 0, text: 'generations', startTime: 6.94, endTime: 7.5 },
        { index: 1, text: 'have', startTime: 7.52, endTime: 8.3 },
        { index: 2, text: 'been', startTime: 8.32, endTime: 8.58 },
      ],
      [
        { text: 'generations', startTime: 6.9, endTime: 7.57 },
        { text: 'have', startTime: 8.7, endTime: 8.86 },
        { text: 'been', startTime: 8.88, endTime: 9.08 },
      ],
    );

    expect(words[1]!.text).toBe('have');
    expect(words[1]!.startTime).toBeCloseTo(8.7, 1);
    expect(words[1]!.startTime).toBeGreaterThanOrEqual(words[0]!.endTime);
    expect(words[2]!.startTime).toBeGreaterThanOrEqual(words[1]!.endTime);
  });

  it('overlays a short function word from wav2vec2 even when CTC starts later than DTW', () => {
    const words = refineWhisperWordsWithForcedAlignment(
      [
        { index: 0, text: 'history', startTime: 1.08, endTime: 1.6 },
        { index: 1, text: 'of', startTime: 1.6, endTime: 2.2 },
        { index: 2, text: 'the', startTime: 2.22, endTime: 2.3 },
      ],
      [
        { text: 'history', startTime: 1.08, endTime: 1.58 },
        { text: 'of', startTime: 1.07, endTime: 1.15 },
        { text: 'the', startTime: 2.22, endTime: 2.3 },
      ],
    );

    expect(words[1]!.text).toBe('of');
    expect(words[1]!.startTime).toBeGreaterThanOrEqual(words[0]!.endTime);
    expect(words[1]!.endTime - words[1]!.startTime).toBeLessThanOrEqual(0.25);
    expect(words[2]!.startTime).toBeGreaterThan(words[1]!.endTime);
  });

  it('overlays a globally shifted CTC sequence instead of keeping DTW', () => {
    const words = refineWhisperWordsWithForcedAlignment(
      [
        { index: 0, text: 'In', startTime: 0.06, endTime: 0.242 },
        { index: 1, text: 'the', startTime: 0.242, endTime: 0.423 },
        { index: 2, text: 'long', startTime: 0.5, endTime: 1.06 },
      ],
      [
        { text: 'In', startTime: 0.0, endTime: 0.04 },
        { text: 'the', startTime: 0.04, endTime: 0.12 },
        { text: 'long', startTime: 0.5, endTime: 0.9 },
      ],
    );

    expect(words[0]!.text).toBe('In');
    expect(words[0]!.startTime).toBeLessThan(0.05);
    expect(words[1]!.endTime - words[1]!.startTime).toBeLessThanOrEqual(0.2);
  });

  it('clips unmatched short words that wav2vec2 did not overlay', () => {
    const words = refineWhisperWordsWithForcedAlignment(
      [
        { index: 0, text: 'history', startTime: 1.08, endTime: 1.6 },
        { index: 1, text: 'of', startTime: 1.6, endTime: 2.2 },
        { index: 2, text: 'the', startTime: 2.22, endTime: 2.3 },
      ],
      [
        { text: 'history', startTime: 1.08, endTime: 1.58 },
        { text: 'the', startTime: 2.22, endTime: 2.3 },
      ],
    );

    expect(words[1]!.text).toBe('of');
    expect(words[1]!.endTime - words[1]!.startTime).toBeLessThanOrEqual(0.25);
  });

  it('still clips short DTW words when wav2vec2 returns no alignment', () => {
    const words = refineWhisperWordsWithForcedAlignment(
      [
        { index: 0, text: 'history', startTime: 1.08, endTime: 1.6 },
        { index: 1, text: 'of', startTime: 1.6, endTime: 2.2 },
        { index: 2, text: 'the', startTime: 2.22, endTime: 2.3 },
      ],
      [],
    );

    expect(words[1]!.text).toBe('of');
    expect(words[1]!.endTime - words[1]!.startTime).toBeLessThanOrEqual(0.25);
  });

  it('grows a zero-duration aligned word after wav2vec2 overlay', () => {
    const words = refineWhisperWordsWithForcedAlignment(
      [
        { index: 0, text: 'and', startTime: 0.4, endTime: 0.62, tokenIds: [11] },
        { index: 1, text: 'a', startTime: 0.62, endTime: 0.64, tokenIds: [12] },
        { index: 2, text: 'new', startTime: 0.64, endTime: 0.9, tokenIds: [13] },
      ],
      [
        { text: 'and', startTime: 0.4, endTime: 0.6 },
        { text: 'a', startTime: 0.6, endTime: 0.6 },
        { text: 'new', startTime: 0.62, endTime: 0.9 },
      ],
    );

    expect(words[1]!.text).toBe('a');
    expect(words[1]!.endTime - words[1]!.startTime).toBeGreaterThanOrEqual(0.07);
    expect(words[1]!.startTime).toBeGreaterThanOrEqual(words[0]!.endTime);
    expect(words[2]!.startTime).toBeGreaterThanOrEqual(words[1]!.endTime);
  });

  it('splits words across a long pause so wav2vec2 can refine each phrase', () => {
    const groups = splitWhisperWordsByPause([
      { index: 0, text: 'world,', startTime: 1.92, endTime: 2.48 },
      { index: 1, text: 'only', startTime: 6.16, endTime: 6.38 },
      { index: 2, text: 'a', startTime: 6.38, endTime: 6.5 },
    ]);
    expect(groups).toHaveLength(2);
    expect(groups[0]!.map((word) => word.text)).toEqual(['world,']);
    expect(groups[1]!.map((word) => word.text)).toEqual(['only', 'a']);
  });

  it('treats CTC as unanchored when the first word is more than 0.8s later than Whisper', () => {
    expect(
      forcedAlignmentLooksAnchored(
        [{ startTime: 0.06 }],
        [{ startTime: 2.79 }],
      ),
    ).toBe(false);
    expect(
      forcedAlignmentLooksAnchored(
        [{ startTime: 0.06 }],
        [{ startTime: 0.0 }],
      ),
    ).toBe(true);
  });
});
