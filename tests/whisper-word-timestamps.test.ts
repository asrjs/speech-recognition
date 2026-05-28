import { describe, expect, it } from 'vitest';
import { buildWhisperWordTimestampsFromTokenDetails } from '../src/models/whisper-seq2seq/word-timestamps.js';
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
});
