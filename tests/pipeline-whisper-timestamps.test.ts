import {
  collateWhisperWordTimestamps,
  decodeWhisperTimestampSpans,
  isWhisperTimestampToken,
  mergeWhisperTokenSequences,
  whisperTimestampTokenToSeconds,
  type WhisperTokenTimestamp,
} from '@asrjs/speech-recognition';
import { describe, expect, it } from 'vitest';

const TIMESTAMP_BEGIN = 50_364;

function timestamp(seconds: number): number {
  return TIMESTAMP_BEGIN + Math.round(seconds / 0.02);
}

describe('Whisper timestamp token helpers', () => {
  it('detects timestamp tokens and converts them to seconds', () => {
    expect(isWhisperTimestampToken(TIMESTAMP_BEGIN - 1, { timestampBegin: TIMESTAMP_BEGIN })).toBe(false);
    expect(isWhisperTimestampToken(TIMESTAMP_BEGIN, { timestampBegin: TIMESTAMP_BEGIN })).toBe(true);
    expect(isWhisperTimestampToken(timestamp(30.02), { timestampBegin: TIMESTAMP_BEGIN })).toBe(false);
    expect(whisperTimestampTokenToSeconds(timestamp(1.24), { timestampBegin: TIMESTAMP_BEGIN })).toBe(1.24);
  });

  it('decodes paired timestamp tokens into text spans without model-specific inference', () => {
    const spans = decodeWhisperTimestampSpans(
      [timestamp(0), 11, 12, timestamp(1.5), timestamp(1.5), 13, timestamp(2.2)],
      {
        timestampBegin: TIMESTAMP_BEGIN,
        decodeTokens: (tokens) => tokens.map((token) => ({ 11: ' Hello', 12: ' world', 13: ' again' })[token] ?? '').join(''),
      },
    );

    expect(spans).toEqual([
      { index: 0, startTime: 0, endTime: 1.5, tokenIds: [11, 12], text: ' Hello world' },
      { index: 1, startTime: 1.5, endTime: 2.2, tokenIds: [13], text: ' again' },
    ]);
  });

  it('merges overlapping chunk token sequences with longest-common-sequence tolerance', () => {
    const [tokens, tokenTimestamps] = mergeWhisperTokenSequences(
      [
        [1, 2, 3, 4],
        [3, 4, 5, 6],
        [6, 7],
      ],
      [
        [0, 0.2, 0.4, 0.6],
        [0.42, 0.62, 0.8, 1.0],
        [1.02, 1.2],
      ],
    );

    expect(tokens).toEqual([1, 2, 3, 4, 5, 6, 7]);
    expect(tokenTimestamps).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]);
  });

  it('collates word timestamps while attaching punctuation to neighboring words', () => {
    const tokenTimestamps: WhisperTokenTimestamp[] = [
      { tokenId: 1, text: ' Hello', startTime: 0, endTime: 0.2 },
      { tokenId: 2, text: ',', startTime: 0.2, endTime: 0.22 },
      { tokenId: 3, text: ' world', startTime: 0.25, endTime: 0.5 },
      { tokenId: 4, text: '!', startTime: 0.5, endTime: 0.52 },
      { tokenId: 5, text: ' "', startTime: 0.6, endTime: 0.62 },
      { tokenId: 6, text: 'again', startTime: 0.62, endTime: 0.9 },
    ];

    expect(collateWhisperWordTimestamps(tokenTimestamps)).toEqual([
      { index: 0, text: 'Hello,', startTime: 0, endTime: 0.22, tokenIds: [1, 2] },
      { index: 1, text: 'world!', startTime: 0.25, endTime: 0.52, tokenIds: [3, 4] },
      { index: 2, text: '"again', startTime: 0.6, endTime: 0.9, tokenIds: [5, 6] },
    ]);
  });

  it('collates CJK-like token timestamps as unicode units instead of whitespace words', () => {
    const tokenTimestamps: WhisperTokenTimestamp[] = [
      { tokenId: 1, text: '你', startTime: 0, endTime: 0.2 },
      { tokenId: 2, text: '好', startTime: 0.2, endTime: 0.4 },
      { tokenId: 3, text: '。', startTime: 0.4, endTime: 0.42 },
    ];

    expect(collateWhisperWordTimestamps(tokenTimestamps, { language: 'chinese' })).toEqual([
      { index: 0, text: '你', startTime: 0, endTime: 0.2, tokenIds: [1] },
      { index: 1, text: '好。', startTime: 0.2, endTime: 0.42, tokenIds: [2, 3] },
    ]);
  });
});
