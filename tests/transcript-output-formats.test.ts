import {
  formatSubtitleTimestamp,
  partitionWordsIntoSentences,
  transcriptToSrt,
  transcriptToVtt,
  type TranscriptResult,
  type TranscriptWord,
} from '@asrjs/speech-recognition';
import { describe, expect, it } from 'vitest';

function word(index: number, text: string, startTime: number, endTime: number): TranscriptWord {
  return { index, text, startTime, endTime };
}

function transcript(words: TranscriptWord[]): TranscriptResult {
  const sentences = partitionWordsIntoSentences(words);
  return {
    text: words.map((item) => item.text).join(' '),
    warnings: [],
    meta: {
      detailLevel: 'sentences+words',
      isFinal: true,
      wordCount: words.length,
      sentenceCount: sentences.length,
      segmentCount: sentences.length,
      language: 'en',
    },
    sentences,
    words,
  };
}

describe('transcript output formats', () => {
  it('formats SRT and VTT timestamps with millisecond precision', () => {
    expect(formatSubtitleTimestamp(0, 'srt')).toBe('00:00:00,000');
    expect(formatSubtitleTimestamp(65.4321, 'srt')).toBe('00:01:05,432');
    expect(formatSubtitleTimestamp(65.4321, 'vtt')).toBe('00:01:05.432');
    expect(formatSubtitleTimestamp(3661.9999, 'vtt')).toBe('01:01:02.000');
  });

  it('exposes first-class sentence spans with word indices', () => {
    const sentences = partitionWordsIntoSentences([
      word(0, 'The', 0, 0.1),
      word(1, 'boy', 0.1, 0.3),
      word(2, 'rose.', 0.3, 0.7),
      word(3, 'A', 1.2, 1.3),
      word(4, 'rod', 1.3, 1.6),
    ]);

    expect(sentences).toMatchObject([
      { index: 0, text: 'The boy rose.', startTime: 0, endTime: 0.7, wordIndices: [0, 1, 2] },
      { index: 1, text: 'A rod', startTime: 1.2, endTime: 1.6, wordIndices: [3, 4] },
    ]);
  });

  it('writes SRT from sentence timestamps and escapes cue arrows', () => {
    const result: TranscriptResult = {
      text: 'hello --> world',
      warnings: [],
      meta: { detailLevel: 'sentences', isFinal: true, sentenceCount: 1 },
      sentences: [
        { index: 0, text: 'hello --> world', startTime: 0, endTime: 1.25, speaker: 'SPEAKER_00' },
      ],
    };

    expect(transcriptToSrt(result)).toBe([
      '1',
      '00:00:00,000 --> 00:00:01,250',
      '[SPEAKER_00]: hello -> world',
      '',
    ].join('\n'));
  });

  it('writes VTT cues from words when requested', () => {
    const result = transcript([
      word(0, 'Hello', 0, 0.4),
      word(1, 'world.', 0.45, 0.9),
      word(2, 'Again', 4.2, 4.6),
      word(3, 'now.', 4.7, 5.0),
    ]);

    expect(transcriptToVtt(result, { source: 'words', maxGapSeconds: 3 })).toBe([
      'WEBVTT',
      '',
      '00:00:00.000 --> 00:00:00.900',
      'Hello world.',
      '',
      '00:00:04.200 --> 00:00:05.000',
      'Again now.',
      '',
    ].join('\n'));
  });
});
