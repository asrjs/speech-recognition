import {
  createPipelineContext,
  createSentenceSegmentationStage,
  runPipelineStages,
  type TranscriptResult,
  type TranscriptWord,
} from '@asrjs/speech-recognition';
import { describe, expect, it } from 'vitest';

function word(index: number, text: string, startTime: number, endTime: number): TranscriptWord {
  return { index, text, startTime, endTime };
}

function transcript(words: readonly TranscriptWord[]): TranscriptResult {
  return {
    text: words.map((item) => item.text).join(' '),
    warnings: [],
    meta: {
      detailLevel: 'words',
      isFinal: true,
      wordCount: words.length,
      segmentCount: 1,
    },
    segments: [
      {
        index: 0,
        text: words.map((item) => item.text).join(' '),
        startTime: words[0]?.startTime ?? 0,
        endTime: words.at(-1)?.endTime ?? 0,
        wordIndices: words.map((item) => item.index),
      },
    ],
    words,
  };
}

describe('sentence segmentation pipeline stage', () => {
  it('derives sentence spans from transcript words and updates metadata', async () => {
    const words = [
      word(0, 'Hello', 0, 0.2),
      word(1, 'world.', 0.25, 0.6),
      word(2, 'Again', 0.8, 1.0),
      word(3, 'now.', 1.05, 1.4),
    ];

    const result = await runPipelineStages(
      createPipelineContext({ transcript: transcript(words) }),
      [createSentenceSegmentationStage()],
    );

    expect(result.transcript?.words).toBe(words);
    expect(result.transcript?.tokens).toBeUndefined();
    expect(result.transcript?.text).toBe('Hello world. Again now.');
    expect(result.transcript?.sentences).toMatchObject([
      { index: 0, text: 'Hello world.', startTime: 0, endTime: 0.6, wordIndices: [0, 1] },
      { index: 1, text: 'Again now.', startTime: 0.8, endTime: 1.4, wordIndices: [2, 3] },
    ]);
    expect(result.transcript?.meta).toMatchObject({
      detailLevel: 'words',
      isFinal: true,
      wordCount: 4,
      sentenceCount: 2,
      segmentCount: 1,
    });
  });

  it('can replace legacy segments with sentence spans', async () => {
    const words = [
      word(0, 'First', 0, 0.2),
      word(1, 'line.', 0.25, 0.5),
      word(2, 'Second', 4.2, 4.6),
      word(3, 'line', 4.7, 4.9),
    ];

    const result = await runPipelineStages(
      createPipelineContext({ transcript: transcript(words) }),
      [createSentenceSegmentationStage({ updateSegments: true })],
    );

    expect(result.transcript?.segments).toMatchObject([
      { index: 0, text: 'First line.', startTime: 0, endTime: 0.5, wordIndices: [0, 1] },
      { index: 1, text: 'Second line', startTime: 4.2, endTime: 4.9, wordIndices: [2, 3] },
    ]);
    expect(result.transcript?.meta.segmentCount).toBe(2);
    expect(result.transcript?.meta.sentenceCount).toBe(2);
  });

  it('leaves transcripts without words unchanged', async () => {
    const original: TranscriptResult = {
      text: 'plain text only',
      warnings: [],
      meta: { detailLevel: 'text', isFinal: true },
    };

    const result = await runPipelineStages(
      createPipelineContext({ transcript: original }),
      [createSentenceSegmentationStage()],
    );

    expect(result.transcript).toBe(original);
  });
});
