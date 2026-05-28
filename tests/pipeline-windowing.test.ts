import {
  createDefaultModelInferenceLimits,
  dedupeWindowWords,
  partitionWordsIntoSegments,
  resolveWindowPolicy,
  transcribeWithWindowing,
  type TranscriptResult,
} from '@asrjs/speech-recognition';
import { describe, expect, it } from 'vitest';

function word(index: number, text: string, startTime: number, endTime: number) {
  return { index, text, startTime, endTime };
}

function result(words: ReturnType<typeof word>[], offsetText = ''): TranscriptResult {
  return {
    text: offsetText || words.map((item) => item.text).join(' '),
    warnings: [],
    meta: {
      detailLevel: 'words',
      isFinal: true,
      metrics: {
        totalMs: 10,
        audioDurationSec: 1,
      },
    },
    words,
  };
}

describe('pipeline windowing primitives', () => {
  it('resolves model-specific Parakeet and Whisper defaults', () => {
    const parakeet = createDefaultModelInferenceLimits({ family: 'nemo-tdt', modelId: 'parakeet' });
    const whisper = createDefaultModelInferenceLimits({ family: 'whisper-seq2seq', modelId: 'whisper' });

    expect(resolveWindowPolicy({ inference: parakeet }).windowDurationSec).toBe(90);
    expect(resolveWindowPolicy({ inference: parakeet }).maxWindowDurationSec).toBe(180);
    expect(resolveWindowPolicy({ inference: whisper }).windowDurationSec).toBe(30);
    expect(resolveWindowPolicy({ inference: whisper, windowDurationSeconds: 90 }).windowDurationSec).toBe(30);
    expect(
      resolveWindowPolicy({
        inference: whisper,
        windowDurationSeconds: 90,
        unsafeAllowOverMaxWindow: true,
      }).windowDurationSec,
    ).toBe(90);
  });

  it('segments timestamped words conservatively', () => {
    const segments = partitionWordsIntoSegments([
      word(0, 'Dr.', 0, 0.2),
      word(1, 'Smith', 0.3, 0.8),
      word(2, 'arrived.', 0.9, 1.4),
      word(3, 'He', 1.5, 1.7),
      word(4, 'left?', 1.8, 2.1),
      word(5, 'Yes', 5.5, 5.8),
    ]);

    expect(segments.map((segment) => segment.text)).toEqual([
      'Dr. Smith arrived.',
      'He left?',
      'Yes',
    ]);
  });

  it('dedupes overlapping words by normalized text', () => {
    expect(
      dedupeWindowWords([
        word(0, 'hello', 0, 0.5),
        word(1, 'Hello,', 0.2, 1.0),
        word(2, 'world', 1.1, 1.5),
      ]).map((item) => item.text),
    ).toEqual(['Hello,', 'world']);
  });

  it('routes long audio through window transcription and merges words', async () => {
    const audio = new Float32Array(4 * 16000);
    let calls = 0;
    const transcript = await transcribeWithWindowing({
      input: audio,
      inference: {
        sampleRate: 16000,
        maxInputDurationSec: 2,
        recommendedWindowDurationSec: 2,
        minWindowDurationSec: 1,
        maxWindowDurationSec: 2,
        autoWindowThresholdSec: 2,
        defaultOverlapSec: 0.5,
        supportsWordTimestamps: true,
        supportsSegmentTimestamps: true,
        defaultSegmentationStrategy: 'word-punctuation',
        defaultMergeStrategy: 'word-dedupe',
      },
      options: { detail: 'words' },
      async transcribeWindow(_windowAudio) {
        calls += 1;
        if (calls === 1) {
          return result([word(0, 'Hello', 0, 0.4), word(1, 'world.', 0.5, 1.0)]);
        }
        return result([word(0, 'Again', 0, 0.4), word(1, 'done.', 0.5, 1.0)]);
      },
    });

    expect(calls).toBeGreaterThan(1);
    expect(transcript.text).toContain('Hello world.');
    expect(transcript.words?.length).toBeGreaterThan(2);
    expect(transcript.meta.metrics?.rtf).toBeGreaterThan(0);
  });
});
