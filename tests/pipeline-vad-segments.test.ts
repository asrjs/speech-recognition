import {
  mergeNearbySpeechSegments,
  padSpeechSegments,
  speechSegmentsToWindows,
  splitLongSpeechSegments,
  type SpeechSegment,
} from '@asrjs/speech-recognition';
import { describe, expect, it } from 'vitest';

function segment(index: number, startTime: number, endTime: number, confidence?: number): SpeechSegment {
  return {
    index,
    startTime,
    endTime,
    ...(confidence === undefined ? {} : { confidence }),
  };
}

describe('VAD speech segment helpers', () => {
  it('pads speech segments and clamps them to audio bounds', () => {
    const result = padSpeechSegments(
      [segment(8, 0.01, 0.2, 0.7), segment(9, 0.8, 0.99, 0.9)],
      { padSeconds: 0.03, minTime: 0, maxTime: 1 },
    );

    expect(result).toEqual([
      { index: 0, startTime: 0, endTime: 0.23, confidence: 0.7 },
      { index: 1, startTime: 0.77, endTime: 1, confidence: 0.9 },
    ]);
  });

  it('merges overlapping or nearby speech segments and preserves separated silence', () => {
    const result = mergeNearbySpeechSegments(
      [segment(0, 0, 0.4, 0.6), segment(1, 0.45, 0.9, 0.8), segment(2, 1.2, 1.5, 0.5)],
      { minSilenceSeconds: 0.1 },
    );

    expect(result).toEqual([
      { index: 0, startTime: 0, endTime: 0.9, confidence: 0.8 },
      { index: 1, startTime: 1.2, endTime: 1.5, confidence: 0.5 },
    ]);
  });

  it('splits long speech segments into catalog-safe windows', () => {
    const result = splitLongSpeechSegments([segment(0, 0, 75, 0.7)], {
      maxDurationSeconds: 30,
      minDurationSeconds: 0.25,
    });

    expect(result).toEqual([
      { index: 0, startTime: 0, endTime: 30, confidence: 0.7 },
      { index: 1, startTime: 30, endTime: 60, confidence: 0.7 },
      { index: 2, startTime: 60, endTime: 75, confidence: 0.7 },
    ]);
  });

  it('plans VAD windows by filtering short segments, padding, merging, and splitting', () => {
    const result = speechSegmentsToWindows(
      [segment(0, 0.1, 0.2), segment(1, 0.5, 1.0, 0.6), segment(2, 1.06, 1.4, 0.9)],
      {
        audioDurationSeconds: 2,
        minSpeechSeconds: 0.25,
        minSilenceSeconds: 0.1,
        padSeconds: 0.03,
        maxWindowSeconds: 0.5,
      },
    );

    expect(result).toEqual([
      { index: 0, startTime: 0.47, endTime: 0.97, confidence: 0.9, sourceSegmentIndices: [1, 2] },
      { index: 1, startTime: 0.97, endTime: 1.43, confidence: 0.9, sourceSegmentIndices: [1, 2] },
    ]);
  });
});
