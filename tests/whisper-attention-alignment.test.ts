import { describe, expect, it } from 'vitest';
import {
  computeWhisperDtwTokenTimestamps,
  medianFilterWhisperAttention,
} from '../src/models/whisper-seq2seq/index.js';

describe('Whisper attention DTW alignment helpers', () => {
  it('median filters attention over the audio-frame axis with reflected edges', () => {
    const filtered = medianFilterWhisperAttention(
      Float32Array.from([
        0, 10, 0, 8, 0,
        1, 1, 9, 1, 1,
      ]),
      { tokenCount: 2, frameCount: 5, width: 3 },
    );

    expect(Array.from(filtered)).toEqual([
      10, 0, 8, 0, 8,
      1, 1, 1, 1, 1,
    ]);
  });

  it('computes monotonic token timestamps from selected cross-attention heads', () => {
    const timestamps = computeWhisperDtwTokenTimestamps({
      // one selected alignment head, 3 decoder text tokens, 6 encoder frames
      attentionHeads: [
        {
          values: Float32Array.from([
            9, 8, 2, 1, 0, 0,
            0, 2, 9, 8, 2, 0,
            0, 0, 1, 2, 8, 9,
          ]),
          tokenCount: 3,
          frameCount: 6,
        },
      ],
      tokenCount: 3,
      frameCount: 6,
      medianFilterWidth: 1,
      timePrecisionSeconds: 0.02,
    });

    expect(timestamps).toHaveLength(4);
    expect(timestamps[0]).toBe(0);
    expect(timestamps[1]).toBeGreaterThanOrEqual(0.02);
    expect(timestamps[2]).toBeGreaterThanOrEqual(timestamps[1]!);
    expect(timestamps[3]).toBeCloseTo(0.1, 5);
  });
});
