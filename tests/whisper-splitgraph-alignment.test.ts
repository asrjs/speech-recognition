import { describe, expect, it } from 'vitest';
import { processSplitGraphAlignment } from '../src/models/whisper-seq2seq/executor.js';

describe('splitGraph alignment processing', () => {
  // Simulate a decoder_align output: [T_all=7, S=3] flat matrix
  // 7 total tokens = 4 prompt + 3 text
  const totalTokens = 7;
  const promptLen = 4;
  const textTokenCount = 3;
  const frameCount = 3;

  // Build a simple alignment: each text token attends to one frame
  // Row 0 (prompt): [0.0, 0.0, 0.0]
  // Row 1 (prompt): [0.0, 0.0, 0.0]
  // Row 2 (prompt): [0.0, 0.0, 0.0]
  // Row 3 (prompt): [0.0, 0.0, 0.0]
  // Row 4 (text token 0): [0.9, 0.05, 0.05]
  // Row 5 (text token 1): [0.05, 0.9, 0.05]
  // Row 6 (text token 2): [0.05, 0.05, 0.9]
  function buildAlignment(): Float32Array {
    const total = totalTokens * frameCount;
    const data = new Float32Array(total);
    // text token 0 → frame 0
    data[4 * frameCount + 0] = 0.9;
    data[4 * frameCount + 1] = 0.05;
    data[4 * frameCount + 2] = 0.05;
    // text token 1 → frame 1
    data[5 * frameCount + 0] = 0.05;
    data[5 * frameCount + 1] = 0.9;
    data[5 * frameCount + 2] = 0.05;
    // text token 2 → frame 2
    data[6 * frameCount + 0] = 0.05;
    data[6 * frameCount + 1] = 0.05;
    data[6 * frameCount + 2] = 0.9;
    return data;
  }

  it('slices prompt rows and produces DTW timestamps', () => {
    const alignment = buildAlignment();
    const timestamps = processSplitGraphAlignment({
      alignmentData: alignment,
      totalTokens,
      promptLen,
      textTokenCount,
      frameCount,
      timePrecisionSeconds: 0.02,
    });

    // Should return textTokenCount + 1 timestamps
    expect(timestamps).toHaveLength(textTokenCount + 1);

    // All timestamps should be non-negative and increasing
    for (let i = 1; i < timestamps.length; i++) {
      expect(timestamps[i]!).toBeGreaterThanOrEqual(timestamps[i - 1]!);
      expect(timestamps[i]!).toBeGreaterThanOrEqual(0);
    }

    // Last timestamp should be close to end of audio
    expect(timestamps[textTokenCount]!).toBeGreaterThan(0);
  });

  it('returns zero timestamps when no text tokens', () => {
    const alignment = new Float32Array(4 * 3); // 4 prompt tokens, 0 text
    const timestamps = processSplitGraphAlignment({
      alignmentData: alignment,
      totalTokens: 4,
      promptLen: 4,
      textTokenCount: 0,
      frameCount: 3,
      timePrecisionSeconds: 0.02,
    });
    expect(timestamps).toEqual([0]);
  });

  it('handles alignment with varying attention patterns', () => {
    // Smoother alignment: token 0 → frame 0-1, token 1 → frame 1-2, token 2 → frame 2
    const data = new Float32Array(totalTokens * frameCount);
    // token 4 (text 0): [0.7, 0.3, 0.0]
    data[4 * frameCount + 0] = 0.7;
    data[4 * frameCount + 1] = 0.3;
    // token 5 (text 1): [0.0, 0.6, 0.4]
    data[5 * frameCount + 1] = 0.6;
    data[5 * frameCount + 2] = 0.4;
    // token 6 (text 2): [0.0, 0.0, 1.0]
    data[6 * frameCount + 2] = 1.0;

    const timestamps = processSplitGraphAlignment({
      alignmentData: data,
      totalTokens,
      promptLen,
      textTokenCount,
      frameCount,
      timePrecisionSeconds: 0.02,
    });

    expect(timestamps).toHaveLength(textTokenCount + 1);
    expect(timestamps[0]!).toBeGreaterThanOrEqual(0);
    expect(timestamps[textTokenCount]!).toBeGreaterThan(timestamps[0]!);
  });
});
