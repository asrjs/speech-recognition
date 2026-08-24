import { describe, expect, it } from 'vitest';
import {
  buildWhisperForcedAlignmentTokenIds,
  collectSplitGraphTextTokenRows,
  extractSplitGraphAlignmentRows,
  processSplitGraphAlignment,
  processSplitGraphAlignmentByTimestampSpans,
} from '../src/models/whisper-seq2seq/executor.js';

describe('splitGraph alignment processing', () => {
  it('builds the reference no-timestamps forced-alignment prompt', () => {
    const ids = new Map([
      ['<|startoftranscript|>', 50258],
      ['<|en|>', 50259],
      ['<|transcribe|>', 50360],
      ['<|translate|>', 50359],
      ['<|notimestamps|>', 50363],
      ['<|endoftext|>', 50257],
    ]);
    const tokenizer = { getTokenId: (token: string) => ids.get(token) };

    expect(buildWhisperForcedAlignmentTokenIds(tokenizer, 'en', [11, 12])).toEqual([
      50258, 50259, 50360, 50363, 11, 12, 50257,
    ]);
    expect(buildWhisperForcedAlignmentTokenIds(tokenizer, 'en', [11], 'translate')).toEqual([
      50258, 50259, 50359, 50363, 11, 50257,
    ]);
  });

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

  it('skips timestamp-token rows instead of treating them as text', () => {
    const timestampTokenId = 50_364;
    const tokenIds = [1, 2, 3, 4, timestampTokenId, 11, 12, timestampTokenId];
    const { tokenIds: textIds, rowIndices } = collectSplitGraphTextTokenRows(
      tokenIds,
      4,
      (id) => id !== timestampTokenId && id > 4,
    );
    expect(textIds).toEqual([11, 12]);
    expect(rowIndices).toEqual([5, 6]);

    const data = new Float32Array(tokenIds.length * frameCount);
    data[4 * frameCount + 0] = 1;
    data[5 * frameCount + 1] = 1;
    data[6 * frameCount + 2] = 1;

    const extracted = extractSplitGraphAlignmentRows(data, rowIndices, frameCount);
    const naiveSlice = data.subarray(4 * frameCount, 6 * frameCount);
    expect(Array.from(extracted)).not.toEqual(Array.from(naiveSlice));
    expect(extracted[1]).toBe(1);
    expect(extracted[frameCount + 2]).toBe(1);

    const timestamps = processSplitGraphAlignment({
      alignmentData: data,
      totalTokens: tokenIds.length,
      promptLen: 4,
      textTokenCount: 2,
      frameCount,
      timePrecisionSeconds: 0.02,
      textTokenRowIndices: rowIndices,
    });
    expect(timestamps).toHaveLength(3);
    expect(timestamps[2]!).toBeGreaterThan(timestamps[0]!);
  });

  it('crops padded encoder frames to the audio duration', () => {
    const paddedFrames = 8;
    const data = new Float32Array(totalTokens * paddedFrames);
    data[4 * paddedFrames + 0] = 0.9;
    data[5 * paddedFrames + 1] = 0.9;
    data[6 * paddedFrames + 2] = 0.9;
    const timestamps = processSplitGraphAlignment({
      alignmentData: data,
      totalTokens,
      promptLen,
      textTokenCount,
      frameCount: paddedFrames,
      timePrecisionSeconds: 0.02,
      cropFrameCount: 3,
    });
    expect(timestamps).toHaveLength(textTokenCount + 1);
    expect(timestamps[textTokenCount]!).toBeLessThanOrEqual(0.04);
  });

  it('DTW-aligns each timestamp span only against that span\'s encoder frames', () => {
    const timestampBegin = 50_364;
    const hop = 0.02;
    const ts = (seconds: number) => timestampBegin + Math.round(seconds / hop);
    const promptLen = 4;
    const frameCount = 6;
    const tokenIds = [1, 2, 3, 4, ts(0), 11, 12, ts(0.06), 13, ts(0.12)];
    const data = new Float32Array(tokenIds.length * frameCount);
    data[5 * frameCount + 0] = 1;
    data[6 * frameCount + 1] = 1;
    // Token 13 would steal early frames in a global DTW; the span window is [3, 6).
    data[8 * frameCount + 1] = 1;
    data[8 * frameCount + 4] = 0.2;

    const timestamps = processSplitGraphAlignmentByTimestampSpans({
      alignmentData: data,
      tokenIds,
      promptLen,
      frameCount,
      timePrecisionSeconds: hop,
      medianFilterWidth: 1,
      isTextToken: (id) => id === 11 || id === 12 || id === 13,
      isTimestampToken: (id) => id >= timestampBegin,
      timestampTokenToSeconds: (id) => (id - timestampBegin) * hop,
    });

    expect(timestamps).toBeDefined();
    expect(timestamps).toHaveLength(4);
    expect(timestamps![0]!).toBeGreaterThanOrEqual(0);
    expect(timestamps![2]!).toBeGreaterThanOrEqual(0.06);
    expect(timestamps![2]!).toBeLessThan(0.12);
    expect(timestamps![3]!).toBeCloseTo(0.12, 5);
  });
});
