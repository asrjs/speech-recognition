import { describe, expect, it } from 'vitest';
import { crossAttentionDtwTimestamps } from '../src/alignment/cross-attention-dtw.js';

describe('crossAttentionDtwTimestamps (standalone DTW alignment)', () => {
  it('returns [0] for empty tokens', () => {
    const result = crossAttentionDtwTimestamps([], [], 10);
    expect(result).toEqual([0]);
  });

  it('returns zeros when numFrames is 0', () => {
    const result = crossAttentionDtwTimestamps(
      [new Float32Array([1, 2, 3])],
      [1, 2, 3],
      0,
    );
    expect(result).toHaveLength(4);
    for (const t of result) {
      expect(t).toBe(0);
    }
  });

  it('computes monotonic timestamps from a single attention head', () => {
    // 3 tokens, 6 frames — clear diagonal attention pattern
    const matrix = new Float32Array([
      0.9, 0.8, 0.1, 0.0, 0.0, 0.0,
      0.0, 0.1, 0.9, 0.8, 0.1, 0.0,
      0.0, 0.0, 0.1, 0.1, 0.8, 0.9,
    ]);
    const tokens = [1, 2, 3];
    const timestamps = crossAttentionDtwTimestamps([matrix], tokens, 6, 0.02);

    expect(timestamps).toHaveLength(4);
    // timestamps should be monotonic non-decreasing
    for (let i = 1; i < timestamps.length; i++) {
      expect(timestamps[i]).toBeGreaterThanOrEqual(timestamps[i - 1]!);
    }
    // end timestamp = (numFrames - 1) * frameDuration
    expect(timestamps[3]).toBeCloseTo(0.1, 5);
  });

  it('averages multiple attention heads before DTW', () => {
    // 2 tokens, 3 frames — two heads with different but complementary patterns
    const headA = new Float32Array([
      0.8, 0.5, 0.1,
      0.1, 0.5, 0.8,
    ]);
    const headB = new Float32Array([
      0.7, 0.6, 0.2,
      0.2, 0.6, 0.7,
    ]);
    const tokens = [10, 20];
    const timestamps = crossAttentionDtwTimestamps([headA, headB], tokens, 3, 0.02);

    expect(timestamps).toHaveLength(3);
    expect(timestamps[0]).toBeGreaterThanOrEqual(0);
    expect(timestamps[1]).toBeGreaterThanOrEqual(timestamps[0]!);
    // end timestamp
    expect(timestamps[2]).toBeCloseTo(0.04, 5);
  });

  it('returns monotonic timestamps even when attention is flat', () => {
    // Degenerate case: uniform attention
    const matrix = new Float32Array([0.1, 0.1, 0.1, 0.1]);
    const tokens = [42];
    const timestamps = crossAttentionDtwTimestamps([matrix], tokens, 4, 0.02);

    expect(timestamps).toHaveLength(2);
    expect(timestamps[0]).toBeGreaterThanOrEqual(0);
    expect(timestamps[1]).toBeCloseTo(0.06, 5); // (4-1) * 0.02
  });
});
