import {
  AudioChunker,
  AudioFeatureCache,
  LayeredAudioBuffer,
  createAudioCacheKey,
} from '@asrjs/speech-recognition/realtime';
import { FrameAlignedTokenMerger, LcsPtfaTokenMerger } from '@asrjs/speech-recognition/inference';
import { describe, expect, it } from 'vitest';

describe('app helpers', () => {
  it('caches derived values by normalized audio content', async () => {
    const cache = new AudioFeatureCache<Float32Array>({
      maxSizeMB: 1,
      estimateSizeBytes: (value) => value.byteLength,
    });
    const input = new Float32Array([0, 0.1, -0.1, 0.25]);
    let computeCount = 0;

    const first = await cache.getOrCreate(input, async () => {
      computeCount += 1;
      return new Float32Array([1, 2, 3]);
    });
    const second = await cache.getOrCreate(input, async () => {
      computeCount += 1;
      return new Float32Array([4, 5, 6]);
    });

    expect(first.cached).toBe(false);
    expect(second.cached).toBe(true);
    expect(computeCount).toBe(1);
    expect(first.key).toBe(createAudioCacheKey(input));
    expect(cache.getStats().hitCount).toBe(1);
  });

  it('does not cache values when the factory throws an error', async () => {
    const cache = new AudioFeatureCache<Float32Array>({
      maxSizeMB: 1,
      estimateSizeBytes: (value) => value.byteLength,
    });
    const input = new Float32Array([0, 0.1, -0.1, 0.25]);
    let computeCount = 0;

    await expect(
      cache.getOrCreate(input, async () => {
        computeCount += 1;
        throw new Error('Factory error');
      }),
    ).rejects.toThrow('Factory error');

    const second = await cache.getOrCreate(input, async () => {
      computeCount += 1;
      return new Float32Array([4, 5, 6]);
    });

    expect(second.cached).toBe(false);
    expect(computeCount).toBe(2);
    expect(cache.getStats().hitCount).toBe(0);
  });

  it('splits audio into overlapping chunks for long-form orchestration', () => {
    const chunker = new AudioChunker({
      chunkLengthMs: 1000,
      overlapMs: 250,
    });

    const chunks = chunker.split(new Float32Array(16000 * 3));

    expect(chunks.length).toBeGreaterThan(2);
    expect(chunks[0]?.durationSeconds).toBeCloseTo(1, 2);
    expect(chunks[1]?.startTimeSeconds).toBeCloseTo(0.75, 2);
  });

  it('stores per-chunk derived layers alongside buffered audio', () => {
    const buffer = new LayeredAudioBuffer<{ transcript: string; features: Float32Array }>({
      maxWindowMs: 1500,
      overlapMs: 250,
    });

    const first = buffer.push(new Float32Array(8000), 0);
    buffer.setLayer(first.chunk.sequence ?? 0, 'transcript', 'hello');
    buffer.push(new Float32Array(8000), 0.5);
    buffer.push(new Float32Array(8000), 1.0);

    expect(buffer.getLayer(first.chunk.sequence ?? 0, 'transcript')).toBe('hello');
    expect(buffer.getBufferedDurationSeconds()).toBeLessThanOrEqual(1.5);
    expect(buffer.toPcmAudioBuffer().numberOfFrames).toBeGreaterThan(0);
  });

  it('merges overlapping frame-aligned token streams', () => {
    const merger = new FrameAlignedTokenMerger({
      frameTimeStride: 0.1,
      timeTolerance: 0.12,
    });

    merger.processChunk(
      {
        tokenIds: [1, 2, 3],
        frameIndices: [0, 1, 2],
      },
      0,
      0,
    );
    const merged = merger.processChunk(
      {
        tokenIds: [2, 3, 4],
        frameIndices: [0, 1, 3],
      },
      0.2,
      0.2,
    );

    expect(merged.anchorsFound).toBeGreaterThan(0);
    expect(merger.getAllTokens().length).toBeGreaterThan(0);
  });

  it('supports LCS/PTFA style overlap arbitration', () => {
    const merger = new LcsPtfaTokenMerger({
      frameTimeStride: 0.1,
      timeTolerance: 0.12,
      sequenceAnchorLength: 2,
    });

    merger.processChunk(
      {
        tokenIds: [10, 11, 12, 13],
        frameIndices: [0, 1, 2, 3],
        logProbs: [-0.2, -0.2, -0.1, -0.1],
      },
      0,
      0,
    );
    const merged = merger.processChunk(
      {
        tokenIds: [12, 13, 14],
        frameIndices: [0, 1, 2],
        logProbs: [-0.1, -0.1, -0.05],
      },
      0.2,
      0.2,
    );

    expect(merged.lcsLength).toBeGreaterThan(0);
    expect(merger.getState().pendingCount).toBeGreaterThanOrEqual(0);
  });
});
