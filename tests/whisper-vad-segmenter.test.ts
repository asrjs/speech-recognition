/**
 * Tests for VAD segmenter — audio pre-segmentation for Whisper chunking.
 * Phase 6: pure segment merging + backend interface.
 */

import { describe, it, expect } from 'vitest';
import {
  mergeVadSegments,
  type VadSpeechSegment,
  type WhisperVadBackend,
} from '../src/models/whisper-seq2seq/vad-segmenter.js';

// ---------------------------------------------------------------------------
// mergeVadSegments tests
// ---------------------------------------------------------------------------

describe('mergeVadSegments', () => {
  it('merges close segments', () => {
    const segments: VadSpeechSegment[] = [
      { startSeconds: 0, endSeconds: 2.0, durationSeconds: 2.0 },
      { startSeconds: 2.1, endSeconds: 4.0, durationSeconds: 1.9 },
    ];
    const result = mergeVadSegments(segments, 200, 400, 29000);
    // Gap = 0.1s = 100ms < minSilenceDuration 200ms → merged
    expect(result).toHaveLength(1);
    expect(result[0]!.startSeconds).toBeCloseTo(0, 1);
    expect(result[0]!.endSeconds).toBeCloseTo(4.4, 1); // 4.0 + 0.4 pad
  });

  it('keeps far segments separate', () => {
    const segments: VadSpeechSegment[] = [
      { startSeconds: 0, endSeconds: 1.0, durationSeconds: 1.0 },
      { startSeconds: 3.0, endSeconds: 4.0, durationSeconds: 1.0 },
    ];
    const result = mergeVadSegments(segments, 200, 400, 29000);
    // Gap = 2.0s > 200ms → separate
    expect(result).toHaveLength(2);
  });

  it('pads segment edges', () => {
    const segments: VadSpeechSegment[] = [
      { startSeconds: 1.0, endSeconds: 3.0, durationSeconds: 2.0 },
    ];
    const result = mergeVadSegments(segments, 100, 500, 29000);
    expect(result).toHaveLength(1);
    expect(result[0]!.startSeconds).toBeCloseTo(0.5, 1); // 1.0 - 0.5
    expect(result[0]!.endSeconds).toBeCloseTo(3.5, 1);   // 3.0 + 0.5
  });

  it('clamps padded start to 0', () => {
    const segments: VadSpeechSegment[] = [
      { startSeconds: 0.1, endSeconds: 1.0, durationSeconds: 0.9 },
    ];
    const result = mergeVadSegments(segments, 100, 400, 29000);
    expect(result[0]!.startSeconds).toBe(0.0);
  });

  it('caps segments at max duration', () => {
    const segments: VadSpeechSegment[] = [
      { startSeconds: 0, endSeconds: 35.0, durationSeconds: 35.0 },
    ];
    const maxMs = 29000; // 29s
    const result = mergeVadSegments(segments, 100, 400, maxMs);
    expect(result).toHaveLength(2); // split into 2
    expect(result[0]!.startSeconds).toBeCloseTo(0, 1);
    expect(result[0]!.endSeconds).toBeLessThanOrEqual(29.0);
    expect(result[1]!.startSeconds).toBeLessThan(30.0);
    expect(result[1]!.endSeconds).toBeCloseTo(35.4, 1);
  });

  it('filters segments shorter than min duration', () => {
    const segments: VadSpeechSegment[] = [
      { startSeconds: 0, endSeconds: 0.1, durationSeconds: 0.1 },
      { startSeconds: 1.0, endSeconds: 3.0, durationSeconds: 2.0 },
    ];
    const result = mergeVadSegments(segments, 100, 400, 29000);
    // 0.1s < 250ms minSpeechDuration → filtered
    expect(result).toHaveLength(1);
    expect(result[0]!.startSeconds).toBeCloseTo(0.6, 1); // 1.0 - 0.4
  });

  it('matches merged segment duration correctly', () => {
    const segments: VadSpeechSegment[] = [
      { startSeconds: 0, endSeconds: 5.0, durationSeconds: 5.0 },
      { startSeconds: 5.05, endSeconds: 10.0, durationSeconds: 4.95 },
    ];
    // Gap = 50ms < 100ms → merged
    const result = mergeVadSegments(segments, 100, 400, 29000);
    expect(result).toHaveLength(1);
    // Start: 0 - 0.4 → clamped to 0
    // End: 10.0 + 0.4 = 10.4
    expect(result[0]!.endSeconds).toBeCloseTo(10.4, 1);
  });
});

// ---------------------------------------------------------------------------
// Interface validation
// ---------------------------------------------------------------------------

describe('WhisperVadBackend interface', () => {
  it('can be implemented by mock backend', async () => {
    const mockBackend: WhisperVadBackend = {
      async segment(audio, sampleRate, _threshold) {
        if (audio.length === 0) return [];
        // Simple: one segment covering the whole audio
        const duration = audio.length / sampleRate;
        if (duration < 0.25) return []; // too short
        return [{
          startSeconds: 0,
          endSeconds: duration,
          durationSeconds: duration,
        }];
      },
    };

    const audio = new Float32Array(16000 * 5); // 5 seconds
    const result = await mockBackend.segment(audio, 16000, 0.5);
    expect(result).toHaveLength(1);
    expect(result[0]!.durationSeconds).toBeCloseTo(5.0, 0);
  });

  it('returns empty for very short audio', async () => {
    const mockBackend: WhisperVadBackend = {
      async segment(audio, sampleRate, _threshold) {
        const duration = audio.length / sampleRate;
        if (duration < 0.25) return [];
        return [{ startSeconds: 0, endSeconds: duration, durationSeconds: duration }];
      },
    };

    const audio = new Float32Array(16000 * 0.1); // 100ms
    const result = await mockBackend.segment(audio, 16000, 0.5);
    expect(result).toHaveLength(0);
  });
});
