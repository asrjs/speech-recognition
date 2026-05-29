/**
 * Tests for standalone chunking/ module.
 * Phase B: drift-handler + vad-segmenter moved from whisper-seq2seq.
 */

import { describe, it, expect } from 'vitest';
import {
  DriftHandler,
  mergeVadSegments,
  type VadSpeechSegment,
  type WhisperVadBackend,
} from '../src/chunking/index.js';

const SAMPLE_RATE = 16000;

// ---------------------------------------------------------------------------
// DriftHandler
// ---------------------------------------------------------------------------

describe('DriftHandler', () => {
  it('starts with seek at 0', () => {
    const handler = new DriftHandler();
    expect(handler.getSeekSeconds(SAMPLE_RATE)).toBe(0);
  });

  it('advances correctly', () => {
    const handler = new DriftHandler();
    handler.advanceBy(5.0, SAMPLE_RATE);
    handler.advanceBy(3.5, SAMPLE_RATE);
    expect(handler.getSeekSeconds(SAMPLE_RATE)).toBeCloseTo(8.5, 0);
  });

  it('resets to 0', () => {
    const handler = new DriftHandler();
    handler.advanceBy(10.0, SAMPLE_RATE);
    handler.reset(1000);
    expect(handler.getSeekSeconds(SAMPLE_RATE)).toBe(0);
  });

  it('corrects timestamps when drift exceeds maxDrift', () => {
    const handler = new DriftHandler();
    const result = handler.correctTimestamps(30.0, 35.0, SAMPLE_RATE, 1.0);
    expect(result.corrected).toBe(true);
    expect(result.start).toBeCloseTo(0, 0);
    expect(result.end).toBeCloseTo(5, 0);
  });

  it('keeps timestamps when within maxDrift', () => {
    const handler = new DriftHandler();
    handler.advanceBy(10.0, SAMPLE_RATE);
    const result = handler.correctTimestamps(10.3, 15.7, SAMPLE_RATE, 1.0);
    expect(result.corrected).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// mergeVadSegments
// ---------------------------------------------------------------------------

describe('mergeVadSegments', () => {
  it('merges close segments', () => {
    const segs: VadSpeechSegment[] = [
      { startSeconds: 0, endSeconds: 2.0, durationSeconds: 2.0 },
      { startSeconds: 2.1, endSeconds: 4.0, durationSeconds: 1.9 },
    ];
    const result = mergeVadSegments(segs, 200, 400, 29000);
    expect(result).toHaveLength(1);
  });

  it('keeps far segments separate', () => {
    const segs: VadSpeechSegment[] = [
      { startSeconds: 0, endSeconds: 1.0, durationSeconds: 1.0 },
      { startSeconds: 3.0, endSeconds: 4.0, durationSeconds: 1.0 },
    ];
    const result = mergeVadSegments(segs, 200, 400, 29000);
    expect(result).toHaveLength(2);
  });

  it('pads segment edges', () => {
    const segs: VadSpeechSegment[] = [
      { startSeconds: 1.0, endSeconds: 3.0, durationSeconds: 2.0 },
    ];
    const result = mergeVadSegments(segs, 100, 500, 29000);
    expect(result[0]!.startSeconds).toBeCloseTo(0.5, 1);
    expect(result[0]!.endSeconds).toBeCloseTo(3.5, 1);
  });

  it('filters short segments', () => {
    const segs: VadSpeechSegment[] = [
      { startSeconds: 0, endSeconds: 0.1, durationSeconds: 0.1 },
      { startSeconds: 1.0, endSeconds: 3.0, durationSeconds: 2.0 },
    ];
    const result = mergeVadSegments(segs, 100, 400, 29000);
    expect(result).toHaveLength(1);
  });

  it('splits long segments', () => {
    const segs: VadSpeechSegment[] = [
      { startSeconds: 0, endSeconds: 35.0, durationSeconds: 35.0 },
    ];
    const result = mergeVadSegments(segs, 100, 400, 29000);
    expect(result.length).toBeGreaterThanOrEqual(2);
  });
});

// ---------------------------------------------------------------------------
// WhisperVadBackend interface
// ---------------------------------------------------------------------------

describe('WhisperVadBackend', () => {
  it('can be implemented by mock', async () => {
    const mock: WhisperVadBackend = {
      async segment(audio, sampleRate, _threshold) {
        const duration = audio.length / sampleRate;
        if (duration < 0.25) return [];
        return [{ startSeconds: 0, endSeconds: duration, durationSeconds: duration }];
      },
    };
    const result = await mock.segment(new Float32Array(16000 * 5), 16000, 0.5);
    expect(result).toHaveLength(1);
  });
});
