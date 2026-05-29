/**
 * Tests for DriftHandler — whisper.cpp-style seek counter for long audio.
 * Phase 5: prevents cumulative timestamp drift in multi-chunk transcription.
 */

import { describe, it, expect } from 'vitest';
import { DriftHandler } from '../src/models/whisper-seq2seq/drift-handler.js';

const SAMPLE_RATE = 16000;

describe('DriftHandler', () => {
  it('starts with seek at 0', () => {
    const handler = new DriftHandler();
    const handler2 = new DriftHandler();
    handler2.reset(480000); // 30s
    expect(handler.getSeekSeconds(SAMPLE_RATE)).toBe(0);
    expect(handler2.getSeekSeconds(SAMPLE_RATE)).toBe(0);
  });

  it('reset() sets seek to 0', () => {
    const handler = new DriftHandler();
    handler.advanceBy(10.0, SAMPLE_RATE);
    expect(handler.getSeekSeconds(SAMPLE_RATE)).toBeGreaterThan(9.9);
    handler.reset(1000);
    expect(handler.getSeekSeconds(SAMPLE_RATE)).toBe(0);
  });

  it('advanceBy() increments seek counter', () => {
    const handler = new DriftHandler();
    handler.advanceBy(5.0, SAMPLE_RATE);
    expect(handler.getSeekSeconds(SAMPLE_RATE)).toBeCloseTo(5.0, 0);
  });

  it('multiple advances accumulate', () => {
    const handler = new DriftHandler();
    handler.advanceBy(2.0, SAMPLE_RATE);
    handler.advanceBy(3.5, SAMPLE_RATE);
    handler.advanceBy(1.0, SAMPLE_RATE);
    expect(handler.getSeekSeconds(SAMPLE_RATE)).toBeCloseTo(6.5, 0);
  });

  it('correctTimestamps returns model timestamps when drift is small', () => {
    const handler = new DriftHandler();
    // No advance yet — seek is at 0
    const result = handler.correctTimestamps(0.0, 5.0, SAMPLE_RATE);
    expect(result.start).toBeCloseTo(0.0, 1);
    expect(result.end).toBeCloseTo(5.0, 1);
    expect(result.corrected).toBe(false);
  });

  it('correctTimestamps uses seek when drift exceeds maxDrift', () => {
    const handler = new DriftHandler();
    // Simulate 30 seconds of processed audio, but model thinks it's been 30s
    // With no drift: seek=0, model says start=30 → drift=30s > maxDrift=1s
    const result = handler.correctTimestamps(30.0, 35.0, SAMPLE_RATE, 1.0);
    expect(result.corrected).toBe(true);
    expect(result.start).toBeCloseTo(0.0, 0); // reset to seek
    expect(result.end).toBeCloseTo(5.0, 0);  // duration preserved
  });

  it('correctTimestamps preserves duration', () => {
    const handler = new DriftHandler();
    handler.advanceBy(10.0, SAMPLE_RATE); // seek is at 10.0s
    // Model thinks it transcribed from 10.0 to 15.0 — correct
    const result = handler.correctTimestamps(10.0, 15.0, SAMPLE_RATE, 1.0);
    expect(result.corrected).toBe(false);
    expect(result.start).toBeCloseTo(10.0, 1);
    expect(result.end).toBeCloseTo(15.0, 1);
  });

  it('correctTimestamps defaults maxDrift to 1.0 seconds', () => {
    const handler = new DriftHandler();
    handler.advanceBy(20.0, SAMPLE_RATE); // seek at 20s
    // Model reports start at 25s while seek is at 20s → drift = 5s > 1s
    const result = handler.correctTimestamps(25.0, 30.0, SAMPLE_RATE);
    expect(result.corrected).toBe(true);
    expect(result.start).toBeCloseTo(20.0, 0);
    expect(result.end).toBeCloseTo(25.0, 0); // 5s duration
  });

  it('correctTimestamps does not correct when within maxDrift', () => {
    const handler = new DriftHandler();
    handler.advanceBy(10.0, SAMPLE_RATE); // seek at 10.0s
    // Model says 10.3 → 15.7: drift is 0.3 < 1.0
    const result = handler.correctTimestamps(10.3, 15.7, SAMPLE_RATE, 1.0);
    expect(result.corrected).toBe(false);
    expect(result.start).toBeCloseTo(10.3, 1);
    expect(result.end).toBeCloseTo(15.7, 1);
  });

  it('handles zero-duration segments', () => {
    const handler = new DriftHandler();
    const result = handler.correctTimestamps(0.0, 0.0, SAMPLE_RATE);
    expect(result.start).toBe(0);
    expect(result.end).toBe(0);
    expect(result.corrected).toBe(false);
  });
});
