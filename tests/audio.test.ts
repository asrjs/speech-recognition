import { describe, expect, it } from 'vitest';
import { AudioChunk } from '../src/audio/audio.js';

describe('AudioChunk', () => {
  it('should initialize with sequence, start and end time correctly', () => {
    const chunk = new AudioChunk({
      sampleRate: 16000,
      channels: [new Float32Array(16000)], // 1 second of audio
      sequence: 1,
      startTimeSeconds: 2.5,
      endTimeSeconds: 3.5,
      isLast: false,
    });

    expect(chunk.sequence).toBe(1);
    expect(chunk.startTimeSeconds).toBe(2.5);
    expect(chunk.endTimeSeconds).toBe(3.5);
    expect(chunk.isLast).toBe(false);
  });

  it('should infer endTimeSeconds from startTimeSeconds and durationSeconds', () => {
    const chunk = new AudioChunk({
      sampleRate: 16000,
      channels: [new Float32Array(32000)], // 2 seconds of audio
      startTimeSeconds: 5.0,
    });

    expect(chunk.startTimeSeconds).toBe(5.0);
    // duration is 2.0s, so endTimeSeconds should be 5.0 + 2.0 = 7.0
    expect(chunk.endTimeSeconds).toBe(7.0);
  });

  it('should have undefined endTimeSeconds if startTimeSeconds and endTimeSeconds are omitted', () => {
    const chunk = new AudioChunk({
      sampleRate: 16000,
      channels: [new Float32Array(16000)],
    });

    expect(chunk.startTimeSeconds).toBeUndefined();
    expect(chunk.endTimeSeconds).toBeUndefined();
  });

  it('should handle undefined isLast correctly', () => {
    const chunk = new AudioChunk({
      sampleRate: 16000,
      channels: [new Float32Array(16000)],
    });

    expect(chunk.isLast).toBeUndefined();
  });
});
