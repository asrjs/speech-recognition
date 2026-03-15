import { describe, expect, it } from 'vitest';
import { normalizePcmInput, PcmAudioBuffer } from '../src/audio/audio.js';

describe('normalizePcmInput', () => {
  it('handles Float32Array with default sample rate', () => {
    const input = new Float32Array([0.5, -0.5, 1.0]);
    const result = normalizePcmInput(input);
    expect(result).toBeInstanceOf(PcmAudioBuffer);
    expect(result.sampleRate).toBe(16000);
    expect(result.numberOfChannels).toBe(1);
    expect(result.numberOfFrames).toBe(3);
    expect(Array.from(result.channels[0]!)).toEqual([0.5, -0.5, 1.0]);
  });

  it('handles Float32Array with provided sample rate', () => {
    const input = new Float32Array([0.5, 0.25]);
    const result = normalizePcmInput(input, { sampleRate: 8000 });
    expect(result.sampleRate).toBe(8000);
    expect(result.numberOfChannels).toBe(1);
    expect(result.numberOfFrames).toBe(2);
    expect(Array.from(result.channels[0]!)).toEqual([0.5, 0.25]);
  });

  it('handles Float64Array', () => {
    const input = new Float64Array([0.5, -0.5, 1.0]);
    const result = normalizePcmInput(input);
    expect(result).toBeInstanceOf(PcmAudioBuffer);
    expect(result.sampleRate).toBe(16000);
    expect(result.numberOfChannels).toBe(1);
    expect(result.numberOfFrames).toBe(3);
    expect(Array.from(result.channels[0]!)).toEqual([0.5, -0.5, 1.0]);
  });

  it('handles AudioBufferLike with channel data', () => {
    const input = {
      sampleRate: 48000,
      numberOfChannels: 2,
      numberOfFrames: 2,
      durationSeconds: 2 / 48000,
      channels: [new Float32Array([0.5, 0.25]), new Float32Array([-0.5, -0.25])],
    };
    const result = normalizePcmInput(input);
    expect(result.sampleRate).toBe(48000);
    expect(result.numberOfChannels).toBe(2);
    expect(result.numberOfFrames).toBe(2);
    expect(Array.from(result.channels[0]!)).toEqual([0.5, 0.25]);
    expect(Array.from(result.channels[1]!)).toEqual([-0.5, -0.25]);
  });

  it('handles AudioBufferLike with Int16Array data (interleaved)', () => {
    const input = {
      sampleRate: 16000,
      numberOfChannels: 2,
      numberOfFrames: 2,
      durationSeconds: 2 / 16000,
      data: new Int16Array([16384, -16384, 8192, -8192]),
    };
    const result = normalizePcmInput(input);
    expect(result.sampleRate).toBe(16000);
    expect(result.numberOfChannels).toBe(2);
    expect(result.numberOfFrames).toBe(2);
    expect(Array.from(result.channels[0]!)).toEqual([0.5, 0.25]);
    expect(Array.from(result.channels[1]!)).toEqual([-0.5, -0.25]);
  });

  it('handles AudioBufferLike with Float32Array data (interleaved)', () => {
    const input = {
      sampleRate: 24000,
      numberOfChannels: 2,
      numberOfFrames: 2,
      durationSeconds: 2 / 24000,
      data: new Float32Array([0.5, -0.5, 0.25, -0.25]),
    };
    const result = normalizePcmInput(input);
    expect(result.sampleRate).toBe(24000);
    expect(result.numberOfChannels).toBe(2);
    expect(result.numberOfFrames).toBe(2);
    expect(Array.from(result.channels[0]!)).toEqual([0.5, 0.25]);
    expect(Array.from(result.channels[1]!)).toEqual([-0.5, -0.25]);
  });

  it('throws TypeError for invalid input', () => {
    expect(() => {
      // @ts-expect-error Testing invalid input
      normalizePcmInput({});
    }).toThrow(TypeError);
  });
});
