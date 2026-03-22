import { describe, expect, it } from 'vitest';

import { AudioChunk, normalizePcmInput, PcmAudioBuffer } from '../src/audio/audio.js';

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

  it('handles Float32Array with a provided sample rate', () => {
    const input = new Float32Array([0.5, 0.25]);
    const result = normalizePcmInput(input, { sampleRate: 8000 });

    expect(result.sampleRate).toBe(8000);
    expect(result.numberOfChannels).toBe(1);
    expect(result.numberOfFrames).toBe(2);
    expect(Array.from(result.channels[0]!)).toEqual([0.5, 0.25]);
  });

  it('handles Float64Array input', () => {
    const input = new Float64Array([0.5, -0.5, 1.0]);
    const result = normalizePcmInput(input);

    expect(result.sampleRate).toBe(16000);
    expect(result.numberOfChannels).toBe(1);
    expect(result.numberOfFrames).toBe(3);
    expect(Array.from(result.channels[0]!)).toEqual([0.5, -0.5, 1.0]);
  });

  it('handles planar channel input', () => {
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

  it('handles interleaved Int16Array input', () => {
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

  it('handles interleaved Float32Array input', () => {
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

  it('throws for invalid input', () => {
    expect(() => {
      // @ts-expect-error exercising runtime validation
      normalizePcmInput({});
    }).toThrow(TypeError);
  });
});

describe('AudioChunk', () => {
  it('preserves explicit sequence and timing fields', () => {
    const chunk = new AudioChunk({
      sampleRate: 16000,
      channels: [new Float32Array(16000)],
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

  it('infers endTimeSeconds from the chunk duration', () => {
    const chunk = new AudioChunk({
      sampleRate: 16000,
      channels: [new Float32Array(32000)],
      startTimeSeconds: 5,
    });

    expect(chunk.startTimeSeconds).toBe(5);
    expect(chunk.endTimeSeconds).toBe(7);
  });

  it('leaves omitted optional properties undefined', () => {
    const chunk = new AudioChunk({
      sampleRate: 16000,
      channels: [new Float32Array(16000)],
    });

    expect(chunk.sequence).toBeUndefined();
    expect(chunk.startTimeSeconds).toBeUndefined();
    expect(chunk.endTimeSeconds).toBeUndefined();
    expect(chunk.isLast).toBeUndefined();
  });
});

describe('PcmAudioBuffer', () => {
  describe('toMono', () => {
    it('returns the same instance when the buffer is already mono', () => {
      const buffer = new PcmAudioBuffer({
        sampleRate: 16000,
        channels: [new Float32Array([1, 2, 3, 4])],
      });

      const mono = buffer.toMono();

      expect(mono).toBe(buffer);
      expect(mono.numberOfChannels).toBe(1);
    });

    it('downmixes stereo audio to a single channel', () => {
      const buffer = new PcmAudioBuffer({
        sampleRate: 16000,
        channels: [new Float32Array([1, 2, 3, 4]), new Float32Array([5, 6, 7, 8])],
      });

      const mono = buffer.toMono();

      expect(mono).not.toBe(buffer);
      expect(mono.numberOfChannels).toBe(1);
      expect(mono.sampleRate).toBe(16000);
      expect(Array.from(mono.channels[0]!)).toEqual([3, 4, 5, 6]);
    });

    it('downmixes more than two channels correctly', () => {
      const buffer = new PcmAudioBuffer({
        sampleRate: 16000,
        channels: [
          new Float32Array([1, 0, 0]),
          new Float32Array([0, 2, 0]),
          new Float32Array([0, 0, 3]),
          new Float32Array([3, 2, 1]),
        ],
      });

      const mono = buffer.toMono();

      expect(mono.numberOfChannels).toBe(1);
      expect(Array.from(mono.channels[0]!)).toEqual([1, 1, 1]);
    });
  });

  describe('sliceFrames', () => {
    const buffer = new PcmAudioBuffer({
      sampleRate: 16000,
      channels: [new Float32Array([1, 2, 3, 4, 5, 6]), new Float32Array([10, 20, 30, 40, 50, 60])],
    });

    it('slices both channels within bounds', () => {
      const slice = buffer.sliceFrames(1, 4);

      expect(slice.numberOfFrames).toBe(3);
      expect(slice.sampleRate).toBe(16000);
      expect(slice.numberOfChannels).toBe(2);
      expect(Array.from(slice.channels[0]!)).toEqual([2, 3, 4]);
      expect(Array.from(slice.channels[1]!)).toEqual([20, 30, 40]);
    });

    it('clamps a negative start frame to zero', () => {
      const slice = buffer.sliceFrames(-2, 3);

      expect(slice.numberOfFrames).toBe(3);
      expect(Array.from(slice.channels[0]!)).toEqual([1, 2, 3]);
      expect(Array.from(slice.channels[1]!)).toEqual([10, 20, 30]);
    });

    it('clamps an end frame past the buffer length', () => {
      const slice = buffer.sliceFrames(4, 10);

      expect(slice.numberOfFrames).toBe(2);
      expect(Array.from(slice.channels[0]!)).toEqual([5, 6]);
      expect(Array.from(slice.channels[1]!)).toEqual([50, 60]);
    });

    it('returns an empty buffer when endFrame does not exceed startFrame', () => {
      const slice1 = buffer.sliceFrames(3, 2);
      const slice2 = buffer.sliceFrames(3, 3);

      expect(slice1.numberOfFrames).toBe(0);
      expect(Array.from(slice1.channels[0]!)).toEqual([]);
      expect(Array.from(slice1.channels[1]!)).toEqual([]);
      expect(slice2.numberOfFrames).toBe(0);
      expect(Array.from(slice2.channels[0]!)).toEqual([]);
      expect(Array.from(slice2.channels[1]!)).toEqual([]);
    });

    it('floors the start and ceilings the end for fractional frame bounds', () => {
      const slice = buffer.sliceFrames(1.8, 3.2);

      expect(slice.numberOfFrames).toBe(3);
      expect(Array.from(slice.channels[0]!)).toEqual([2, 3, 4]);
      expect(Array.from(slice.channels[1]!)).toEqual([20, 30, 40]);
    });
  });
});
