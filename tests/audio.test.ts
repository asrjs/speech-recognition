import { describe, expect, it } from 'vitest';
import { PcmAudioBuffer } from '../src/audio/audio.js';

describe('PcmAudioBuffer', () => {
  describe('toMono', () => {
    it('returns the same instance if already mono', () => {
      const buffer = new PcmAudioBuffer({
        sampleRate: 16000,
        channels: [new Float32Array([1, 2, 3, 4])],
      });

      const mono = buffer.toMono();
      expect(mono).toBe(buffer);
      expect(mono.numberOfChannels).toBe(1);
    });

    it('downmixes multiple channels to a single mono channel', () => {
      const buffer = new PcmAudioBuffer({
        sampleRate: 16000,
        channels: [
          new Float32Array([1, 2, 3, 4]),
          new Float32Array([5, 6, 7, 8]),
        ],
      });

      const mono = buffer.toMono();
      expect(mono).not.toBe(buffer);
      expect(mono.numberOfChannels).toBe(1);
      expect(mono.sampleRate).toBe(16000);
      expect(Array.from(mono.channels[0]!)).toEqual([3, 4, 5, 6]);
    });

    it('handles more than 2 channels correctly', () => {
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
      expect(Array.from(mono.channels[0]!)).toEqual([1, 1, 1]); // (1+0+0+3)/4 = 1, (0+2+0+2)/4 = 1, (0+0+3+1)/4 = 1
    });
  });

  describe('sliceFrames', () => {
    const buffer = new PcmAudioBuffer({
      sampleRate: 16000,
      channels: [
        new Float32Array([1, 2, 3, 4, 5, 6]),
        new Float32Array([10, 20, 30, 40, 50, 60]),
      ],
    });

    it('slices normally within bounds', () => {
      const slice = buffer.sliceFrames(1, 4);
      expect(slice.numberOfFrames).toBe(3);
      expect(slice.sampleRate).toBe(16000);
      expect(slice.numberOfChannels).toBe(2);
      expect(Array.from(slice.channels[0]!)).toEqual([2, 3, 4]);
      expect(Array.from(slice.channels[1]!)).toEqual([20, 30, 40]);
    });

    it('clamps startFrame to 0 when negative', () => {
      const slice = buffer.sliceFrames(-2, 3);
      expect(slice.numberOfFrames).toBe(3);
      expect(Array.from(slice.channels[0]!)).toEqual([1, 2, 3]);
    });

    it('clamps endFrame to numberOfFrames when greater than length', () => {
      const slice = buffer.sliceFrames(4, 10);
      expect(slice.numberOfFrames).toBe(2);
      expect(Array.from(slice.channels[0]!)).toEqual([5, 6]);
    });

    it('returns empty buffer when endFrame <= startFrame', () => {
      const slice1 = buffer.sliceFrames(3, 2);
      expect(slice1.numberOfFrames).toBe(0);
      expect(Array.from(slice1.channels[0]!)).toEqual([]);

      const slice2 = buffer.sliceFrames(3, 3);
      expect(slice2.numberOfFrames).toBe(0);
      expect(Array.from(slice2.channels[0]!)).toEqual([]);
    });

    it('handles floating point inputs by flooring start and ceiling end', () => {
      const slice = buffer.sliceFrames(1.8, 3.2);
      // clampedStart = Math.floor(1.8) = 1
      // clampedEnd = Math.ceil(3.2) = 4
      expect(slice.numberOfFrames).toBe(3);
      expect(Array.from(slice.channels[0]!)).toEqual([2, 3, 4]);
    });
  });
});
