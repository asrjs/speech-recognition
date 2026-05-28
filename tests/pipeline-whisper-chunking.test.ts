import { planWhisperChunks, type WhisperChunkPlan } from '@asrjs/speech-recognition';
import { describe, expect, it } from 'vitest';

describe('Whisper chunk planning helpers', () => {
  it('creates overlapping chunks with transformer-compatible stride metadata', () => {
    const chunks = planWhisperChunks(60 * 16_000, 16_000, 30, 5);

    expect(chunks).toEqual<WhisperChunkPlan[]>([
      {
        index: 0,
        startSample: 0,
        endSample: 480_000,
        inputLengthSamples: 480_000,
        isFirst: true,
        isLast: false,
        stride: [480_000, 0, 80_000],
        startTime: 0,
        endTime: 30,
      },
      {
        index: 1,
        startSample: 320_000,
        endSample: 800_000,
        inputLengthSamples: 480_000,
        isFirst: false,
        isLast: false,
        stride: [480_000, 80_000, 80_000],
        startTime: 20,
        endTime: 50,
      },
      {
        index: 2,
        startSample: 640_000,
        endSample: 960_000,
        inputLengthSamples: 320_000,
        isFirst: false,
        isLast: true,
        stride: [320_000, 80_000, 0],
        startTime: 40,
        endTime: 60,
      },
    ]);
  });

  it('uses chunkLengthSeconds / 6 as the default symmetric stride', () => {
    const chunks = planWhisperChunks(60 * 16_000, 16_000, 30);

    expect(chunks.map((chunk) => chunk.stride)).toEqual([
      [480_000, 0, 80_000],
      [480_000, 80_000, 80_000],
      [320_000, 80_000, 0],
    ]);
  });

  it('accepts asymmetric left and right strides', () => {
    const chunks = planWhisperChunks(50 * 16_000, 16_000, 30, [4, 6]);

    expect(chunks.map((chunk) => ({ startSample: chunk.startSample, stride: chunk.stride }))).toEqual([
      { startSample: 0, stride: [480_000, 0, 96_000] },
      { startSample: 320_000, stride: [480_000, 64_000, 0] },
    ]);
  });

  it('returns a single unstrided chunk when chunking is disabled or audio fits', () => {
    expect(planWhisperChunks(15 * 16_000, 16_000, 0)).toEqual([
      {
        index: 0,
        startSample: 0,
        endSample: 240_000,
        inputLengthSamples: 240_000,
        isFirst: true,
        isLast: true,
        stride: [240_000, 0, 0],
        startTime: 0,
        endTime: 15,
      },
    ]);

    expect(planWhisperChunks(10 * 16_000, 16_000, 30)).toHaveLength(1);
  });

  it('rejects stride settings that do not leave forward progress', () => {
    expect(() => planWhisperChunks(60 * 16_000, 16_000, 30, 15)).toThrow(
      /stride.*less than half/i,
    );
    expect(() => planWhisperChunks(60 * 16_000, 16_000, 30, [12, 18])).toThrow(
      /left.*right.*less than chunk/i,
    );
  });
});
