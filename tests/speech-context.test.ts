import { describe, expect, it } from 'vitest';
import { SpeechContextBuffer } from '../src/runtime/speech-context.js';

describe('SpeechContextBuffer', () => {
  it('preserves captured silence and includes overlapping VAD audio once', () => {
    const context = new SpeechContextBuffer({ sampleRate: 1, maxDurationSeconds: 10, maxGapSeconds: 3 });
    context.write(0, Float32Array.of(1, 2, 3, 4));
    context.write(4, Float32Array.of(5, 6, 7, 8, 9, 10));

    const first = context.snapshot(1, 4, 1, 3);
    const second = context.snapshot(2, 7, 4, 6);

    expect(Array.from(first.pcm)).toEqual([2, 3, 4]);
    expect(Array.from(second.pcm)).toEqual([2, 3, 4, 5, 6, 7]);
    expect(second.id).toBe(first.id);
    expect(second.revision).toBe(first.revision + 1);
  });

  it('starts a new context after a logical speech gap exceeds the limit', () => {
    const context = new SpeechContextBuffer({ sampleRate: 1, maxDurationSeconds: 10, maxGapSeconds: 2 });
    context.write(0, Float32Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
    const first = context.snapshot(0, 2, 0, 1);
    const next = context.snapshot(4, 8, 5, 6);

    expect(next.id).not.toBe(first.id);
    expect(next.revision).toBe(1);
    expect(Array.from(next.pcm)).toEqual([5, 6, 7, 8]);
  });

  it('fails closed on capture discontinuity, overwritten context and mismatched boundaries', () => {
    const context = new SpeechContextBuffer({ sampleRate: 1, maxDurationSeconds: 5, maxGapSeconds: 2 });
    context.write(0, Float32Array.of(1, 2, 3));
    expect(() => context.write(4, Float32Array.of(5))).toThrow(/discontinuity/);
    context.write(3, Float32Array.of(4, 5, 6, 7, 8));
    expect(() => context.snapshot(0, 5)).toThrow(/overwritten/);
    expect(() => context.snapshot(4, 9)).toThrow(/beyond captured audio/);
  });

  it('fails when a live context was overwritten despite a valid new segment range', () => {
    const context = new SpeechContextBuffer({ sampleRate: 1, maxDurationSeconds: 10, maxGapSeconds: 5 });
    context.write(0, Float32Array.from({ length: 10 }, (_, index) => index + 1));
    context.snapshot(0, 4, 0, 3);
    context.write(10, Float32Array.of(11, 12));

    expect(() => context.snapshot(6, 8, 6, 7)).toThrow(/Speech context frame 0 was overwritten/);
  });

  it('starts a bounded context at the first unowned frame when maximum duration is reached', () => {
    const context = new SpeechContextBuffer({ sampleRate: 1, maxDurationSeconds: 6, maxGapSeconds: 5 });
    context.write(0, Float32Array.from({ length: 4 }, (_, index) => index + 1));
    const first = context.snapshot(0, 4, 0, 3);
    context.write(4, Float32Array.from({ length: 4 }, (_, index) => index + 5));
    const next = context.snapshot(2, 8, 4, 7);

    expect(next.id).not.toBe(first.id);
    expect(next.startFrame).toBe(first.endFrame);
    expect(next.endFrame).toBe(8);
    expect(Array.from(next.pcm)).toEqual([5, 6, 7, 8]);
  });

  it('retains detector-lag PCM beyond the inference duration before splitting without replay', () => {
    const context = new SpeechContextBuffer({
      sampleRate: 10,
      maxDurationSeconds: 30,
      retentionDurationSeconds: 32,
      maxGapSeconds: 3,
    });
    context.write(0, Float32Array.from({ length: 298 }, (_, index) => index));
    const first = context.snapshot(0, 298, 0, 297);
    context.write(298, Float32Array.of(298, 299, 300, 301));
    const next = context.snapshot(295, 302, 298, 301);

    expect(first.pcm.length).toBe(298);
    expect(next.id).not.toBe(first.id);
    expect(next.startFrame).toBe(first.endFrame);
    expect(next.endFrame - next.startFrame).toBe(4);
    expect(Array.from(next.pcm)).toEqual([298, 299, 300, 301]);
  });

  it('includes captured nonzero preroll and never duplicates prior-context audio', () => {
    const context = new SpeechContextBuffer({
      sampleRate: 1,
      maxDurationSeconds: 20,
      retentionDurationSeconds: 30,
      maxGapSeconds: 3,
      prerollSeconds: 3,
    });
    context.write(0, Float32Array.from({ length: 30 }, (_, index) => index + 1));
    const first = context.snapshot(10, 15, 11, 14);
    const next = context.snapshot(13, 20, 18, 19);

    expect(first.startFrame).toBe(7);
    expect(Array.from(first.pcm)).toEqual([8, 9, 10, 11, 12, 13, 14, 15]);
    expect(next.id).not.toBe(first.id);
    expect(next.startFrame).toBe(first.endFrame);
    expect(Array.from(next.pcm)).toEqual([16, 17, 18, 19, 20]);
  });

  it('clamps preroll to captured audio and accounts for it in the logical duration limit', () => {
    const context = new SpeechContextBuffer({
      sampleRate: 1,
      maxDurationSeconds: 5,
      retentionDurationSeconds: 15,
      maxGapSeconds: 3,
      prerollSeconds: 2,
    });
    context.write(4, Float32Array.from({ length: 10 }, (_, index) => index + 1));
    const first = context.snapshot(5, 7, 6, 7);
    const next = context.snapshot(8, 11, 9, 10);

    expect(first.startFrame).toBe(4);
    expect(Array.from(first.pcm)).toEqual([1, 2, 3]);
    expect(next.id).not.toBe(first.id);
    expect(next.startFrame).toBe(first.endFrame);
    expect(Array.from(next.pcm)).toEqual([4, 5, 6, 7]);
  });

  it('splits on extracted PCM duration even when logical speech bounds fit', () => {
    const context = new SpeechContextBuffer({
      sampleRate: 1,
      maxDurationSeconds: 5,
      retentionDurationSeconds: 10,
      maxGapSeconds: 3,
      prerollSeconds: 1,
    });
    context.write(0, Float32Array.from({ length: 10 }, (_, index) => index + 1));
    const first = context.snapshot(1, 3, 2, 3);
    const next = context.snapshot(3, 7, 4, 5);

    expect(first.endFrame - first.startFrame).toBe(3);
    expect(next.id).not.toBe(first.id);
    expect(next.startFrame).toBe(first.endFrame);
    expect(next.endFrame - next.startFrame).toBe(4);
    expect(Array.from(next.pcm)).toEqual([4, 5, 6, 7]);
  });

  it('resets captured frames between microphone sessions', () => {
    const context = new SpeechContextBuffer({ sampleRate: 1, maxDurationSeconds: 5, maxGapSeconds: 2 });
    context.write(0, Float32Array.of(1, 2));
    const first = context.snapshot(0, 2);
    context.reset();
    context.write(0, Float32Array.of(8, 9));
    const next = context.snapshot(0, 2);

    expect(next.id).not.toBe(first.id);
    expect(Array.from(next.pcm)).toEqual([8, 9]);
  });

  it('finalizes once with bounded captured right context and a new revision of the same context', () => {
    const context = new SpeechContextBuffer({
      sampleRate: 1,
      maxDurationSeconds: 10,
      maxGapSeconds: 2,
      prerollSeconds: 3,
    });
    context.write(0, Float32Array.from({ length: 10 }, (_, index) => index + 1));
    const first = context.snapshot(2, 5, 2, 5);
    const final = context.finalize();

    expect(final).not.toBeNull();
    expect(final).toMatchObject({
      id: first.id,
      revision: first.revision + 1,
      startFrame: first.startFrame,
      endFrame: 8,
      finalized: true,
    });
    expect(Array.from(final!.pcm)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(context.finalize()).toBeNull();
  });

  it('skips finalization without a context or room under the duration cap', () => {
    const empty = new SpeechContextBuffer({ sampleRate: 1, maxDurationSeconds: 5, maxGapSeconds: 2 });
    expect(empty.finalize()).toBeNull();

    const context = new SpeechContextBuffer({
      sampleRate: 1,
      maxDurationSeconds: 5,
      retentionDurationSeconds: 10,
      maxGapSeconds: 2,
      prerollSeconds: 3,
    });
    context.write(0, Float32Array.from({ length: 10 }, (_, index) => index + 1));
    const first = context.snapshot(0, 5);

    expect(context.finalize()).toBeNull();
    expect(first.pcm.length).toBe(5);
  });

  it('fails finalization when the existing context start was overwritten', () => {
    const context = new SpeechContextBuffer({
      sampleRate: 1,
      maxDurationSeconds: 5,
      retentionDurationSeconds: 5,
      maxGapSeconds: 2,
      prerollSeconds: 3,
    });
    context.write(0, Float32Array.of(1, 2, 3));
    context.snapshot(0, 2);
    context.write(3, Float32Array.of(4, 5, 6, 7));

    expect(() => context.finalize()).toThrow(/was overwritten/);
  });
});
