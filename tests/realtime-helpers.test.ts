import {
  AudioRingBuffer,
  StreamingWindowBuilder,
  UtteranceTranscriptMerger,
  type TranscriptResult,
} from 'asr.js';
import { describe, expect, it } from 'vitest';

describe('realtime helpers', () => {
  it('stores and reads PCM frames from an overwrite-safe ring buffer', () => {
    const ring = new AudioRingBuffer({
      sampleRate: 4,
      durationSeconds: 2,
    });

    ring.write(new Float32Array([1, 2, 3, 4]));
    ring.write(new Float32Array([5, 6, 7, 8, 9]));

    expect(ring.getCurrentFrame()).toBe(9);
    expect(ring.getBaseFrameOffset()).toBe(1);
    expect(Array.from(ring.read(1, 9))).toEqual([2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('builds initial and cursor-aware streaming windows from the ring buffer head', () => {
    const ring = new AudioRingBuffer({
      sampleRate: 4,
      durationSeconds: 20,
    });
    const builder = new StreamingWindowBuilder(ring, null, {
      sampleRate: 4,
      minInitialDurationSec: 1,
      minDurationSec: 2,
      maxDurationSec: 5,
    });

    ring.write(new Float32Array([1, 2, 3, 4]));
    const initial = builder.buildWindow();
    expect(initial?.isInitial).toBe(true);
    expect(initial?.durationSeconds).toBeCloseTo(1, 5);

    builder.markSentenceEnd(4);
    builder.advanceMatureCursor(4);
    ring.write(new Float32Array([5, 6, 7, 8, 9, 10, 11, 12]));

    const next = builder.buildWindow();
    expect(next?.isInitial).toBe(false);
    expect(next?.startFrame).toBe(4);
    expect(next?.endFrame).toBe(12);
    expect(next?.durationSeconds).toBeCloseTo(2, 5);
  });

  it('commits mature sentences while keeping the current tail as preview text', () => {
    const merger = new UtteranceTranscriptMerger();
    const result: TranscriptResult = {
      text: 'Hello world. This is still running',
      warnings: [],
      meta: {
        detailLevel: 'words',
        isFinal: false,
      },
      words: [
        { index: 0, text: 'Hello', startTime: 0, endTime: 0.4 },
        { index: 1, text: 'world.', startTime: 0.4, endTime: 0.8 },
        { index: 2, text: 'This', startTime: 0.8, endTime: 1.1 },
        { index: 3, text: 'is', startTime: 1.1, endTime: 1.3 },
        { index: 4, text: 'still', startTime: 1.3, endTime: 1.6 },
        { index: 5, text: 'running', startTime: 1.6, endTime: 2.0 },
      ],
    };

    const snapshot = merger.process(result);

    expect(snapshot.committedText).toBe('Hello world.');
    expect(snapshot.previewText).toBe('This is still running');
    expect(snapshot.committedSentences).toHaveLength(1);
    expect(snapshot.matureCursorTime).toBeCloseTo(0.8, 5);
  });

  it('can finalize a punctuation-complete pending sentence on flush', () => {
    const merger = new UtteranceTranscriptMerger();
    merger.process({
      text: 'A complete sentence.',
      warnings: [],
      meta: {
        detailLevel: 'words',
        isFinal: false,
      },
      words: [
        { index: 0, text: 'A', startTime: 0, endTime: 0.1 },
        { index: 1, text: 'complete', startTime: 0.1, endTime: 0.5 },
        { index: 2, text: 'sentence.', startTime: 0.5, endTime: 0.9 },
      ],
    });

    const finalized = merger.finalizePendingIfComplete();

    expect(finalized?.committedText).toBe('A complete sentence.');
    expect(finalized?.previewText).toBe('');
    expect(finalized?.committedSentences).toHaveLength(1);
  });
});
