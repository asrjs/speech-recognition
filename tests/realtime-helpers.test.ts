import {
  AudioRingBuffer,
  RoughSpeechGate,
  StreamingWindowBuilder,
  UtteranceTranscriptMerger,
  type TranscriptResult,
  VoiceActivityProbabilityBuffer,
} from '@asrjs/speech-recognition/realtime';
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

  it('samples real waveform points directly from the visible ring-buffer range', () => {
    const ring = new AudioRingBuffer({
      sampleRate: 8,
      durationSeconds: 2,
    });

    ring.write(new Float32Array([0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25]));
    const waveform = ring.getSamplePoints(5, 8);

    expect(Array.from(waveform.samples)).toEqual([0, 0.5, 1, 0.75, 0.25]);
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

  it('bucketizes VAD probabilities across the visible waveform range', () => {
    const vad = new VoiceActivityProbabilityBuffer({
      sampleRate: 16000,
      maxDurationSeconds: 2,
      hopFrames: 160,
      speechThreshold: 0.5,
    });

    vad.appendProbabilities([0.1, 0.85, 0.9, 0.2]);
    const timeline = vad.getTimeline(0, 640, 4);

    expect(timeline).toHaveLength(4);
    expect(timeline[0]?.probability).toBeCloseTo(0.1, 3);
    expect(timeline[1]?.probability).toBeCloseTo(0.85, 3);
    expect(timeline[2]?.probability).toBeCloseTo(0.9, 3);
    expect(timeline[3]?.probability).toBeCloseTo(0.2, 3);
  });

  it('bucketizes rough energy across the visible waveform range', () => {
    const gate = new RoughSpeechGate({
      sampleRate: 16000,
      analysisWindowMs: 80,
      minSpeechDurationMs: 80,
      minSilenceDurationMs: 160,
      minSpeechLevelDbfs: -45,
      maxHistoryChunks: 8,
    });

    gate.process(new Float32Array(1280).fill(0.001));
    gate.process(new Float32Array(1280).fill(0.05));
    gate.process(new Float32Array(1280).fill(0.05));
    gate.process(new Float32Array(1280).fill(0.001));

    const timeline = gate.getTimeline(0, 5120, 4);

    expect(timeline).toHaveLength(4);
    expect(timeline[1]?.energy).toBeGreaterThan(timeline[0]?.energy ?? 0);
    expect(timeline[2]?.energy).toBeGreaterThan(timeline[3]?.energy ?? 0);
    expect(timeline[1]?.isSpeech).toBe(true);
  });
});
