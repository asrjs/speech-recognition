/**
 * Deterministic tests for realtime latency instrumentation.
 */
import {
  RealtimeLatencyTracker,
  RealtimeTranscriptionController,
} from '@asrjs/speech-recognition/realtime';
import { type TranscriptResult } from '@asrjs/speech-recognition';
import { describe, expect, it, vi } from 'vitest';

function baseRecord(overrides: Partial<Parameters<RealtimeLatencyTracker['noteUpdate']>[0]> = {}) {
  return {
    kind: 'partial' as const,
    trigger: 'push',
    windowStartFrame: 0,
    windowEndFrame: 8,
    revision: 1,
    committedText: '',
    previewText: 'hello',
    ...overrides,
  };
}

describe('RealtimeLatencyTracker', () => {
  it('measures first-partial and end-of-utterance latency against ingest marks', () => {
    let clock = 1000;
    const tracker = new RealtimeLatencyTracker({ sampleRate: 4, now: () => clock });

    // Two seconds of audio (frames 0-8) ingested at t=1000.
    tracker.noteIngest(8);
    clock = 1150;
    tracker.noteUpdate(baseRecord());

    // Speech-end at frame 8 arrived at t=1000; the final lands at t=1300.
    clock = 1300;
    tracker.noteUpdate(
      baseRecord({
        kind: 'final',
        trigger: 'silence-finalize',
        revision: 2,
        committedText: 'hello',
        previewText: '',
        speechEndFrame: 8,
      }),
    );

    const summary = tracker.getSummary();
    expect(summary.totalPartials).toBe(1);
    expect(summary.totalFinals).toBe(1);
    expect(summary.lastFirstPartialLatencyMs).toBe(150);
    expect(summary.lastEndOfUtteranceLatencyMs).toBe(300);
    const utterance = summary.completedUtterances[0]!;
    expect(utterance.updates.map((u) => u.kind)).toEqual(['partial', 'final']);
  });

  it('detects committed-text shrink and stagnant updates', () => {
    let clock = 0;
    const tracker = new RealtimeLatencyTracker({ sampleRate: 4, now: () => clock });
    tracker.noteIngest(8);

    clock = 10;
    tracker.noteUpdate(baseRecord({ revision: 1, committedText: 'one two', previewText: ' x' }));
    clock = 20;
    tracker.noteUpdate(baseRecord({ revision: 2, committedText: 'one', previewText: ' two' }));
    clock = 30;
    tracker.noteUpdate(
      baseRecord({ revision: 3, committedText: 'one two three', previewText: '' }),
    );

    const summary = tracker.getSummary();
    expect(summary.totalCommitShrinkCount).toBe(1);
    expect(summary.totalStagnantUpdateCount).toBe(1);
  });

  it('derives process latency from transcribe marks when not provided', () => {
    let clock = 0;
    const tracker = new RealtimeLatencyTracker({ sampleRate: 4, now: () => clock });
    tracker.noteIngest(8);
    tracker.noteTranscribeStart();
    clock = 25;
    tracker.noteUpdate(baseRecord());

    const summary = tracker.getSummary();
    expect(summary.meanProcessLatencyMs).toBe(25);
    expect(summary.p95ProcessLatencyMs).toBe(25);
  });

  it('reports null emit lag when no ingest mark covers the window end', () => {
    const tracker = new RealtimeLatencyTracker({ sampleRate: 4, now: () => 0 });
    tracker.noteUpdate(baseRecord({ windowEndFrame: 9999 }));

    const summary = tracker.getSummary();
    expect(summary.meanEmitLagMs).toBeNull();
    expect(summary.inProgressUtterance?.updates[0]?.emitLagMs).toBeNull();
  });
});

describe('RealtimeTranscriptionController latency integration', () => {
  function makeTranscript(text: string, isFinal: boolean): TranscriptResult {
    return { text, warnings: [], meta: { detailLevel: 'text', isFinal } };
  }

  it('stays null unless enabled and summarizes updates when enabled', async () => {
    const plain = new RealtimeTranscriptionController({
      sampleRate: 4,
      transcribe: () => makeTranscript('hi', false),
      window: { sampleRate: 4, minInitialDurationSec: 1, minDurationSec: 1, maxDurationSec: 6 },
    });
    await plain.pushAudio(new Float32Array([1, 1, 1, 1]));
    expect(plain.getState().latency).toBeNull();

    let clock = 500;
    const controller = new RealtimeTranscriptionController({
      sampleRate: 4,
      finalizeSilenceSeconds: 0.5,
      latency: { sampleRate: 4, now: () => clock },
      transcribe: () => makeTranscript('hi', false),
      window: { sampleRate: 4, minInitialDurationSec: 1, minDurationSec: 1, maxDurationSec: 6 },
    });

    const partial = await controller.pushAudio(new Float32Array([1, 1, 1, 1]), {
      vadObservation: { startFrame: 0, endFrame: 4, speechProbability: 0.95, isSpeech: true },
    });
    clock += 120;
    expect(partial?.kind).toBe('partial');

    const finalized = await controller.pushAudio(new Float32Array([0, 0, 0, 0]), {
      vadObservation: { startFrame: 4, endFrame: 8, speechProbability: 0.05, isSpeech: false },
    });
    expect(finalized?.kind).toBe('final');

    const latency = controller.getState().latency!;
    expect(latency.totalUpdates).toBe(2);
    expect(latency.totalPartials).toBe(1);
    expect(latency.totalFinals).toBe(1);
    expect(latency.lastFirstPartialLatencyMs).toBeTypeOf('number');
    expect(latency.completedUtterances).toHaveLength(1);
    expect(latency.meanEmitLagMs).not.toBeNull();
  });

  it('clears in-progress state on reset while keeping session totals', async () => {
    let clock = 0;
    const transcribe = vi.fn(() => makeTranscript('hi', false));
    const controller = new RealtimeTranscriptionController({
      sampleRate: 4,
      latency: { sampleRate: 4, now: () => clock },
      transcribe,
      window: { sampleRate: 4, minInitialDurationSec: 1, minDurationSec: 1, maxDurationSec: 6 },
    });

    await controller.pushAudio(new Float32Array([1, 1, 1, 1]));
    expect(controller.getState().latency?.inProgressUtterance).not.toBeNull();

    controller.reset();
    const after = controller.getState().latency!;
    expect(after.inProgressUtterance).toBeNull();
    expect(after.totalUpdates).toBe(1);

    clock = 5000;
    await controller.pushAudio(new Float32Array([1, 1, 1, 1]));
    const restarted = controller.getState().latency!;
    expect(restarted.totalUpdates).toBe(2);
    expect(restarted.inProgressUtterance?.updates[0]?.emitLagMs).toBeTypeOf('number');
  });
});
