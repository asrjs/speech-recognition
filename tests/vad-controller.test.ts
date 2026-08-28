import {
  RealtimeTranscriptionController,
  VoiceActivityProbabilityBuffer,
  VoiceActivityTimeline,
} from '@asrjs/speech-recognition/realtime';
import { type TranscriptResult } from '@asrjs/speech-recognition';
import { describe, expect, it, vi } from 'vitest';

describe('voice activity timeline', () => {
  it('tracks speech ranges, silence tails, and boundary search', () => {
    const timeline = new VoiceActivityTimeline({
      sampleRate: 4,
      maxDurationSeconds: 8,
      speechThreshold: 0.5,
    });

    timeline.appendChunk(0, 4, 0.95, true);
    timeline.appendChunk(4, 2, 0.1, false);
    timeline.appendChunk(6, 2, 0.9, true);
    timeline.appendChunk(8, 4, 0.05, false);

    expect(timeline.hasSpeechInRange(0, 8, 0.5)).toBe(true);
    expect(timeline.findSilenceBoundary(12, 6, 0.5)).toBe(12);
    expect(timeline.getSilenceTailDuration(0.5)).toBe(4);
    expect(timeline.createSnapshot().trailingSilenceSeconds).toBeCloseTo(1, 5);
  });
});

describe('voice activity probability buffer', () => {
  it('writes probabilities at absolute frame positions for aligned visual timelines', () => {
    const buffer = new VoiceActivityProbabilityBuffer({
      sampleRate: 16000,
      maxDurationSeconds: 1,
      hopFrames: 320,
      speechThreshold: 0.5,
    });

    buffer.appendProbabilitiesAtFrame([0.8], 640);

    expect(buffer.getLatestFrame()).toBe(960);
    expect(buffer.getTimeline(0, 1280, 4)).toEqual([
      expect.objectContaining({ startFrame: 0, endFrame: 320, probability: 0 }),
      expect.objectContaining({ startFrame: 320, endFrame: 640, probability: 0 }),
      expect.objectContaining({ startFrame: 640, endFrame: 960, probability: expect.closeTo(0.8) }),
      expect.objectContaining({ startFrame: 960, endFrame: 1280, probability: 0 }),
    ]);
  });

  it('tracks speech ranges, silence tails, and boundary search across hop-sized entries', () => {
    const buffer = new VoiceActivityProbabilityBuffer({
      sampleRate: 4,
      maxDurationSeconds: 8,
      hopFrames: 2,
      speechThreshold: 0.5,
    });

    buffer.appendProbabilities([0.95, 0.1, 0.9, 0.05, 0.02]);

    expect(buffer.getLatestFrame()).toBe(10);
    expect(buffer.hasSpeechInRange(0, 8, 0.5)).toBe(true);
    expect(buffer.findSilenceBoundary(10, 4, 0.5)).toBe(10);
    expect(buffer.getSilenceTailDuration(0.5)).toBe(4);
  });

  it('rejects non-finite hop sizes', () => {
    expect(
      () =>
        new VoiceActivityProbabilityBuffer({
          sampleRate: 16_000,
          maxDurationSeconds: 8,
          hopFrames: Number.NaN,
          speechThreshold: 0.5,
        }),
    ).toThrow('positive finite hopFrames');
  });

  it('drops overwritten history while keeping recent VAD queries stable', () => {
    const buffer = new VoiceActivityProbabilityBuffer({
      sampleRate: 4,
      maxDurationSeconds: 2,
      hopFrames: 1,
      speechThreshold: 0.5,
    });

    buffer.appendProbabilities([0.9, 0.8, 0.1, 0.2, 0.95, 0.1, 0.1, 0.1, 0.1]);

    expect(buffer.getBaseFrame()).toBe(1);
    expect(buffer.hasSpeechInRange(0, 1, 0.5)).toBe(false);
    expect(buffer.hasSpeechInRange(4, 6, 0.5)).toBe(true);
    expect(buffer.getSilenceTailDuration(0.5)).toBe(4);
  });
});

describe('realtime transcription controller', () => {
  it('coordinates ring buffering, windows, VAD, and transcript finalization', async () => {
    const transcribe = vi.fn(
      async (): Promise<TranscriptResult> => ({
        text: 'hello there',
        warnings: [],
        meta: {
          detailLevel: 'words',
          isFinal: false,
        },
        words: [
          { index: 0, text: 'hello', startTime: 0, endTime: 0.5 },
          { index: 1, text: 'there', startTime: 0.5, endTime: 1.0 },
        ],
      }),
    );

    const controller = new RealtimeTranscriptionController({
      sampleRate: 4,
      bufferDurationSeconds: 10,
      finalizeSilenceSeconds: 0.5,
      transcribe,
      window: {
        sampleRate: 4,
        minInitialDurationSec: 1,
        minDurationSec: 1,
        maxDurationSec: 6,
      },
    });

    const partial = await controller.pushAudio(new Float32Array([1, 1, 1, 1]), {
      vadObservation: {
        startFrame: 0,
        endFrame: 4,
        speechProbability: 0.95,
        isSpeech: true,
      },
    });

    expect(partial?.kind).toBe('partial');
    expect(partial?.partial.previewText).toBe('hello there');
    expect(transcribe).toHaveBeenCalledOnce();

    const finalized = await controller.pushAudio(new Float32Array([0, 0, 0, 0]), {
      vadObservation: {
        startFrame: 4,
        endFrame: 8,
        speechProbability: 0.05,
        isSpeech: false,
      },
    });

    expect(finalized?.kind).toBe('final');
    expect(finalized?.trigger).toBe('silence-finalize');
    expect(finalized?.partial.committedText).toBe('hello there');
    expect(finalized?.partial.previewText).toBe('');
    expect(controller.getState().trailingSilenceSeconds).toBeCloseTo(1, 5);
  });

  it('serializes concurrent audio pushes so inference observes capture order', async () => {
    let releaseFirst!: () => void;
    const firstFinished = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    const transcribe = vi.fn(async (): Promise<TranscriptResult> => {
      if (transcribe.mock.calls.length === 1) {
        await firstFinished;
      }
      return {
        text: 'ordered',
        warnings: [],
        meta: { detailLevel: 'text', isFinal: false },
      };
    });
    const controller = new RealtimeTranscriptionController({
      sampleRate: 4,
      transcribe,
      window: { sampleRate: 4, minInitialDurationSec: 1, minDurationSec: 1, maxDurationSec: 6 },
    });

    const first = controller.pushAudio(new Float32Array([1, 1, 1, 1]));
    const second = controller.pushAudio(new Float32Array([1, 1, 1, 1]));
    await vi.waitFor(() => expect(transcribe).toHaveBeenCalledOnce());

    releaseFirst();
    await Promise.all([first, second]);
    expect(transcribe).toHaveBeenCalledTimes(2);
  });

  it('does not publish an in-flight result after reset starts a new session', async () => {
    let release!: () => void;
    const pending = new Promise<void>((resolve) => {
      release = resolve;
    });
    const transcribe = vi.fn(async (): Promise<TranscriptResult> => {
      await pending;
      return { text: 'stale', warnings: [], meta: { detailLevel: 'text', isFinal: false } };
    });
    const controller = new RealtimeTranscriptionController({
      sampleRate: 4,
      transcribe,
      window: { sampleRate: 4, minInitialDurationSec: 1, minDurationSec: 1, maxDurationSec: 6 },
    });

    const inFlight = controller.pushAudio(new Float32Array([1, 1, 1, 1]));
    await vi.waitFor(() => expect(transcribe).toHaveBeenCalledOnce());
    const queued = controller.pushAudio(new Float32Array([1, 1, 1, 1]));
    controller.reset();
    release();

    expect(await inFlight).toBeNull();
    expect(await queued).toBeNull();
    expect(transcribe).toHaveBeenCalledOnce();
    expect(controller.getState().currentFrame).toBe(0);
    expect(controller.getState().snapshot.fullText).toBe('');
  });

  it('aborts an in-flight callback on reset and allows a fresh session', async () => {
    let callCount = 0;
    const transcribe = vi.fn(
      (request: { readonly signal: AbortSignal }): Promise<TranscriptResult> =>
        new Promise<TranscriptResult>((resolve, reject) => {
          callCount += 1;
          if (callCount > 1) {
            resolve({
              text: 'fresh',
              warnings: [],
              meta: { detailLevel: 'text', isFinal: false },
            });
            return;
          }
          if (request.signal.aborted) {
            reject(new Error('transcription aborted'));
            return;
          }
          request.signal.addEventListener('abort', () => reject(new Error('transcription aborted')), {
            once: true,
          });
        }),
    );
    const controller = new RealtimeTranscriptionController({
      sampleRate: 4,
      transcribe,
      window: { sampleRate: 4, minInitialDurationSec: 1, minDurationSec: 1, maxDurationSec: 6 },
    });

    const inFlight = controller.pushAudio(new Float32Array([1, 1, 1, 1]));
    await vi.waitFor(() => expect(transcribe).toHaveBeenCalledOnce());
    const signal = transcribe.mock.calls[0]?.[0]?.signal;
    controller.reset();

    expect(signal?.aborted).toBe(true);
    await expect(inFlight).resolves.toBeNull();

    const fresh = await controller.pushAudio(new Float32Array([1, 1, 1, 1]));
    expect(fresh?.partial.text).toBe('fresh');
    expect(transcribe).toHaveBeenCalledTimes(2);
  });
});
