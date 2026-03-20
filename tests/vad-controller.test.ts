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
    expect(buffer.findSilenceBoundary(10, 4, 0.5)).toBe(8);
    expect(buffer.getSilenceTailDuration(0.5)).toBe(4);
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
});
