import { expect, it } from 'vitest';
import {
  RealtimeTranscriptionController,
  type TranscriptResult,
} from '@asrjs/speech-recognition/realtime';

it('preserves PCM, committed text and all canonical time coordinates across rolling origins', async () => {
  const sampleRate = 4;
  const pcm = Float32Array.from({ length: 136 }, (_, index) => index / 136);
  const originals: TranscriptResult[] = [];
  const starts: number[] = [];
  const controller = new RealtimeTranscriptionController({
    sampleRate, bufferDurationSeconds: 8, finalizeSilenceSeconds: 0,
    window: { minInitialDurationSec: 2, minDurationSec: 2, maxDurationSec: 6,
      useActivityBoundaries: false },
    transcribe(request) {
      expect(request.pcm).toEqual(pcm.slice(request.startFrame, request.endFrame));
      expect(request.startTimeSeconds).toBe(request.startFrame / sampleRate);
      expect(request.endTimeSeconds).toBe(request.endFrame / sampleRate);
      expect(request.durationSeconds).toBeLessThanOrEqual(6);
      starts.push(request.startFrame);
      const label = `call-${originals.length}.`;
      const result: TranscriptResult = {
        text: `${label} pending`, warnings: [], meta: { detailLevel: 'detailed', isFinal: false },
        words: [{ index: 0, text: label, startTime: 0.125, endTime: 0.75 },
          { index: 1, text: 'pending', startTime: 0.75, endTime: 1.25 }],
        segments: [{ index: 0, text: label, startTime: 0.125, endTime: 0.75, wordIndices: [0] }],
        sentences: [{ index: 0, text: label, startTime: 0.125, endTime: 0.75, wordIndices: [0] }],
        tokens: [{ index: 0, text: label, id: 99, startTime: 0.125, endTime: 0.75 },
          { index: 1, text: 'untimed', id: 100 }],
      };
      originals.push(result);
      return result;
    },
  });
  function verify(update: Awaited<ReturnType<typeof controller.pushAudio>>) {
    if (!update) return;
    const original = originals.at(-1)!;
    const offset = update.window.startFrame / sampleRate;
    for (const field of ['words', 'segments', 'sentences'] as const) {
      expect(update.canonical[field]).toEqual(original[field]!.map(item => ({ ...item,
        startTime: item.startTime + offset, endTime: item.endTime + offset })));
      expect(update.partial[field]).toEqual(update.canonical[field]);
    }
    expect(update.canonical.tokens?.[0]).toEqual({ ...original.tokens![0],
      startTime: 0.125 + offset, endTime: 0.75 + offset });
    expect(update.canonical.tokens?.[1].startTime).toBeUndefined();
    expect(update.canonical.tokens?.[1].endTime).toBeUndefined();
    // Input results must remain relative, including repeated callback reuse.
    expect(original.sentences?.[0].startTime).toBe(0.125);
    expect(original.words?.[0].endTime).toBe(0.75);
  }
  try {
    for (let start = 0; start < pcm.length; start += 8) {
      verify(await controller.pushAudio(pcm.subarray(start, start + 8)));
    }
    expect(starts.some(start => start > 0)).toBe(true);
    expect(controller.getState().baseFrame).toBeGreaterThan(0);
    const final = await controller.finalize();
    verify(final);
    for (let index = 0; index < originals.length; index += 1) {
      expect(final?.snapshot.committedText).toContain(`call-${index}.`);
    }
    expect(controller.getState().isFinalized).toBe(true);
  } finally {
    controller.reset();
    expect(controller.getState().currentFrame).toBe(0);
    expect(controller.getState().snapshot.fullText).toBe('');
  }
});
