import { describe, expect, it } from 'vitest';

import { PcmAudioBuffer } from '../src/audio/index.js';
import { LongAudioCoordinator } from '../src/inference/streaming/long-audio.js';
import type {
  BaseTranscriptionOptions,
  SpeechSession,
  TranscriptResult,
} from '../src/types/index.js';

function makeTranscript(text: string, durationSeconds: number): TranscriptResult {
  return {
    text,
    warnings: [],
    meta: {
      detailLevel: 'text',
      isFinal: true,
      durationSeconds,
    },
  };
}

describe('long-audio coordinator', () => {
  it('chunks long audio and forwards time offsets to the session', async () => {
    const calls: Array<{ duration: number; timeOffsetSeconds?: number }> = [];

    const session: SpeechSession<BaseTranscriptionOptions, never> = {
      async transcribe(input, options) {
        const audio = input as PcmAudioBuffer;
        calls.push({
          duration: audio.durationSeconds,
          timeOffsetSeconds: options?.timeOffsetSeconds,
        });
        return makeTranscript(`chunk@${options?.timeOffsetSeconds ?? 0}`, audio.durationSeconds);
      },
      dispose() {
        return undefined;
      },
    };

    const coordinator = new LongAudioCoordinator(session);
    const audio = PcmAudioBuffer.fromMono(new Float32Array(10), 4);

    const result = await coordinator.transcribe(audio, {
      chunkLengthSeconds: 1,
      overlapSeconds: 0.25,
    });

    expect(calls).toEqual([
      { duration: 1, timeOffsetSeconds: 0 },
      { duration: 1, timeOffsetSeconds: 0.75 },
      { duration: 1, timeOffsetSeconds: 1.5 },
    ]);
    expect(result.text).toBe('chunk@0 chunk@0.75 chunk@1.5');
    expect(result.meta.durationSeconds).toBeCloseTo(3);
  });

  it('falls back to a single transcription when chunking is not needed', async () => {
    const calls: Array<number | undefined> = [];

    const session: SpeechSession<BaseTranscriptionOptions, never> = {
      async transcribe(input, options) {
        const audio = input as PcmAudioBuffer;
        calls.push(options?.timeOffsetSeconds);
        return makeTranscript('single-pass', audio.durationSeconds);
      },
      dispose() {
        return undefined;
      },
    };

    const coordinator = new LongAudioCoordinator(session);
    const audio = PcmAudioBuffer.fromMono(new Float32Array(4), 4);

    const result = await coordinator.transcribe(audio, {
      chunkLengthSeconds: 2,
    });

    expect(calls).toEqual([undefined]);
    expect(result.text).toBe('single-pass');
  });
});
