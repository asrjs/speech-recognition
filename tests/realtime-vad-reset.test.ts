import { describe, expect, it, vi } from 'vitest';
import { RealtimeTranscriptionController } from '@asrjs/speech-recognition/realtime';
import type { TranscriptResult, VoiceActivityEvent } from '@asrjs/speech-recognition';

const speech: VoiceActivityEvent = { isSpeech: true, speechProbability: 0.9 };
const transcript: TranscriptResult = {
  text: 'fresh',
  warnings: [],
  meta: { detailLevel: 'text', isFinal: false },
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

function setup() {
  const pending = deferred<VoiceActivityEvent>();
  const entered = deferred<void>();
  const analyze = vi.fn(() => {
    if (analyze.mock.calls.length === 1) {
      entered.resolve();
      return pending.promise;
    }
    return Promise.resolve(speech);
  });
  const transcribe = vi.fn(async () => transcript);
  const controller = new RealtimeTranscriptionController({
    sampleRate: 4,
    vad: { analyze, reset: vi.fn() },
    transcribe,
    window: { sampleRate: 4, minInitialDurationSec: 1, minDurationSec: 1, maxDurationSec: 6 },
  });
  return { controller, pending, entered, analyze, transcribe };
}

describe('realtime VAD reset boundary', () => {
  it.each([new Error('late detector failure'), undefined, null, 'detector failed'])(
    'ignores obsolete detector rejection after reset: %s',
    async (failure) => {
      const { controller, pending, entered, analyze, transcribe } = setup();
      const old = controller.pushAudio(new Float32Array([1, 1, 1, 1]));
      // Attach immediately so even the RED result is an observed rejection.
      const oldOutcome = old.then(
        (value) => ({ value }),
        (error) => ({ error }),
      );
      await entered.promise;
      const queued = controller.pushAudio(new Float32Array([2, 2, 2, 2]));
      controller.reset();
      const resetState = structuredClone(controller.getState());
      pending.reject(failure);

      expect(await oldOutcome).toEqual({ value: null });
      expect(await queued).toBeNull();
      expect(controller.getState()).toEqual(resetState);
      expect(analyze).toHaveBeenCalledOnce();
      expect(transcribe).not.toHaveBeenCalled();

      const fresh = await controller.pushAudio(new Float32Array([3, 3, 3, 3]));
      expect(fresh?.canonical.text).toBe('fresh');
      expect(controller.getState().currentFrame).toBe(4);
      expect(analyze).toHaveBeenCalledTimes(2);
      expect(transcribe).toHaveBeenCalledOnce();
    },
  );

  it('does not publish obsolete successful VAD after reset', async () => {
    const { controller, pending, entered, transcribe } = setup();
    const old = controller.pushAudio(new Float32Array([1, 1, 1, 1]));
    await entered.promise;
    controller.reset();
    const state = structuredClone(controller.getState());
    pending.resolve(speech);
    expect(await old).toBeNull();
    expect(controller.getState()).toEqual(state);
    expect(transcribe).not.toHaveBeenCalled();
  });

  it.each([new Error('active detector failure'), undefined, null, 'detector failed'])(
    'preserves current-generation detector rejection identity: %s',
    async (failure) => {
      const { controller, pending, entered, transcribe } = setup();
      const active = controller.pushAudio(new Float32Array([1, 1, 1, 1]));
      const outcome = active.then(
        (value) => ({ value }),
        (error) => ({ error }),
      );
      await entered.promise;
      pending.reject(failure);
      expect(await outcome).toEqual({ error: failure });
      expect(transcribe).not.toHaveBeenCalled();
      expect(controller.getState().currentFrame).toBe(4);
      expect((await controller.pushAudio(new Float32Array([2, 2, 2, 2])))?.canonical.text).toBe(
        'fresh',
      );
    },
  );
});
