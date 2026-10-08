import { expect, it, vi } from 'vitest';
import { DefaultStreamingTranscriber } from '@asrjs/speech-recognition/inference';
import type { SpeechSession, TranscriptResult } from '@asrjs/speech-recognition';

const result: TranscriptResult = {
  text: 'simulation',
  warnings: [],
  meta: { detailLevel: 'text', isFinal: true },
};

function fixture() {
  const session: SpeechSession = {
    transcribe: vi.fn(async () => result),
    dispose: vi.fn(),
  };
  return { session, transcriber: new DefaultStreamingTranscriber(session) };
}

it('all callers await the same pending session release', async () => {
  const { session, transcriber } = fixture();
  const started = Promise.withResolvers<void>();
  const gate = Promise.withResolvers<void>();
  vi.mocked(session.dispose).mockImplementation(() => {
    started.resolve();
    return gate.promise;
  });
  const first = transcriber.dispose();
  await started.promise;
  let secondSettled = false;
  const second = transcriber.dispose().then(() => {
    secondSettled = true;
  });
  await Promise.resolve();
  expect(secondSettled).toBe(false);
  gate.resolve();
  await Promise.all([first, second]);
  await transcriber.dispose();
  expect(session.dispose).toHaveBeenCalledOnce();
});

it.each([new Error('session release'), undefined, null])(
  'retains release failure %s for concurrent and later callers without retry',
  async (failure) => {
    const { session, transcriber } = fixture();
    const started = Promise.withResolvers<void>();
    const gate = Promise.withResolvers<void>();
    vi.mocked(session.dispose).mockImplementation(() => {
      started.resolve();
      return gate.promise;
    });
    // Attach rejection observers before releasing the rejecting gate.
    const first = transcriber.dispose();
    const firstCheck = expect(first).rejects.toBe(failure);
    await started.promise;
    const secondCheck = expect(transcriber.dispose()).rejects.toBe(failure);
    gate.reject(failure);
    await Promise.all([firstCheck, secondCheck]);
    await expect(transcriber.dispose()).rejects.toBe(failure);
    expect(session.dispose).toHaveBeenCalledOnce();
  },
);

it('retains a synchronous release throw without retry', async () => {
  const { session, transcriber } = fixture();
  const failure = new Error('sync release');
  vi.mocked(session.dispose).mockImplementation(() => {
    throw failure;
  });
  await expect(transcriber.dispose()).rejects.toBe(failure);
  await expect(transcriber.dispose()).rejects.toBe(failure);
  expect(session.dispose).toHaveBeenCalledOnce();
});

it('drains active/queued work and rejects further use while every disposal caller waits', async () => {
  const { session, transcriber } = fixture();
  const started = Promise.withResolvers<void>();
  const inference = Promise.withResolvers<TranscriptResult>();
  let signal: AbortSignal | undefined;
  vi.mocked(session.transcribe).mockImplementation((_input, options) => {
    signal = options?.signal;
    started.resolve();
    return inference.promise;
  });
  const active = transcriber.pushAudio(new Float32Array(8000));
  await started.promise;
  const queued = expect(transcriber.flush()).rejects.toThrow('disposed');
  const first = transcriber.dispose();
  let secondSettled = false;
  const second = transcriber.dispose().then(() => {
    secondSettled = true;
  });
  expect(signal?.aborted).toBe(true);
  await Promise.resolve();
  expect(session.dispose).not.toHaveBeenCalled();
  expect(secondSettled).toBe(false);
  await expect(transcriber.reset()).rejects.toThrow('disposed');
  inference.resolve(result);
  expect((await active).text).toBe('');
  await queued;
  await Promise.all([first, second]);
  expect(session.transcribe).toHaveBeenCalledOnce();
  expect(session.dispose).toHaveBeenCalledOnce();
  await expect(transcriber.pushAudio(new Float32Array(8000))).rejects.toThrow('disposed');
  await expect(transcriber.finalize()).rejects.toThrow('disposed');
});

it('abort-listener reentrant disposal also waits for owned release', async () => {
  const { session, transcriber } = fixture();
  const started = Promise.withResolvers<void>();
  const inference = Promise.withResolvers<TranscriptResult>();
  const released = Promise.withResolvers<void>();
  let reentrant: Promise<void> | undefined;
  vi.mocked(session.transcribe).mockImplementation((_input, options) => {
    options?.signal?.addEventListener('abort', () => {
      reentrant = transcriber.dispose();
    });
    started.resolve();
    return inference.promise;
  });
  vi.mocked(session.dispose).mockImplementation(() => released.promise);
  const active = transcriber.pushAudio(new Float32Array(8000));
  await started.promise;
  const first = transcriber.dispose();
  expect(reentrant).toBeDefined();
  let settled = false;
  const observer = reentrant!.then(() => {
    settled = true;
  });
  await Promise.resolve();
  expect(settled).toBe(false);
  inference.resolve(result);
  await active;
  released.resolve();
  await Promise.all([first, observer]);
  expect(session.dispose).toHaveBeenCalledOnce();
});
