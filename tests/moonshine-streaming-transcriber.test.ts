import { describe, expect, it, vi } from 'vitest';
import { MoonshineStreamingTranscriber } from '../src/models/moonshine/streaming-transcriber.js';
import type { MoonshineStreamingSessionLike } from '../src/models/moonshine/streaming-transcriber.js';

function fakeSession() {
  let committed = 0;
  let broken = false;
  const session: MoonshineStreamingSessionLike = {
    get committedEncoderFrames() {
      return committed;
    },
    pushAudio: vi.fn(async (pcm: Float32Array) => {
      committed += Math.floor(pcm.length / 320);
      return committed;
    }),
    preview: vi.fn(async () => ({
      text: broken ? '' : `preview-${committed}`,
      tokenIds: broken ? [] : [7, committed],
      encoderFrames: committed,
    })),
    finalize: vi.fn(async () => ({
      text: 'final text',
      tokenIds: [1, 2, 3],
      encoderFrames: committed,
    })),
    reset: vi.fn(() => {
      committed = 0;
    }),
  };
  return session;
}

const oneSecond = () => new Float32Array(16000).fill(0.1);

describe('MoonshineStreamingTranscriber', () => {
  it('publishes previews per push with canonical meta and growing revisions', async () => {
    const session = fakeSession();
    const transcriber = new MoonshineStreamingTranscriber(session, {
      modelId: 'moonshine-streaming-tiny',
    });
    const first = await transcriber.pushAudio(oneSecond());
    const second = await transcriber.pushAudio(oneSecond());
    expect(first.kind).toBe('partial');
    expect(first.revision).toBeLessThan(second.revision);
    expect(second.previewText).toBe('preview-100');
    expect(second.committedText).toBe('');
    expect(second.meta.modelFamily).toBe('moonshine-streaming');
    expect(second.meta.modelId).toBe('moonshine-streaming-tiny');
    expect(second.meta.isFinal).toBe(false);
    expect(second.meta.sampleRate).toBe(16000);
    expect(second.meta.durationSeconds).toBeCloseTo(2);
    expect(second.meta.tokenCount).toBe(2);
  });

  it('finalize publishes the final transcript as committed and preview text', async () => {
    const session = fakeSession();
    const transcriber = new MoonshineStreamingTranscriber(session, { backendId: 'webgpu' });
    await transcriber.pushAudio(oneSecond());
    const final = await transcriber.finalize();
    expect(final.kind).toBe('final');
    expect(final.text).toBe('final text');
    expect(final.committedText).toBe('final text');
    expect(final.previewText).toBe('final text');
    expect(final.meta.isFinal).toBe(true);
    expect(final.meta.backendId).toBe('webgpu');
    expect(transcriber.getState().isFinalized).toBe(true);
    expect(session.finalize).toHaveBeenCalledTimes(1);
  });

  it('flush re-decodes without finalizing', async () => {
    const session = fakeSession();
    const transcriber = new MoonshineStreamingTranscriber(session);
    await transcriber.pushAudio(oneSecond());
    const flushed = await transcriber.flush();
    expect(flushed.kind).toBe('partial');
    expect(session.preview).toHaveBeenCalledTimes(2); // push + flush
    expect(transcriber.getState().isFinalized).toBe(false);
  });

  it('rejects non-16k audio instead of resampling silently', async () => {
    const session = fakeSession();
    const transcriber = new MoonshineStreamingTranscriber(session);
    await expect(
      transcriber.pushAudio({
        sampleRate: 8000,
        numberOfChannels: 1,
        numberOfFrames: 8000,
        channels: [new Float32Array(8000)],
      } as never),
    ).rejects.toThrow('16000 Hz');
    expect(session.pushAudio).not.toHaveBeenCalled();
  });

  it('blocks pushes after finalize until reset, then streams again', async () => {
    const transcriber = new MoonshineStreamingTranscriber(fakeSession());
    await transcriber.pushAudio(oneSecond());
    await transcriber.finalize();
    await expect(transcriber.pushAudio(oneSecond())).rejects.toThrow('finalized');
    transcriber.reset();
    const state = transcriber.getState();
    expect(state.isFinalized).toBe(false);
    expect(state.bufferedDurationSeconds).toBe(0);
    expect(state.committedText).toBe('');
    await expect(transcriber.pushAudio(oneSecond())).resolves.toMatchObject({ kind: 'partial' });
  });

  it('emitPartials=false skips preview decodes on push', async () => {
    const session = fakeSession();
    const transcriber = new MoonshineStreamingTranscriber(session, { emitPartials: false });
    await transcriber.pushAudio(oneSecond());
    expect(session.preview).not.toHaveBeenCalled();
    await transcriber.flush();
    expect(session.preview).toHaveBeenCalledTimes(1);
  });

  it('dispose is terminal and idempotent', async () => {
    const session = fakeSession();
    const transcriber = new MoonshineStreamingTranscriber(session);
    await transcriber.pushAudio(oneSecond());
    transcriber.dispose();
    transcriber.dispose();
    expect(session.reset).toHaveBeenCalledTimes(1);
    await expect(transcriber.pushAudio(oneSecond())).rejects.toThrow('disposed');
    expect(() => transcriber.reset()).toThrow('disposed');
  });
});
