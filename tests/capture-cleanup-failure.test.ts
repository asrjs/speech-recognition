import {
  startMicrophoneCapture,
  type BrowserAudioContextLike,
} from '@asrjs/speech-recognition/browser';
import { describe, expect, it, vi } from 'vitest';

function resources() {
  const processor = { onaudioprocess: null, connect: vi.fn(), disconnect: vi.fn() };
  const source = { connect: vi.fn(), disconnect: vi.fn() };
  const tracks = [{ stop: vi.fn() }, { stop: vi.fn() }];
  const stream = { getTracks: () => tracks } as unknown as MediaStream;
  const context: BrowserAudioContextLike = {
    sampleRate: 16000,
    destination: {},
    createMediaStreamSource: () => source,
    createScriptProcessor: () => processor,
    close: vi.fn(async () => {}),
  };
  const start = () =>
    startMicrophoneCapture({
      getUserMedia: async () => stream,
      createAudioContext: () => context,
      onChunk: vi.fn(),
    });
  return { processor, source, tracks, stream, context, start };
}

describe('microphone cleanup failures', () => {
  it('shares the same rejection between concurrent stops', async () => {
    const r = resources();
    const error = new Error('close');
    let rejectClose!: (error: unknown) => void;
    const gate = new Promise<void>((_, reject) => {
      rejectClose = reject;
    });
    r.context.close = vi.fn(() => gate);
    const handle = await r.start();
    const first = handle.stop().catch((failure: unknown) => failure);
    const second = handle.stop().catch((failure: unknown) => failure);
    rejectClose(error);
    expect(await Promise.all([first, second])).toEqual([error, error]);
    expect(r.context.close).toHaveBeenCalledOnce();
    expect(r.tracks[0]!.stop).toHaveBeenCalledOnce();
    expect(r.tracks[1]!.stop).toHaveBeenCalledOnce();
  });

  it('does not stop borrowed tracks on successful default cleanup', async () => {
    const r = resources();
    const handle = await startMicrophoneCapture({
      stream: r.stream,
      createAudioContext: () => r.context,
      onChunk: vi.fn(),
    });
    await handle.stop();
    expect(r.context.close).toHaveBeenCalledOnce();
    expect(r.tracks[0]!.stop).not.toHaveBeenCalled();
    expect(r.tracks[1]!.stop).not.toHaveBeenCalled();
  });

  it('attempts every release and reports all failures', async () => {
    const r = resources();
    const errors = [
      new Error('processor'),
      new Error('source'),
      new Error('close'),
      new Error('track'),
    ];
    r.processor.disconnect.mockImplementation(() => {
      throw errors[0];
    });
    r.source.disconnect.mockImplementation(() => {
      throw errors[1];
    });
    r.context.close = vi.fn(async () => {
      throw errors[2];
    });
    r.tracks[0]!.stop.mockImplementation(() => {
      throw errors[3];
    });
    const handle = await r.start();
    const failure = await handle.stop().catch((error: unknown) => error);
    expect(r.processor.onaudioprocess).toBeNull();
    expect(r.source.disconnect).toHaveBeenCalledOnce();
    expect(r.context.close).toHaveBeenCalledOnce();
    expect(r.tracks[1]!.stop).toHaveBeenCalledOnce();
    expect(failure).toBeInstanceOf(AggregateError);
    expect((failure as AggregateError).errors).toEqual(errors);
    await expect(handle.stop()).rejects.toBe(failure);
    expect(r.processor.disconnect).toHaveBeenCalledOnce();
  });

  it('waits for the same pending cleanup on concurrent stops', async () => {
    const r = resources();
    let finish!: () => void;
    const gate = new Promise<void>((resolve) => {
      finish = resolve;
    });
    r.context.close = vi.fn(() => gate);
    const handle = await r.start();
    const first = handle.stop();
    let secondFinished = false;
    const second = handle.stop().then(() => {
      secondFinished = true;
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(secondFinished).toBe(false);
    finish();
    await Promise.all([first, second]);
    expect(r.context.close).toHaveBeenCalledOnce();
    expect(r.tracks[0]!.stop).toHaveBeenCalledOnce();
  });

  it('preserves the original single cleanup error and still stops tracks', async () => {
    const r = resources();
    const error = new Error('close');
    r.context.close = vi.fn(async () => {
      throw error;
    });
    const handle = await r.start();
    await expect(handle.stop()).rejects.toBe(error);
    expect(r.tracks[0]!.stop).toHaveBeenCalledOnce();
    expect(r.tracks[1]!.stop).toHaveBeenCalledOnce();
  });

  it('preserves borrowed stream ownership after cleanup failure', async () => {
    const r = resources();
    const error = new Error('close');
    r.context.close = vi.fn(async () => {
      throw error;
    });
    const handle = await startMicrophoneCapture({
      stream: r.stream,
      createAudioContext: () => r.context,
      onChunk: vi.fn(),
    });
    await expect(handle.stop()).rejects.toBe(error);
    expect(r.tracks[0]!.stop).not.toHaveBeenCalled();
    expect(r.tracks[1]!.stop).not.toHaveBeenCalled();
  });

  it('retains startup and cleanup errors without skipping remaining releases', async () => {
    const r = resources();
    const startup = new Error('connect');
    const cleanup = new Error('disconnect');
    r.source.connect.mockImplementation(() => {
      throw startup;
    });
    r.processor.disconnect.mockImplementation(() => {
      throw cleanup;
    });
    const failure = await r.start().catch((error: unknown) => error);
    expect(r.context.close).toHaveBeenCalledOnce();
    expect(r.tracks[1]!.stop).toHaveBeenCalledOnce();
    expect(failure).toBeInstanceOf(AggregateError);
    expect((failure as AggregateError).errors).toEqual([startup, cleanup]);
  });
});
