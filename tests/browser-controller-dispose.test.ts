import { describe, expect, it, vi } from 'vitest';
import {
  createBrowserRealtimeMicrophoneController,
  type BrowserRealtimeMicrophoneControllerOptions,
  type BrowserRealtimeMonitor,
  type BrowserRealtimeStarter,
} from '@asrjs/speech-recognition/browser';

type CaptureOptions = Parameters<
  NonNullable<BrowserRealtimeMicrophoneControllerOptions['startCapture']>
>[0];
type CaptureHandle = Awaited<
  ReturnType<NonNullable<BrowserRealtimeMicrophoneControllerOptions['startCapture']>>
>;

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function resources() {
  const unsubscribe = vi.fn();
  const starter: BrowserRealtimeStarter = {
    detector: {} as never,
    tenVad: {} as never,
    vadBuffer: {} as never,
    controller: null,
    subscribe: () => unsubscribe,
    start: vi.fn(async () => undefined),
    processChunk: vi.fn(),
    flush: vi.fn(),
    stop: vi.fn(async () => null),
    updateConfig: vi.fn(),
    getSnapshot: () => ({ latency: null }) as never,
    dispose: vi.fn(async () => undefined),
  };
  const monitor: BrowserRealtimeMonitor = {
    subscribe: () => () => undefined,
    getSnapshot: () => null,
    flush: vi.fn(),
    dispose: vi.fn(),
  };
  const stopCapture = vi.fn(async () => undefined);
  const handle: CaptureHandle = {
    sampleRate: 16000,
    deviceSampleRate: 48000,
    contextSampleRate: 48000,
    chunkFrames: 256,
    chunkDurationMs: 16,
    stream: { getAudioTracks: () => [] } as unknown as MediaStream,
    stop: stopCapture,
  };
  return { starter, monitor, handle, stopCapture, unsubscribe };
}

describe('browser controller disposal during startup', () => {
  it('stops a late microphone handle once and ignores its callbacks', async () => {
    const r = resources();
    const gate = deferred<CaptureHandle>();
    const entered = deferred<CaptureOptions>();
    const onUtterance = vi.fn();
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'manual',
      createStarter: () => r.starter,
      createMonitor: () => r.monitor,
      startCapture: (options) => {
        entered.resolve(options);
        return gate.promise;
      },
      onUtterance,
    });
    const starting = controller.start();
    const captureOptions = await entered.promise;
    const disposing = controller.dispose();
    expect(controller.getState().isMicActive).toBe(false);
    gate.resolve(r.handle);
    await Promise.all([starting, disposing]);
    const state = controller.getState();
    captureOptions.onChunk({
      pcm: Float32Array.of(1),
      sampleRate: 16000,
      deviceSampleRate: 48000,
      contextSampleRate: 48000,
      startFrame: 0,
      endFrame: 1,
      startTimeSeconds: 0,
      endTimeSeconds: 1 / 16000,
    });
    captureOptions.onError?.(new Error('late microphone error'));
    controller.flush();
    expect(controller.getState()).toEqual(state);
    expect(onUtterance).not.toHaveBeenCalled();
    expect(r.starter.start).not.toHaveBeenCalled();
    expect(r.stopCapture).toHaveBeenCalledOnce();
    expect(r.unsubscribe).toHaveBeenCalledOnce();
    expect(r.monitor.dispose).toHaveBeenCalledOnce();
    expect(r.starter.dispose).toHaveBeenCalledOnce();
    await controller.dispose();
    await expect(controller.start()).rejects.toThrow('disposed');
    expect(r.stopCapture).toHaveBeenCalledOnce();
    expect(r.starter.dispose).toHaveBeenCalledOnce();
  });

  it.each(['resolve', 'reject'] as const)(
    'cleans pending VAD startup that later %ss',
    async (outcome) => {
      const r = resources();
      const gate = deferred<void>();
      const entered = deferred<NonNullable<Parameters<BrowserRealtimeStarter['start']>[0]>>();
      r.starter.start = vi.fn(async (options) => {
        entered.resolve(options!);
        await gate.promise;
        if (outcome === 'reject') throw new Error('aborted VAD initialization');
      });
      const controller = createBrowserRealtimeMicrophoneController({
        micMode: 'speech-detect',
        createStarter: () => r.starter,
        createMonitor: () => r.monitor,
        startCapture: async () => r.handle,
      });
      const starting = controller.start();
      const options = await entered.promise;
      const disposing = controller.dispose();
      expect(options.signal?.aborted).toBe(true);
      expect(controller.getState().isMicActive).toBe(false);
      gate.resolve();
      await Promise.all([starting, disposing]);
      expect(r.starter.stop).toHaveBeenCalledOnce();
      expect(r.starter.stop).toHaveBeenCalledWith({ flush: false });
      expect(r.stopCapture).toHaveBeenCalledOnce();
      expect(r.unsubscribe).toHaveBeenCalledOnce();
      expect(r.monitor.dispose).toHaveBeenCalledOnce();
      expect(r.starter.dispose).toHaveBeenCalledOnce();
      expect(controller.getState().isMicActive).toBe(false);
      await expect(controller.start()).rejects.toThrow('disposed');
    },
  );
});
