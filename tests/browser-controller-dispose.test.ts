import { describe, expect, it, vi } from 'vitest';
import {
  createBrowserRealtimeMicrophoneController,
  type BrowserRealtimeMicrophoneControllerOptions,
  type BrowserRealtimeMonitor,
  type BrowserRealtimeStarter,
} from '@asrjs/speech-recognition/browser';
import { RealtimeTranscriptionController } from '@asrjs/speech-recognition/realtime';

type StarterListener = Parameters<BrowserRealtimeStarter['subscribe']>[0];

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
  let listener!: StarterListener;
  const starter: BrowserRealtimeStarter = {
    detector: {} as never,
    tenVad: {} as never,
    vadBuffer: {} as never,
    controller: null,
    subscribe: (next) => {
      listener = next;
      return unsubscribe;
    },
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
  return {
    starter,
    monitor,
    handle,
    stopCapture,
    unsubscribe,
    emit: (event: Parameters<StarterListener>[0]) => listener(event),
  };
}

function segment(readPcm = vi.fn(() => Float32Array.of(0.25))) {
  return {
    type: 'segment-ready' as const,
    payload: {
      startFrame: 0,
      endFrame: 1,
      sampleRate: 16000,
      reason: 'stop',
      readPcm,
      metadata: {
        profileId: 'test',
        rough: null,
        filter: null,
        tenVad: {
          state: 'ready' as const,
          error: null,
          probability: 1,
          speaking: true,
          threshold: 0.5,
        },
        logicalStartFrame: 0,
        logicalEndFrame: 1,
        source: 'ten-vad' as const,
      },
    },
  };
}

describe('browser controller stop after capture failure', () => {
  it('does not publish manual audio when disposal overtakes a failing stop', async () => {
    const r = resources();
    const failure = new Error('late capture shutdown failure');
    const entered = deferred<void>();
    let rejectStop!: (error: Error) => void;
    r.handle.stop = vi.fn(() => {
      entered.resolve();
      return new Promise<void>((_, reject) => {
        rejectStop = reject;
      });
    });
    let captureOptions!: CaptureOptions;
    const onUtterance = vi.fn();
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'manual',
      createStarter: () => r.starter,
      createMonitor: () => r.monitor,
      startCapture: async (options) => {
        captureOptions = options;
        return r.handle;
      },
      onUtterance,
    });
    await controller.start();
    captureOptions.onChunk({
      pcm: Float32Array.of(0.25),
      sampleRate: 16000,
      deviceSampleRate: 48000,
      contextSampleRate: 48000,
      startFrame: 0,
      endFrame: 1,
      startTimeSeconds: 0,
      endTimeSeconds: 1 / 16000,
    });
    const stopping = controller.stop({ flush: true }).catch((error: unknown) => error);
    await entered.promise;
    const disposing = controller.dispose();
    rejectStop(failure);
    expect(await stopping).toBe(failure);
    await disposing;
    expect(onUtterance).not.toHaveBeenCalled();
    expect(r.unsubscribe).toHaveBeenCalledOnce();
    expect(r.monitor.dispose).toHaveBeenCalledOnce();
    expect(r.starter.dispose).toHaveBeenCalledOnce();
  });

  it('reports all stop failures after attempting monitor finalization', async () => {
    const r = resources();
    const failures = [new Error('capture'), new Error('starter'), new Error('monitor')];
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'speech-detect',
      createStarter: () => r.starter,
      createMonitor: () => r.monitor,
      startCapture: async () => r.handle,
    });
    await controller.start();
    r.stopCapture.mockRejectedValue(failures[0]);
    vi.mocked(r.starter.stop).mockRejectedValue(failures[1]);
    vi.mocked(r.monitor.flush)
      .mockClear()
      .mockImplementation(() => {
        throw failures[2];
      });
    try {
      const failure = await controller.stop({ flush: false }).catch((error: unknown) => error);
      expect(failure).toBeInstanceOf(AggregateError);
      expect((failure as AggregateError).errors).toEqual(failures);
      expect(r.starter.stop).toHaveBeenCalledWith({ flush: false });
      expect(r.monitor.flush).toHaveBeenCalledOnce();
      expect(controller.getState().captureInfo.inputSampleRate).toBeNull();
      expect(controller.getState().micStatus).toBe('Microphone stopped');
    } finally {
      await controller.dispose();
    }
  });

  it.each([
    ['manual', true],
    ['manual', false],
    ['speech-detect', true],
    ['speech-detect', false],
  ] as const)('preserves %s flush=%s and reports capture failure', async (micMode, flush) => {
    const r = resources();
    const failure = new Error('capture shutdown failed');
    r.stopCapture.mockRejectedValue(failure);
    let captureOptions!: CaptureOptions;
    const onUtterance = vi.fn();
    r.starter.stop = vi.fn(async (options) => {
      if (options?.flush) r.emit(segment());
      return null;
    });
    const controller = createBrowserRealtimeMicrophoneController({
      micMode,
      createStarter: () => r.starter,
      createMonitor: () => r.monitor,
      startCapture: async (options) => {
        captureOptions = options;
        return r.handle;
      },
      onUtterance,
    });
    await controller.start();
    captureOptions.onChunk({
      pcm: Float32Array.of(0.25),
      sampleRate: 16000,
      deviceSampleRate: 48000,
      contextSampleRate: 48000,
      startFrame: 0,
      endFrame: 1,
      startTimeSeconds: 0,
      endTimeSeconds: 1 / 16000,
    });
    vi.mocked(r.monitor.flush).mockClear();
    try {
      await expect(controller.stop({ flush })).rejects.toBe(failure);
      expect(controller.getState().isMicActive).toBe(false);
      expect(controller.getState().captureInfo).toMatchObject({
        inputSampleRate: null,
        contextSampleRate: null,
        deviceLabel: '',
      });
      expect(controller.getState().micStatus).toBe('Microphone stopped');
      expect(r.stopCapture).toHaveBeenCalledOnce();
      expect(r.monitor.flush).toHaveBeenCalledOnce();
      if (micMode === 'speech-detect') {
        expect(r.starter.stop).toHaveBeenCalledWith({ flush });
      } else {
        expect(r.starter.stop).not.toHaveBeenCalled();
        controller.flush();
      }
      expect(onUtterance).toHaveBeenCalledTimes(flush ? 1 : 0);
      if (flush) expect(onUtterance.mock.calls[0][0].pcm).toEqual(Float32Array.of(0.25));
    } finally {
      await controller.dispose();
    }
  });
});

describe('browser controller disposal during startup', () => {
  it('attempts every release and preserves multiple disposal failures', async () => {
    const r = resources();
    const failures = ['capture', 'unsubscribe', 'monitor', 'starter'].map(
      (stage) => new Error(`${stage} cleanup failed`),
    );
    r.stopCapture.mockRejectedValue(failures[0]);
    r.unsubscribe.mockImplementation(() => {
      throw failures[1];
    });
    vi.mocked(r.monitor.dispose).mockImplementation(() => {
      throw failures[2];
    });
    vi.mocked(r.starter.dispose).mockRejectedValue(failures[3]);
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'manual',
      createStarter: () => r.starter,
      createMonitor: () => r.monitor,
      startCapture: async () => r.handle,
    });
    await controller.start();
    const failure = await controller.dispose().catch((error: unknown) => error);
    expect(failure).toBeInstanceOf(AggregateError);
    expect((failure as AggregateError).errors).toEqual(failures);
    expect(r.stopCapture).toHaveBeenCalledOnce();
    expect(r.unsubscribe).toHaveBeenCalledOnce();
    expect(r.monitor.dispose).toHaveBeenCalledOnce();
    expect(r.starter.dispose).toHaveBeenCalledOnce();
    await controller.dispose();
    expect(r.starter.dispose).toHaveBeenCalledOnce();
  });

  it('releases controller resources even when capture stop rejects', async () => {
    const r = resources();
    const failure = new Error('capture close failed');
    r.stopCapture.mockRejectedValue(failure);
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'manual',
      createStarter: () => r.starter,
      createMonitor: () => r.monitor,
      startCapture: async () => r.handle,
    });
    await controller.start();
    await expect(controller.dispose()).rejects.toBe(failure);
    expect(controller.getState().isMicActive).toBe(false);
    expect(r.stopCapture).toHaveBeenCalledOnce();
    expect(r.unsubscribe).toHaveBeenCalledOnce();
    expect(r.monitor.dispose).toHaveBeenCalledOnce();
    expect(r.starter.dispose).toHaveBeenCalledOnce();
    await controller.dispose();
    expect(r.stopCapture).toHaveBeenCalledOnce();
    expect(r.starter.dispose).toHaveBeenCalledOnce();
  });

  it.each(['startup', 'capture-stop'] as const)(
    'ignores retained starter events during pending %s disposal',
    async (phase) => {
      const r = resources();
      const entered = deferred<void>();
      const gate = deferred<void>();
      if (phase === 'startup') {
        r.starter.start = vi.fn(async () => {
          entered.resolve();
          await gate.promise;
        });
      } else {
        r.handle.stop = vi.fn(async () => {
          entered.resolve();
          await gate.promise;
        });
      }
      const transcription = new RealtimeTranscriptionController({
        sampleRate: 16000,
        transcribe: () => {
          throw new Error('unexpected transcription');
        },
      });
      const transcribe = vi.spyOn(transcription, 'transcribeUtterance');
      const onUtterance = vi.fn();
      const controller = createBrowserRealtimeMicrophoneController({
        micMode: 'speech-detect',
        createStarter: () => ({ ...r.starter, controller: transcription }),
        createMonitor: () => r.monitor,
        startCapture: async () => r.handle,
        onUtterance,
      });
      const starting = controller.start();
      if (phase === 'startup') await entered.promise;
      else await starting;
      const disposing = controller.dispose();
      if (phase === 'capture-stop') await entered.promise;
      const state = controller.getState();
      const event = segment();
      const sendLateEvents = () => {
        r.emit(event);
        r.emit({ type: 'speech-start', payload: { startFrame: 0, sampleRate: 16000 } });
        r.emit({
          type: 'speech-update',
          payload: { startFrame: 0, endFrame: 1, sampleRate: 16000 },
        });
        r.emit({ type: 'speech-end', payload: { startFrame: 0, endFrame: 1, sampleRate: 16000 } });
        r.emit({ type: 'error', payload: new Error('late starter event') });
      };
      try {
        sendLateEvents();
        expect(controller.getState()).toEqual(state);
        expect(event.payload.readPcm).not.toHaveBeenCalled();
        expect(onUtterance).not.toHaveBeenCalled();
        expect(transcribe).not.toHaveBeenCalled();
      } finally {
        gate.resolve();
        await Promise.all([starting, disposing]);
      }
      const disposedState = controller.getState();
      sendLateEvents();
      expect(controller.getState()).toEqual(disposedState);
      expect(onUtterance).not.toHaveBeenCalled();
      expect(transcribe).not.toHaveBeenCalled();
    },
  );

  it('still publishes the final segment during a normal flushed stop', async () => {
    const r = resources();
    const event = segment();
    r.starter.stop = vi.fn(async () => {
      r.emit(event);
      return null;
    });
    const onUtterance = vi.fn();
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'speech-detect',
      createStarter: () => r.starter,
      createMonitor: () => r.monitor,
      startCapture: async () => r.handle,
      onUtterance,
    });
    await controller.start();
    r.emit({ type: 'speech-start', payload: { startFrame: 0, sampleRate: 16000 } });
    expect(controller.getState().micStatus).toContain('Speech detected');
    await controller.stop({ flush: true });
    expect(onUtterance).toHaveBeenCalledOnce();
    expect(event.payload.readPcm).toHaveBeenCalledOnce();
    expect(r.starter.stop).toHaveBeenCalledWith({ flush: true });
    await controller.dispose();
  });

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
