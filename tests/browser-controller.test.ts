import { describe, expect, it, vi } from 'vitest';
import { normalizePcmInput } from '../src/audio/audio.js';
import {
  createBrowserRealtimeMicrophoneController,
  type BrowserRealtimeMonitor,
  type BrowserRealtimeStarter,
} from '@asrjs/speech-recognition/browser';

function createFakeStarter(): BrowserRealtimeStarter {
  const listeners = new Set<(event: any) => void>();
  return {
    detector: {} as never,
    tenVad: {} as never,
    vadBuffer: {} as never,
    controller: null,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    async start() {
      listeners.forEach((listener) =>
        listener({
          type: 'metrics',
          payload: {
            tenVad: { state: 'ready' },
          },
        }),
      );
    },
    processChunk: vi.fn(),
    flush: vi.fn(),
    stop: vi.fn(async () => null),
    updateConfig: vi.fn(),
    getSnapshot() {
      return {
        tenVad: { state: 'ready' },
        latency: null,
      } as never;
    },
    dispose: vi.fn(async () => undefined),
  };
}

function createFakeMonitor(): BrowserRealtimeMonitor {
  return {
    subscribe() {
      return () => undefined;
    },
    getSnapshot() {
      return null;
    },
    flush: vi.fn(),
    dispose: vi.fn(),
  };
}

describe('browser realtime microphone controller', () => {
  it('publishes the final streaming text and immediately exposes boundary cleanup failures', async () => {
    let capture!: (chunk: any) => void;
    const onPartial = vi.fn();
    const stream = {
      pushAudio: vi.fn(async () => null),
      finalize: vi.fn(async () => ({ text: 'last word', kind: 'final' })),
      reset: vi.fn(async () => undefined),
    };
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'manual', streamingTranscriber: stream as never, onPartialTranscript: onPartial,
      createStarter: createFakeStarter, createMonitor: createFakeMonitor,
      startCapture: async (options) => {
        capture = options.onChunk;
        return { sampleRate: 16000, deviceSampleRate: 16000, contextSampleRate: 16000,
          chunkFrames: 256, chunkDurationMs: 16, stream: { getAudioTracks: () => [] },
          stop: async () => undefined } as never;
      },
    });
    await controller.start();
    stream.reset.mockRejectedValueOnce(new Error('stream reset failed'));
    capture({ pcm: new Float32Array(256), sampleRate: 16000, startFrame: 0, endFrame: 256 });
    controller.flush();
    await vi.waitFor(() => expect(controller.getState().micError).toBe('stream reset failed'));
    expect(onPartial).toHaveBeenCalledWith({ text: 'last word', kind: 'final' });
    await expect(controller.stop({ flush: false })).rejects.toThrow('stream reset failed');
    await controller.dispose().catch(() => undefined);
  });

  it('replays a delayed onset once and queues the split tail before finalize/reset', async () => {
    const listeners = new Set<(event: any) => void>();
    const emit = (type: string, payload: unknown) => listeners.forEach(listener => listener({ type, payload }));
    const starter = createFakeStarter();
    starter.subscribe = (listener) => { listeners.add(listener); return () => listeners.delete(listener); };
    let chunkIndex = 0;
    starter.processChunk = () => {
      chunkIndex++;
      if (chunkIndex === 2) emit('speech-start', { startFrame: 0, sampleRate: 16000 });
      if (chunkIndex === 3 || chunkIndex === 5) {
        const startFrame = chunkIndex === 3 ? 0 : 8;
        const endFrame = chunkIndex * 4;
        emit('speech-end', { startFrame, endFrame, sampleRate: 16000 });
        emit('segment-ready', { startFrame, endFrame, sampleRate: 16000, reason: 'max-duration',
          readPcm: () => Float32Array.from({ length: endFrame - startFrame }, (_, i) => i + startFrame) });
        if (chunkIndex === 3) emit('speech-start', { startFrame: 8, sampleRate: 16000 });
      }
    };
    let samples: number[] = [];
    const finalized: number[][] = [];
    const stream = {
      pushAudio: vi.fn(async (input) => { samples.push(...normalizePcmInput(input).channels[0]!); return null; }),
      finalize: vi.fn(async () => { finalized.push([...samples]); return { text: 'fixture' }; }),
      reset: vi.fn(() => { samples = []; }), flush: vi.fn(), dispose: vi.fn(),
    };
    let feed!: (chunk: any) => void;
    const controller = createBrowserRealtimeMicrophoneController({
      createStarter: () => starter, createMonitor: createFakeMonitor, streamingTranscriber: stream as never,
      startCapture: async (options) => {
        feed = options.onChunk;
        return { sampleRate: 16000, deviceSampleRate: null, contextSampleRate: 16000,
          chunkFrames: 4, chunkDurationMs: 0.25, stream: {} as MediaStream, stop: async () => undefined };
      },
    });
    await controller.start();
    for (let index = 0; index < 5; index++) {
      const pcm = Float32Array.from({ length: 4 }, (_, i) => index * 4 + i);
      feed({ pcm, sampleRate: 16000, startFrame: index * 4, endFrame: index * 4 + 4 });
      pcm.fill(-1); // Capture buffer reuse must not corrupt queued audio.
    }
    await controller.stop({ flush: false });
    expect(finalized).toEqual([
      Array.from({ length: 12 }, (_, i) => i),
      Array.from({ length: 12 }, (_, i) => i + 8),
    ]);
    expect(stream.finalize).toHaveBeenCalledTimes(2);
    expect(controller.getState().micError).toBe('');
    await controller.dispose();
  });
  function pendingStart(config: { ringBufferDurationMs?: number } = {}) {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const starter = createFakeStarter();
    starter.start = vi.fn(async () => {
      await gate;
    });
    let onChunk!: (chunk: any) => void;
    const stopCapture = vi.fn(async () => undefined);
    const earlyPcm = Float32Array.of(0.25, 0.5);
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'speech-detect',
      config,
      createStarter: () => starter,
      createMonitor: createFakeMonitor,
      startCapture: async (options) => {
        onChunk = options.onChunk;
        onChunk({ pcm: earlyPcm, sampleRate: 16000, startFrame: 0, endFrame: 2 });
        return {
          sampleRate: 16000,
          deviceSampleRate: null,
          contextSampleRate: 16000,
          chunkFrames: 256,
          chunkDurationMs: 16,
          stream: {} as MediaStream,
          stop: stopCapture,
        };
      },
    });
    return {
      controller,
      starter,
      stopCapture,
      earlyPcm,
      release,
      feed: (pcm: Float32Array, startFrame = 2) =>
        onChunk({ pcm, sampleRate: 16000, startFrame, endFrame: startFrame + pcm.length }),
    };
  }

  it('buffers owned PCM from acquisition through delayed model readiness in order', async () => {
    const r = pendingStart();
    const starting = r.controller.start();
    await vi.waitFor(() => expect(r.starter.start).toHaveBeenCalledOnce());
    const next = Float32Array.of(0.75);
    r.feed(next);
    r.earlyPcm.fill(9);
    next.fill(9);
    expect(r.starter.processChunk).not.toHaveBeenCalled();
    r.release();
    await starting;
    expect(r.starter.processChunk).toHaveBeenNthCalledWith(1, Float32Array.of(0.25, 0.5), {
      startFrame: 0,
      endFrame: 2,
    });
    expect(r.starter.processChunk).toHaveBeenNthCalledWith(2, Float32Array.of(0.75), {
      startFrame: 2,
      endFrame: 3,
    });
    r.feed(Float32Array.of(1), 3);
    expect(r.starter.processChunk).toHaveBeenCalledTimes(3);
    await r.controller.dispose();
  });

  it('does not replay queued startup PCM if stop invalidates delayed initialization', async () => {
    const r = pendingStart();
    const starting = r.controller.start();
    await vi.waitFor(() => expect(r.starter.start).toHaveBeenCalledOnce());
    r.feed(Float32Array.of(0.75));
    const stopping = r.controller.stop({ flush: false });
    r.release();
    await Promise.all([starting, stopping]);
    expect(r.starter.processChunk).not.toHaveBeenCalled();
    expect(r.stopCapture).toHaveBeenCalledOnce();
    await r.controller.dispose();
  });

  it('fails instead of silently dropping PCM when the bounded startup buffer is exceeded', async () => {
    const r = pendingStart({ ringBufferDurationMs: 16 });
    const starting = r.controller.start();
    const failure = expect(starting).rejects.toThrow(/startup PCM.*ring-buffer duration/);
    await vi.waitFor(() => expect(r.starter.start).toHaveBeenCalledOnce());
    r.feed(new Float32Array(256));
    r.release();
    await failure;
    expect(r.starter.processChunk).not.toHaveBeenCalled();
    expect(r.stopCapture).toHaveBeenCalledOnce();
    expect(r.controller.getState().isMicActive).toBe(false);
    await r.controller.dispose();
  });

  it('discards owned startup PCM when disposal overtakes initialization', async () => {
    const r = pendingStart();
    const starting = r.controller.start();
    await vi.waitFor(() => expect(r.starter.start).toHaveBeenCalledOnce());
    r.feed(Float32Array.of(0.75));
    const disposing = r.controller.dispose();
    r.release();
    await Promise.all([starting, disposing]);
    expect(r.starter.processChunk).not.toHaveBeenCalled();
    expect(r.stopCapture).toHaveBeenCalledOnce();
  });

  it('releases capture without replay when initialization rejects', async () => {
    const r = pendingStart();
    const failure = new Error('model initialization failed');
    const originalStart = r.starter.start;
    r.starter.start = vi.fn(async () => {
      await originalStart();
      throw failure;
    });
    const starting = r.controller.start();
    const rejected = expect(starting).rejects.toBe(failure);
    await vi.waitFor(() => expect(r.starter.start).toHaveBeenCalledOnce());
    r.feed(Float32Array.of(0.75));
    r.release();
    await rejected;
    expect(r.starter.processChunk).not.toHaveBeenCalled();
    expect(r.stopCapture).toHaveBeenCalledOnce();
    expect(r.controller.getState().isMicActive).toBe(false);
    await r.controller.dispose();
  });

  it('captures manual microphone audio and emits a flushed utterance', async () => {
    const utterances: Array<{ pcm: Float32Array; sampleRate: number; reason: string }> = [];
    let onChunk: ((chunk: any) => void) | null = null;
    const stop = vi.fn(async () => undefined);

    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'manual',
      createStarter: createFakeStarter,
      createMonitor: createFakeMonitor,
      startCapture: async (options) => {
        onChunk = options.onChunk;
        return {
          sampleRate: 16000,
          deviceSampleRate: 48000,
          contextSampleRate: 48000,
          chunkFrames: 256,
          chunkDurationMs: 16,
          stream: {
            getAudioTracks() {
              return [{ label: 'Fake mic' }] as MediaStreamTrack[];
            },
          } as MediaStream,
          stop,
        };
      },
      onUtterance(utterance) {
        utterances.push({
          pcm: utterance.pcm,
          sampleRate: utterance.sampleRate,
          reason: utterance.reason,
        });
      },
    });

    await controller.start();
    expect(controller.getState().isMicActive).toBe(true);
    expect(controller.getState().latency).toBeNull();
    expect(controller.getState().captureInfo.deviceLabel).toBe('Fake mic');

    onChunk?.({
      pcm: new Float32Array([0.25, 0.5]),
      sampleRate: 16000,
      startFrame: 0,
      endFrame: 2,
    });
    onChunk?.({
      pcm: new Float32Array([0.75]),
      sampleRate: 16000,
      startFrame: 2,
      endFrame: 3,
    });

    controller.flush('manual');

    expect(utterances).toHaveLength(1);
    expect(Array.from(utterances[0]!.pcm)).toEqual([0.25, 0.5, 0.75]);
    expect(utterances[0]!.sampleRate).toBe(16000);
    expect(utterances[0]!.reason).toBe('manual');

    await controller.stop();
    expect(stop).toHaveBeenCalled();
    expect(controller.getState().isMicActive).toBe(false);
  });

  it('routes speech-detect chunks through the realtime starter', async () => {
    const starter = createFakeStarter();
    let onChunk: ((chunk: any) => void) | null = null;

    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'speech-detect',
      createStarter: () => starter,
      createMonitor: createFakeMonitor,
      startCapture: async (options) => {
        onChunk = options.onChunk;
        return {
          sampleRate: 16000,
          deviceSampleRate: 48000,
          contextSampleRate: 48000,
          chunkFrames: 256,
          chunkDurationMs: 16,
          stream: {
            getAudioTracks() {
              return [] as MediaStreamTrack[];
            },
          } as MediaStream,
          stop: async () => undefined,
        };
      },
    });

    await controller.start();
    onChunk?.({
      pcm: new Float32Array([1, 2, 3]),
      sampleRate: 16000,
      startFrame: 0,
      endFrame: 3,
    });

    expect(starter.processChunk).toHaveBeenCalledWith(
      expect.any(Float32Array),
      expect.objectContaining({ startFrame: 0, endFrame: 3 }),
    );

    await controller.dispose();
  });

  it('sends cumulative original PCM and stable revision metadata across overlapping VAD segments', async () => {
    const listeners = new Set<(event: any) => void>();
    const starter = createFakeStarter();
    starter.subscribe = (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    };
    const transcribeUtterance = vi.fn(async () => null);
    starter.controller = { transcribeUtterance, noteIngest: vi.fn() } as never;
    const emit = (type: string, payload: unknown) => listeners.forEach((listener) => listener({ type, payload }));
    const captured: any[] = [];
    let capture!: (chunk: any) => void;
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'speech-detect',
      transcriptionContext: { maxDurationSeconds: 10, maxGapSeconds: 3, prerollSeconds: 0 },
      onUtterance: (utterance) => captured.push(utterance),
      createStarter: () => starter,
      createMonitor: createFakeMonitor,
      startCapture: async (captureOptions) => {
        capture = captureOptions.onChunk;
        return {
          sampleRate: 16000, deviceSampleRate: null, contextSampleRate: 16000,
          chunkFrames: 4, chunkDurationMs: 0.25, stream: {} as MediaStream,
          stop: async () => undefined,
        };
      },
    });
    await controller.start();
    capture({ pcm: Float32Array.of(1, 2, 3, 4), sampleRate: 16000, startFrame: 0, endFrame: 4 });
    emit('segment-ready', {
      startFrame: 1, endFrame: 3, sampleRate: 16000, reason: 'pause', metadata: { logicalStartFrame: 1, logicalEndFrame: 2 },
      readPcm: () => Float32Array.of(2, 3),
    });
    capture({ pcm: Float32Array.of(5, 6, 7, 8), sampleRate: 16000, startFrame: 4, endFrame: 8 });
    emit('segment-ready', {
      startFrame: 2, endFrame: 6, sampleRate: 16000, reason: 'pause', metadata: { logicalStartFrame: 3, logicalEndFrame: 5 },
      readPcm: () => Float32Array.of(3, 4, 5, 6),
    });

    await vi.waitFor(() => expect(transcribeUtterance).toHaveBeenCalledTimes(2));
    const first = transcribeUtterance.mock.calls[0]!;
    const second = transcribeUtterance.mock.calls[1]!;
    expect(Array.from(first[0] as Float32Array)).toEqual([2, 3]);
    expect(Array.from(second[0] as Float32Array)).toEqual([2, 3, 4, 5, 6]);
    expect((second[1] as any).context).toMatchObject({ id: (first[1] as any).context.id, revision: 2, startFrame: 1, endFrame: 6 });
    expect(Array.from(captured[1].pcm)).toEqual([2, 3, 4, 5, 6]);
    expect(captured[1].pcm.length).toBe(captured[1].context.endFrame - captured[1].context.startFrame);
    await controller.dispose();
  });

  it('publishes one finalized context revision with bounded captured tail on flush stop', async () => {
    const listeners = new Set<(event: any) => void>();
    const starter = createFakeStarter();
    starter.subscribe = (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    };
    const transcribeUtterance = vi.fn(async () => null);
    starter.controller = { transcribeUtterance, noteIngest: vi.fn() } as never;
    const emit = (type: string, payload: unknown) => listeners.forEach((listener) => listener({ type, payload }));
    const utterances: any[] = [];
    let capture!: (chunk: any) => void;
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'speech-detect',
      transcriptionContext: { maxDurationSeconds: 1, maxGapSeconds: 3, prerollSeconds: 0.001 },
      onUtterance: (utterance) => utterances.push(utterance),
      createStarter: () => starter,
      createMonitor: createFakeMonitor,
      startCapture: async (captureOptions) => {
        capture = captureOptions.onChunk;
        return {
          sampleRate: 16000, deviceSampleRate: null, contextSampleRate: 16000,
          chunkFrames: 100, chunkDurationMs: 6.25, stream: {} as MediaStream,
          stop: async () => undefined,
        };
      },
    });
    const pcm = Float32Array.from({ length: 100 }, (_, index) => index + 1);
    await controller.start();
    capture({ pcm, sampleRate: 16000, startFrame: 0, endFrame: pcm.length });
    emit('segment-ready', {
      startFrame: 10, endFrame: 50, sampleRate: 16000, reason: 'pause',
      metadata: { logicalStartFrame: 10, logicalEndFrame: 50 },
      readPcm: () => pcm.slice(10, 50),
    });

    await controller.stop({ flush: true });

    expect(utterances).toHaveLength(2);
    expect(utterances[0].context).toMatchObject({ startFrame: 0, endFrame: 50, revision: 1, finalized: false });
    expect(utterances[1].context).toMatchObject({
      id: utterances[0].context.id,
      startFrame: 0,
      endFrame: 66,
      revision: 2,
      finalized: true,
    });
    expect(utterances[1].reason).toBe('stop');
    expect(utterances[1].pcm).toEqual(pcm.slice(0, 66));
    expect(transcribeUtterance).toHaveBeenCalledTimes(2);
  });

  it('does not publish a finalized context revision when stop skips flushing', async () => {
    const listeners = new Set<(event: any) => void>();
    const starter = createFakeStarter();
    starter.subscribe = (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    };
    const utterances: any[] = [];
    let capture!: (chunk: any) => void;
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'speech-detect',
      transcriptionContext: { maxDurationSeconds: 1, maxGapSeconds: 3, prerollSeconds: 0.001 },
      onUtterance: (utterance) => utterances.push(utterance),
      createStarter: () => starter,
      createMonitor: createFakeMonitor,
      startCapture: async (captureOptions) => {
        capture = captureOptions.onChunk;
        return {
          sampleRate: 16000, deviceSampleRate: null, contextSampleRate: 16000,
          chunkFrames: 100, chunkDurationMs: 6.25, stream: {} as MediaStream,
          stop: async () => undefined,
        };
      },
    });
    const pcm = Float32Array.from({ length: 100 }, (_, index) => index + 1);
    await controller.start();
    capture({ pcm, sampleRate: 16000, startFrame: 0, endFrame: pcm.length });
    listeners.forEach((listener) => listener({
      type: 'segment-ready',
      payload: {
        startFrame: 10, endFrame: 50, sampleRate: 16000, reason: 'pause',
        metadata: { logicalStartFrame: 10, logicalEndFrame: 50 },
        readPcm: () => pcm.slice(10, 50),
      },
    }));

    await controller.stop({ flush: false });

    expect(utterances).toHaveLength(1);
    expect(utterances[0].context).toMatchObject({ finalized: false });
  });

  it('does not emit a batch context final revision for native VAD streaming', async () => {
    const listeners = new Set<(event: any) => void>();
    const starter = createFakeStarter();
    starter.subscribe = (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    };
    const streamingTranscriber = {
      pushAudio: vi.fn(async () => null),
      finalize: vi.fn(async () => ({ text: 'stream result' })),
      reset: vi.fn(async () => undefined),
    };
    const utterances: any[] = [];
    let capture!: (chunk: any) => void;
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'speech-detect',
      streamingTranscriber: streamingTranscriber as never,
      transcriptionContext: { maxDurationSeconds: 1, maxGapSeconds: 3, prerollSeconds: 0.001 },
      onUtterance: (utterance) => utterances.push(utterance),
      createStarter: () => starter,
      createMonitor: createFakeMonitor,
      startCapture: async (captureOptions) => {
        capture = captureOptions.onChunk;
        return {
          sampleRate: 16000, deviceSampleRate: null, contextSampleRate: 16000,
          chunkFrames: 100, chunkDurationMs: 6.25, stream: {} as MediaStream,
          stop: async () => undefined,
        };
      },
    });
    const pcm = Float32Array.from({ length: 100 }, (_, index) => index + 1);
    await controller.start();
    capture({ pcm, sampleRate: 16000, startFrame: 0, endFrame: pcm.length });
    listeners.forEach((listener) => listener({
      type: 'segment-ready',
      payload: {
        startFrame: 10, endFrame: 50, sampleRate: 16000, reason: 'pause',
        metadata: { logicalStartFrame: 10, logicalEndFrame: 50 },
        readPcm: () => pcm.slice(10, 50),
      },
    }));

    await controller.stop({ flush: true });

    expect(utterances).toHaveLength(1);
    expect(utterances[0].context).toMatchObject({ finalized: false });
    expect(streamingTranscriber.finalize).not.toHaveBeenCalled();
  });

  it('does not emit a finalized context after the detector stop rejects', async () => {
    const listeners = new Set<(event: any) => void>();
    const starter = createFakeStarter();
    starter.subscribe = (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    };
    const stopStarter = vi.fn(async () => null);
    starter.stop = stopStarter;
    const utterances: any[] = [];
    let capture!: (chunk: any) => void;
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'speech-detect',
      transcriptionContext: { maxDurationSeconds: 1, maxGapSeconds: 3, prerollSeconds: 0.001 },
      onUtterance: (utterance) => utterances.push(utterance),
      createStarter: () => starter,
      createMonitor: createFakeMonitor,
      startCapture: async (captureOptions) => {
        capture = captureOptions.onChunk;
        return {
          sampleRate: 16000, deviceSampleRate: null, contextSampleRate: 16000,
          chunkFrames: 100, chunkDurationMs: 6.25, stream: {} as MediaStream,
          stop: async () => undefined,
        };
      },
    });
    const pcm = Float32Array.from({ length: 100 }, (_, index) => index + 1);
    await controller.start();
    capture({ pcm, sampleRate: 16000, startFrame: 0, endFrame: pcm.length });
    listeners.forEach((listener) => listener({
      type: 'segment-ready',
      payload: {
        startFrame: 10, endFrame: 50, sampleRate: 16000, reason: 'pause',
        metadata: { logicalStartFrame: 10, logicalEndFrame: 50 },
        readPcm: () => pcm.slice(10, 50),
      },
    }));
    stopStarter.mockRejectedValueOnce(new Error('detector stop failed'));

    await expect(controller.stop({ flush: true })).rejects.toThrow('detector stop failed');

    expect(utterances).toHaveLength(1);
    expect(utterances[0].context).toMatchObject({ finalized: false });
  });

  it('keeps captured audio before delayed VAD onset by default and caps each context at its duration limit', async () => {
    const listeners = new Set<(event: any) => void>();
    const starter = createFakeStarter();
    starter.subscribe = (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    };
    const transcribeUtterance = vi.fn(async () => null);
    starter.controller = { transcribeUtterance, noteIngest: vi.fn() } as never;
    const emit = (type: string, payload: unknown) => listeners.forEach((listener) => listener({ type, payload }));
    const captured: any[] = [];
    let capture!: (chunk: any) => void;
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'speech-detect',
      transcriptionContext: { maxDurationSeconds: 1, maxGapSeconds: 3 },
      onUtterance: (utterance) => captured.push(utterance),
      createStarter: () => starter,
      createMonitor: createFakeMonitor,
      startCapture: async (captureOptions) => {
        capture = captureOptions.onChunk;
        return {
          sampleRate: 16000, deviceSampleRate: null, contextSampleRate: 16000,
          chunkFrames: 256, chunkDurationMs: 16, stream: {} as MediaStream,
          stop: async () => undefined,
        };
      },
    });
    const pcm = Float32Array.from({ length: 16001 }, (_, frame) => frame / 20000);
    await controller.start();
    capture({ pcm, sampleRate: 16000, startFrame: 0, endFrame: pcm.length });

    // A 600ms extracted delay requires the 600ms default to retain frame zero;
    // the previous 300ms default would discard the first 4800 captured frames.
    emit('segment-ready', {
      startFrame: 9600, endFrame: 12000, sampleRate: 16000, reason: 'pause',
      metadata: { logicalStartFrame: 9600, logicalEndFrame: 12000 },
      readPcm: () => pcm.slice(9600, 12000),
    });
    emit('segment-ready', {
      startFrame: 11000, endFrame: 16000, sampleRate: 16000, reason: 'pause',
      metadata: { logicalStartFrame: 12000, logicalEndFrame: 15500 },
      readPcm: () => pcm.slice(11000, 16000),
    });
    emit('segment-ready', {
      startFrame: 16000, endFrame: 16001, sampleRate: 16000, reason: 'pause',
      metadata: { logicalStartFrame: 16000, logicalEndFrame: 16001 },
      readPcm: () => pcm.slice(16000, 16001),
    });

    await vi.waitFor(() => expect(transcribeUtterance).toHaveBeenCalledTimes(3));
    expect(captured[0].context).toMatchObject({ startFrame: 0, endFrame: 12000 });
    expect(captured[0].pcm).toEqual(pcm.slice(0, 12000));
    expect(captured[1].context).toMatchObject({ id: captured[0].context.id, revision: 2, startFrame: 0, endFrame: 16000 });
    expect(captured[2].context).toMatchObject({ startFrame: 16000, endFrame: 16001 });
    for (const utterance of captured) {
      expect(utterance.pcm.length).toBeLessThanOrEqual(16000);
      expect(utterance.pcm.length).toBe(utterance.context.endFrame - utterance.context.startFrame);
    }
    await controller.dispose();
  });

  it('stops a capture that resolves after stop invalidates start', async () => {
    let resolveCapture!: (handle: any) => void;
    const captureReady = new Promise<any>((resolve) => {
      resolveCapture = resolve;
    });
    const startCapture = vi.fn(() => captureReady);
    const stopCapture = vi.fn(async () => undefined);
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'manual',
      createStarter: createFakeStarter,
      createMonitor: createFakeMonitor,
      startCapture,
    });

    const starting = controller.start();
    await vi.waitFor(() => expect(startCapture).toHaveBeenCalledOnce());
    const stopping = controller.stop({ flush: false });
    resolveCapture({
      sampleRate: 16000,
      deviceSampleRate: 48000,
      contextSampleRate: 48000,
      stream: {} as MediaStream,
      stop: stopCapture,
    });

    await Promise.all([starting, stopping]);
    expect(stopCapture).toHaveBeenCalledOnce();
    expect(controller.getState().isMicActive).toBe(false);
    await controller.dispose();
  });

  it('aborts speech-detect starter start when stop runs during VAD init', async () => {
    let startSignal: { readonly aborted: boolean } | undefined;
    let releaseStart!: () => void;
    const startGate = new Promise<void>((resolve) => {
      releaseStart = resolve;
    });
    const starter = {
      ...createFakeStarter(),
      async start(options?: { readonly signal?: { readonly aborted: boolean } | null }) {
        startSignal = options?.signal ?? undefined;
        await startGate;
        if (startSignal?.aborted) {
          const error = new Error('Asset load aborted during "streaming-vad-init".');
          error.name = 'AssetLoadAbortedError';
          (error as Error & { code: string }).code = 'asset-load-aborted';
          throw error;
        }
      },
    };
    const stopCapture = vi.fn(async () => undefined);
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'speech-detect',
      createStarter: () => starter,
      createMonitor: createFakeMonitor,
      startCapture: async () => ({
        sampleRate: 16000,
        deviceSampleRate: 48000,
        contextSampleRate: 48000,
        stream: {} as MediaStream,
        stop: stopCapture,
      }),
    });

    const starting = controller.start();
    await vi.waitFor(() => expect(startSignal).toBeDefined());
    const stopping = controller.stop({ flush: false });
    releaseStart();

    await Promise.all([starting, stopping]);
    expect(startSignal?.aborted).toBe(true);
    expect(starter.stop).toHaveBeenCalled();
    expect(controller.getState().isMicActive).toBe(false);
    await controller.dispose();
  });

  it('ignores chunks from a capture callback after stop', async () => {
    let onChunk: ((chunk: any) => void) | null = null;
    const utterances: unknown[] = [];
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'manual',
      createStarter: createFakeStarter,
      createMonitor: createFakeMonitor,
      startCapture: async (options) => {
        onChunk = options.onChunk;
        return {
          sampleRate: 16000,
          deviceSampleRate: 48000,
          contextSampleRate: 48000,
          stream: {} as MediaStream,
          stop: async () => undefined,
        };
      },
      onUtterance(utterance) {
        utterances.push(utterance);
      },
    });

    await controller.start();
    await controller.stop({ flush: false });
    onChunk?.({ pcm: new Float32Array([1, 2, 3]), sampleRate: 16000 });
    controller.flush('late-callback');

    expect(utterances).toHaveLength(0);
    await controller.dispose();
  });

  it('feeds VAD segments through transcribeUtterance and refreshes latency', async () => {
    const transcribeUtterance = vi.fn(async () => ({ kind: 'final' }));
    const listeners = new Set<(event: any) => void>();
    const latency = {
      lastFirstPartialLatencyMs: 42,
      lastEndOfUtteranceLatencyMs: 40,
      p50ProcessLatencyMs: 38,
      p95EmitLagMs: 41,
    };
    const starter = {
      ...createFakeStarter(),
      controller: {
        transcribeUtterance,
        noteIngest: vi.fn(),
      },
      subscribe(listener: (event: any) => void) {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
      getSnapshot() {
        return {
          tenVad: { state: 'ready' },
          latency,
        } as never;
      },
    };
    let onChunk: ((chunk: any) => void) | null = null;
    const utterances: Array<{ reason: string }> = [];
    const states: Array<{ latency: { lastFirstPartialLatencyMs?: number } | null }> = [];

    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'speech-detect',
      createStarter: () => starter as never,
      createMonitor: createFakeMonitor,
      startCapture: async (options) => {
        onChunk = options.onChunk;
        return {
          sampleRate: 16000,
          deviceSampleRate: 48000,
          contextSampleRate: 48000,
          chunkFrames: 256,
          chunkDurationMs: 16,
          stream: {
            getAudioTracks() {
              return [] as MediaStreamTrack[];
            },
          } as MediaStream,
          stop: async () => undefined,
        };
      },
      onUtterance(utterance) {
        utterances.push({ reason: utterance.reason });
      },
    });
    controller.subscribe((state) => {
      states.push({ latency: state.latency });
    });

    await controller.start();
    onChunk?.({
      pcm: new Float32Array(256).fill(0.2),
      sampleRate: 16000,
      startFrame: 0,
      endFrame: 256,
    });
    listeners.forEach((listener) =>
      listener({
        type: 'segment-ready',
        payload: {
          startFrame: 0,
          endFrame: 16000,
          sampleRate: 16000,
          reason: 'pause',
          metadata: null,
          readPcm: () => new Float32Array(16000).fill(0.2),
        },
      }),
    );

    await vi.waitFor(() => expect(transcribeUtterance).toHaveBeenCalledOnce());
    expect(transcribeUtterance.mock.calls[0]?.[0]).toHaveLength(16000);
    expect(transcribeUtterance.mock.calls[0]?.[1]).toMatchObject({
      startFrame: 0,
      endFrame: 16000,
      sampleRate: 16000,
      reason: 'pause',
    });
    expect(utterances).toEqual([{ reason: 'pause' }]);
    expect(controller.getState().latency?.lastFirstPartialLatencyMs).toBe(42);
    expect(states.some((entry) => entry.latency?.lastFirstPartialLatencyMs === 42)).toBe(true);

    await controller.dispose();
  });
});
