import { describe, expect, it, vi } from 'vitest';
import { createBrowserRealtimeMicrophoneController } from '../src/runtime/browser-controller.js';
import type { BrowserRealtimeStarterOptions } from '../src/runtime/browser-realtime.js';

function setup(model: any, transcribeGate?: Promise<void>, onTranscript = vi.fn(), transcribeOptions?: any, transcribeCallback?: any) {
  let capture!: (chunk: any) => void;
  let starterOptions!: BrowserRealtimeStarterOptions;
  let nextContext: any;
  const requestAbort = new AbortController();
  const resetController = vi.fn();
  const starter = {
    detector: {}, tenVad: {}, vadBuffer: {},
    controller: {
      noteIngest: vi.fn(),
      async transcribeUtterance(pcm: Float32Array, args: any) {
        const canonical = await starterOptions.transcribe!({
          pcm, signal: requestAbort.signal, sampleRate: args.sampleRate,
          startFrame: args.startFrame, endFrame: args.endFrame,
          startTimeSeconds: 0, endTimeSeconds: pcm.length / args.sampleRate,
          durationSeconds: pcm.length / args.sampleRate, isInitialWindow: false,
          matureCursorFrame: args.endFrame, matureCursorTimeSeconds: 0,
          reason: 'finalize', segmentReason: args.reason, context: args.context ?? nextContext,
        });
        await transcribeGate;
        if (!canonical) return null;
        return { kind: 'final', trigger: 'finalize', canonical, partial: { text: canonical.text },
          snapshot: {}, window: {}, activity: null, context: args.context ?? nextContext } as any;
      },
    },
    subscribe() { return () => undefined; },
    async start() {}, processChunk: vi.fn(), flush: vi.fn(),
    async stop() { return null; }, updateConfig: vi.fn(), getSnapshot() { return { latency: null }; },
    async dispose() { requestAbort.abort(); resetController(); },
  } as any;
  const controller = createBrowserRealtimeMicrophoneController({
    model,
    transcribe: transcribeCallback,
    onTranscript,
    transcribeOptions,
    micMode: 'manual',
    createStarter(options) { starterOptions = options!; return starter; },
    createMonitor: () => ({ subscribe: () => () => undefined, getSnapshot: () => null, flush: vi.fn(), dispose: vi.fn() }) as any,
    startCapture: async (options) => {
      capture = options.onChunk;
      return { sampleRate: 16000, deviceSampleRate: 16000, contextSampleRate: 16000,
        chunkFrames: 4, chunkDurationMs: 0.25, stream: { getAudioTracks: () => [] }, stop: async () => undefined } as any;
    },
  });
  return { controller, capture: () => capture, starter, resetController, getStarterOptions: () => starterOptions,
    setContext: (context: any) => { nextContext = context; } };
}

function loadedModel(transcribeMonoPcm = vi.fn(async () => ({ text: 'heard words', warnings: [], meta: {} }))) {
  return {
    model: { info: { inference: { sampleRate: 8000 } } },
    transcribeMonoPcm,
    dispose: vi.fn(),
  } as any;
}

describe('browser microphone loaded-model binding', () => {
  it('resamples once for the model and publishes one transcript update', async () => {
    const inference = vi.fn(async (pcm: Float32Array) => ({ text: `samples:${pcm.length}`, warnings: [], meta: {} }));
    const model = loadedModel(inference);
    const h = setup(model);
    // The callback can be used by the single realtime transcription controller path.
    const result = await h.getStarterOptions().transcribe!({
      pcm: new Float32Array(160), signal: new AbortController().signal, sampleRate: 16000,
      startFrame: 0, endFrame: 160, startTimeSeconds: 0, endTimeSeconds: .01,
      durationSeconds: .01, isInitialWindow: false, matureCursorFrame: 160,
      matureCursorTimeSeconds: .01, reason: 'finalize',
    } as any);
    expect(inference).toHaveBeenCalledOnce();
    expect(inference.mock.calls[0]![0]).toHaveLength(80);
    expect(result.text).toBe('samples:80');
    await h.controller.dispose();
    expect(h.controller.getState().transcriptText).toBeUndefined();
  });

  it('replaces transcript text when a context revision advances', async () => {
    const inference = vi.fn()
      .mockResolvedValueOnce({ text: 'first hypothesis', warnings: [], meta: {} })
      .mockResolvedValueOnce({ text: 'corrected hypothesis', warnings: [], meta: {} });
    const onTranscript = vi.fn();
    const h = setup(loadedModel(inference), undefined, onTranscript);
    const context = { id: 'turn-a', revision: 1, startFrame: 0, endFrame: 160, sampleRate: 16000 };
    await h.controller.start();
    h.setContext(context);
    h.capture()({ pcm: new Float32Array(160), sampleRate: 16000 });
    h.controller.flush();
    await vi.waitFor(() => expect(onTranscript).toHaveBeenCalledTimes(1));
    h.setContext({ ...context, revision: 2 });
    h.capture()({ pcm: new Float32Array(160), sampleRate: 16000 });
    h.controller.flush();
    await vi.waitFor(() => expect(onTranscript).toHaveBeenCalledTimes(2));
    expect(h.controller.getState().transcriptText).toBe('corrected hypothesis');
    await h.controller.dispose();
  });

  it('delivers updates from an explicit transcribe callback too', async () => {
    const onTranscript = vi.fn();
    const transcribe = vi.fn(async () => ({ text: 'custom path', warnings: [], meta: {} }));
    const h = setup(undefined, undefined, onTranscript, undefined, transcribe);
    await h.controller.start();
    h.capture()({ pcm: new Float32Array(160), sampleRate: 16000 });
    h.controller.flush();
    await vi.waitFor(() => expect(onTranscript).toHaveBeenCalledOnce());
    expect(transcribe).toHaveBeenCalledOnce();
    await h.controller.dispose();
  });

  it('rejects duplicate inference paths', () => {
    expect(() => createBrowserRealtimeMicrophoneController({
      model: loadedModel(), transcribeMonoPcm: async () => undefined,
    })).toThrow(/cannot be combined/);
    expect(() => createBrowserRealtimeMicrophoneController({
      model: loadedModel(), streamingTranscriber: {} as any,
    })).toThrow(/cannot be combined/);
  });

  it('stop waits for pending microphone inference and dispose resets the realtime controller first', async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const h = setup(loadedModel(), gate);
    await h.controller.start();
    h.capture()({ pcm: new Float32Array(160), sampleRate: 16000 });
    h.controller.flush();
    let stopped = false;
    const stopping = h.controller.stop().then(() => { stopped = true; });
    await Promise.resolve();
    expect(stopped).toBe(false);
    release();
    await stopping;
    await h.controller.dispose();
    expect(h.resetController).toHaveBeenCalledOnce();
  });

  it('dispose aborts and drains inference without publishing a late transcript', async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const onTranscript = vi.fn();
    const h = setup(loadedModel(), gate, onTranscript);
    await h.controller.start();
    h.capture()({ pcm: new Float32Array(160), sampleRate: 16000 });
    h.controller.flush();
    let disposed = false;
    const disposing = h.controller.dispose().then(() => { disposed = true; });
    await Promise.resolve();
    expect(disposed).toBe(false);
    release();
    await disposing;
    expect(disposed).toBe(true);
    expect(onTranscript).not.toHaveBeenCalled();
    expect(h.resetController).toHaveBeenCalledOnce();
  });

  it('combines caller and realtime abort signals', async () => {
    const userAbort = { aborted: false };
    const requestAbort = new AbortController();
    let finishInference!: () => void;
    const inferenceGate = new Promise<void>((resolve) => { finishInference = resolve; });
    let receivedSignal!: AbortSignal;
    const inference = vi.fn(async (_pcm: Float32Array, _rate: number, options: any) => {
      receivedSignal = options.signal;
      await inferenceGate;
      return { text: 'ok', warnings: [], meta: {} };
    });
    const h = setup(loadedModel(inference), undefined, undefined, { signal: userAbort });
    const transcribing = h.getStarterOptions().transcribe!({
      pcm: new Float32Array(160), signal: requestAbort.signal, sampleRate: 16000,
      startFrame: 0, endFrame: 160, startTimeSeconds: 0, endTimeSeconds: .01,
      durationSeconds: .01, isInitialWindow: false, matureCursorFrame: 160,
      matureCursorTimeSeconds: .01, reason: 'finalize',
    } as any);
    expect(receivedSignal.aborted).toBe(false);
    userAbort.aborted = true;
    await vi.waitFor(() => expect(receivedSignal.aborted).toBe(true));
    finishInference();
    await transcribing;
    await h.controller.dispose();
  });

  it('surfaces inference failures in state and drains them before disposal', async () => {
    const model = loadedModel(vi.fn(async () => { throw new Error('inference failed'); }));
    const h = setup(model);
    await h.controller.start();
    h.capture()({ pcm: new Float32Array(160), sampleRate: 16000 });
    h.controller.flush();
    await expect(h.controller.stop()).rejects.toThrow('inference failed');
    expect(h.controller.getState().micError).toBe('inference failed');
    await h.controller.dispose();
    expect(model.dispose).not.toHaveBeenCalled();
  });
});
