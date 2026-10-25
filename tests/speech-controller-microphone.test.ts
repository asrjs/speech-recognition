import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createSpeechController } from '../src/runtime/speech-controller.js';
import type { BrowserRealtimeMicrophoneControllerState } from '../src/runtime/browser-controller.js';

const { loadSpeechModel, createMicrophone, decodeAudioSourceToMonoPcm } = vi.hoisted(() => ({
  loadSpeechModel: vi.fn(),
  createMicrophone: vi.fn(),
  decodeAudioSourceToMonoPcm: vi.fn(),
}));
vi.mock('../src/runtime/load.js', () => ({ loadSpeechModel }));
vi.mock('../src/runtime/media.js', () => ({ decodeAudioSourceToMonoPcm }));
vi.mock('../src/runtime/browser-controller.js', () => ({
  createBrowserRealtimeMicrophoneController: createMicrophone,
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

function fakeModel() {
  return {
    model: { info: { modelId: 'mic-model', inference: { sampleRate: 16000 } } },
    supportsStreaming: false,
    transcribeMonoPcm: vi.fn(async () => ({ text: 'done', warnings: [], meta: {} })),
    dispose: vi.fn(async () => undefined),
  };
}

function fakeMicrophone(overrides: Record<string, unknown> = {}) {
  const state: BrowserRealtimeMicrophoneControllerState = {
    isMicActive: false,
    micStatus: 'Microphone idle',
    micError: '',
    micMode: 'speech-detect',
    captureInfo: {
      inputSampleRate: null,
      contextSampleRate: null,
      processingSampleRate: 16000,
      chunkFrames: 512,
      chunkDurationMs: 32,
      deviceLabel: '',
    },
    latency: null,
    transcript: null,
    transcriptText: '',
  };
  const listeners = new Set<(value: BrowserRealtimeMicrophoneControllerState) => void>();
  const mic = {
    getState: vi.fn(() => state),
    subscribe: vi.fn((listener: (value: BrowserRealtimeMicrophoneControllerState) => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    }),
    start: vi.fn(async () => {
      state.isMicActive = true;
      state.micStatus = 'Microphone active';
      for (const listener of listeners) listener(state);
    }),
    stop: vi.fn(async () => {
      state.isMicActive = false;
      state.micStatus = 'Microphone stopped';
      for (const listener of listeners) listener(state);
    }),
    dispose: vi.fn(async () => undefined),
    emit(update: NonNullable<BrowserRealtimeMicrophoneControllerState['transcript']>) {
      state.transcript = update;
      state.transcriptText = update.canonical.text;
      for (const listener of listeners) listener(state);
      createMicrophone.mock.calls.at(-1)?.[0].onTranscript(update);
    },
    ...overrides,
  };
  createMicrophone.mockReturnValue(mic);
  return mic;
}

function update(text: string) {
  return {
    kind: 'partial' as const,
    trigger: 'push' as const,
    partial: {
      kind: 'partial' as const,
      revision: 1,
      text,
      committedText: '',
      previewText: text,
      warnings: [],
      meta: {},
    },
    canonical: { text, warnings: [], meta: {} },
    snapshot: { text },
    window: {},
    activity: null,
    context: { id: 'turn-1', revision: 1, startFrame: 0, endFrame: 20, sampleRate: 16000 },
  };
}

describe('createSpeechController microphone composition', () => {
  beforeEach(() => {
    loadSpeechModel.mockReset();
    createMicrophone.mockReset();
    decodeAudioSourceToMonoPcm.mockReset();
  });

  it('uses the loaded model, exposes transcript state, and reuses it after stop', async () => {
    const model = fakeModel();
    const mic = fakeMicrophone();
    loadSpeechModel.mockResolvedValue(model);
    const controller = createSpeechController();
    await controller.load({ modelId: 'mic-model' });
    const onTranscript = vi.fn();
    await controller.startMicrophone({ micMode: 'speech-detect', onTranscript });

    expect(createMicrophone).toHaveBeenCalledWith(
      expect.objectContaining({ model, micMode: 'speech-detect' }),
    );
    expect(controller.getSnapshot().status).toBe('streaming');
    mic.emit(update('hello there'));
    expect(onTranscript).toHaveBeenCalledOnce();
    expect(controller.getSnapshot()).toMatchObject({
      result: { text: 'hello there' },
      microphone: { transcriptText: 'hello there' },
    });
    expect(controller.getSnapshot().result?.segments).toBeUndefined();

    await controller.stopMicrophone();
    expect(mic.stop).toHaveBeenCalledWith({ flush: true });
    expect(mic.dispose).toHaveBeenCalledOnce();
    expect(controller.getSnapshot()).toMatchObject({
      status: 'ready',
      result: { text: 'hello there' },
    });
    await controller.transcribe(new Float32Array([0]), 16000);
    expect(model.transcribeMonoPcm).toHaveBeenCalledOnce();
    expect(controller.getSnapshot().microphone).toBeNull();
    await controller.dispose();
    expect(model.dispose).toHaveBeenCalledOnce();
  });

  it('clears the previous microphone transcript when a new file transcription starts', async () => {
    const model = fakeModel();
    const mic = fakeMicrophone();
    loadSpeechModel.mockResolvedValue(model);
    decodeAudioSourceToMonoPcm.mockResolvedValue({ pcm: new Float32Array([0]), sampleRate: 16000 });
    const controller = createSpeechController();
    await controller.load({ modelId: 'mic-model' });
    await controller.startMicrophone();
    mic.emit(update('old microphone text'));
    await controller.stopMicrophone();

    await controller.transcribeSource('new-file.wav');
    expect(controller.getSnapshot()).toMatchObject({ result: { text: 'done' }, microphone: null });
    await controller.dispose();
  });

  it('keeps model operations exclusive for the whole microphone session', async () => {
    const model = fakeModel();
    fakeMicrophone();
    loadSpeechModel.mockResolvedValue(model);
    const controller = createSpeechController();
    await controller.load({ modelId: 'mic-model' });
    await controller.startMicrophone();
    await expect(controller.load({ modelId: 'other' })).rejects.toThrow(
      'while microphone is active',
    );
    await expect(controller.transcribe(new Float32Array([0]), 16000)).rejects.toThrow(
      'Another controller operation is active',
    );
    await expect(controller.startStreaming()).rejects.toThrow(
      'Another controller operation is active',
    );
    await controller.stopMicrophone();
    await controller.dispose();
  });

  it('disposes the borrowed microphone before waiting for startup and before the model', async () => {
    const starting = deferred<void>();
    const order: string[] = [];
    const model = fakeModel();
    model.dispose.mockImplementation(async () => {
      order.push('model');
    });
    const mic = fakeMicrophone({
      start: vi.fn(() => starting.promise),
      dispose: vi.fn(async () => {
        order.push('mic');
      }),
    });
    loadSpeechModel.mockResolvedValue(model);
    const controller = createSpeechController();
    await controller.load({ modelId: 'mic-model' });
    const launch = controller.startMicrophone().catch((error: unknown) => error);
    await vi.waitFor(() => expect(mic.start).toHaveBeenCalledOnce());
    const disposing = controller.dispose();
    await vi.waitFor(() => expect(mic.dispose).toHaveBeenCalledOnce());
    expect(model.dispose).not.toHaveBeenCalled();
    starting.resolve();
    expect(await launch).toMatchObject({ name: 'AbortError' });
    await disposing;
    expect(order).toEqual(['mic', 'model']);
    expect(model.dispose).toHaveBeenCalledOnce();
  });

  it('stops during startup without leaving a microphone or rejecting the stop request', async () => {
    const starting = deferred<void>();
    const model = fakeModel();
    const mic = fakeMicrophone({
      start: vi.fn(() => starting.promise),
      stop: vi.fn(async () => {
        starting.resolve();
      }),
    });
    loadSpeechModel.mockResolvedValue(model);
    const controller = createSpeechController();
    await controller.load({ modelId: 'mic-model' });
    const launch = controller.startMicrophone().catch((error: unknown) => error);
    await vi.waitFor(() => expect(mic.start).toHaveBeenCalledOnce());

    await expect(controller.stopMicrophone()).resolves.toBeUndefined();
    expect(await launch).toMatchObject({ name: 'AbortError' });
    expect(mic.stop).toHaveBeenCalledWith({ flush: true });
    expect(mic.dispose).toHaveBeenCalledOnce();
    expect(controller.getSnapshot().status).toBe('ready');
    await controller.dispose();
  });

  it('cleans up a failed startup and releases the loaded model for a later operation', async () => {
    const failure = new Error('capture permission denied');
    const model = fakeModel();
    const mic = fakeMicrophone({
      start: vi.fn(async () => {
        throw failure;
      }),
    });
    loadSpeechModel.mockResolvedValue(model);
    const controller = createSpeechController();
    await controller.load({ modelId: 'mic-model' });
    await expect(controller.startMicrophone()).rejects.toBe(failure);
    expect(mic.dispose).toHaveBeenCalledOnce();
    expect(controller.getSnapshot()).toMatchObject({ status: 'error', error: failure });
    await controller.transcribe(new Float32Array([0]), 16000);
    await controller.dispose();
  });

  it('disposes and releases the microphone even when stop fails', async () => {
    const failure = new Error('capture stop failed');
    const model = fakeModel();
    const mic = fakeMicrophone({
      stop: vi.fn(async () => {
        throw failure;
      }),
    });
    loadSpeechModel.mockResolvedValue(model);
    const controller = createSpeechController();
    await controller.load({ modelId: 'mic-model' });
    await controller.startMicrophone();
    mic.emit(update('keep this transcript'));
    await expect(controller.stopMicrophone()).rejects.toBe(failure);
    expect(mic.dispose).toHaveBeenCalledOnce();
    expect(controller.getSnapshot()).toMatchObject({
      status: 'error',
      error: failure,
      microphone: {
        isMicActive: false,
        transcriptText: 'keep this transcript',
      },
    });
    await controller.transcribe(new Float32Array([0]), 16000);
    await controller.dispose();
  });

  it('preserves an undefined stop rejection while releasing microphone resources', async () => {
    const model = fakeModel();
    const mic = fakeMicrophone({ stop: vi.fn(async () => Promise.reject(undefined)) });
    loadSpeechModel.mockResolvedValue(model);
    const controller = createSpeechController();
    await controller.load({ modelId: 'mic-model' });
    await controller.startMicrophone();

    await expect(controller.stopMicrophone()).rejects.toBeUndefined();
    expect(mic.dispose).toHaveBeenCalledOnce();
    expect(controller.getSnapshot()).toMatchObject({
      status: 'error',
      microphone: { isMicActive: false },
    });
    await controller.transcribe(new Float32Array([0]), 16000);
    await controller.dispose();
  });
});
