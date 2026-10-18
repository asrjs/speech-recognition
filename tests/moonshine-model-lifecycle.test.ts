import { beforeEach, describe, expect, it, vi } from 'vitest';
import { MoonshineSpeechSession, createMoonshineModelFamily } from '../src/models/moonshine/model.js';
import { initOrt } from '../src/models/lasr-ctc/ort.js';
import { loadMoonshineStreaming } from '../src/models/moonshine/streaming-loader.js';

vi.mock('../src/models/lasr-ctc/ort.js', () => ({ initOrt: vi.fn() }));
vi.mock('../src/models/moonshine/streaming-loader.js', () => ({ loadMoonshineStreaming: vi.fn() }));

const assets = {
  encoderUrl: 'encoder', crossKvUrl: 'cross', decoderUrl: 'decoder', tokenizerUrl: 'tokenizer',
};

describe('Moonshine model allocation ownership', () => {
  beforeEach(() => vi.clearAllMocks());
  it('does not advertise native streaming for an offline-only artifact set', async () => {
    const model = await createMoonshineModelFamily().createModel({
      modelId: 'moonshine-tiny', options: assets,
    } as never, { backend: { id: 'wasm' } } as never);
    expect(model.createStreamingTranscriber).toBeUndefined();
    await model.dispose();
  });

  it.each(['createSession', 'createStreamingTranscriber'] as const)(
    'releases a late streaming load after the owner is disposed (%s)', async (method) => {
      let finish!: (value: any) => void;
      const loading = new Promise<any>((resolve) => { finish = resolve; });
      vi.mocked(loadMoonshineStreaming).mockReturnValue(loading);
      const release = vi.fn(async () => undefined);
      const model = await createMoonshineModelFamily().createModel({
        modelId: 'moonshine-streaming-tiny', options: { streaming: {
          chunkedEncoderUrl: 'chunked', encoderUrl: 'encoder', prefillUrl: 'prefill',
          stepUrl: 'step', tokenizerUrl: 'tokenizer',
        } },
      } as never, { backend: { id: 'wasm' } } as never);
      const pending = model[method]!();
      await Promise.resolve();
      await model.dispose();
      finish({ transcriber: {}, dispose: release });
      await expect(pending).rejects.toThrow(/disposed/);
      expect(release).toHaveBeenCalledTimes(1);
    },
  );

  it('releases each successful graph exactly once after public session creation fails', async () => {
    const releases = [vi.fn(), vi.fn()];
    vi.mocked(initOrt).mockResolvedValue({ InferenceSession: {
      create: async (url: string) => {
        if (url === 'cross') throw new Error('cross graph failed');
        await new Promise((resolve) => setTimeout(resolve, 5));
        return { run: vi.fn(), release: releases[url === 'encoder' ? 0 : 1] };
      },
    } } as never);
    const model = await createMoonshineModelFamily().createModel({
      modelId: 'moonshine-tiny', options: assets,
    } as never, { backend: { id: 'webgpu' } } as never);
    await expect(model.createSession()).rejects.toThrow('cross graph failed');
    await model.dispose();
    for (const release of releases) expect(release).toHaveBeenCalledTimes(1);
    expect(initOrt).toHaveBeenCalledWith('wasm', expect.any(Object));
  });

  it('releases late allocations once when disposed before initialization completes', async () => {
    const releases = [vi.fn(), vi.fn(), vi.fn()];
    let finish!: () => void;
    const pending = new Promise<void>((resolve) => { finish = resolve; });
    let index = 0;
    vi.mocked(initOrt).mockResolvedValue({ InferenceSession: {
      create: async () => {
        const release = releases[index++];
        await pending;
        return { run: vi.fn(), release };
      },
    } } as never);
    const session = new MoonshineSpeechSession('moonshine-tiny', 'wasm', assets, undefined);
    const initializing = session.initialize();
    await Promise.resolve();
    await session.dispose();
    finish();
    await expect(initializing).rejects.toThrow('disposed during initialization');
    for (const release of releases) expect(release).toHaveBeenCalledTimes(1);
  });
});
