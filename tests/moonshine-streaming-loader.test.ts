import { describe, expect, it, vi } from 'vitest';
import { loadMoonshineStreaming } from '../src/models/moonshine/streaming-loader.js';
import type { MoonshineOrtLike } from '../src/models/moonshine/streaming-loader.js';
import { moonshineBuiltInLocalModelAdapter } from '../src/presets/moonshine-local-adapter.js';
import { MoonshineStreamingTranscriber } from '../src/models/moonshine/streaming-transcriber.js';

const ASSETS = {
  chunkedEncoderUrl: 'blob:chunked',
  encoderUrl: 'blob:whole',
  prefillUrl: 'blob:prefill',
  stepUrl: 'blob:step',
  tokenizerUrl: 'blob:tokenizer',
};

function fakeOrt() {
  const created: Array<{ url: string; options?: Record<string, unknown>; released: boolean }> = [];
  const ort: MoonshineOrtLike = {
    Tensor: class {
      constructor(
        readonly type: string,
        readonly data: ArrayBufferView,
        readonly dims: readonly number[],
      ) {}
    } as unknown as MoonshineOrtLike['Tensor'],
    InferenceSession: {
      create: vi.fn(async (url: string, options?: Record<string, unknown>) => {
        const record = { url, options, released: false };
        created.push(record);
        return {
          run: async () => ({}),
          release: () => {
            record.released = true;
          },
        };
      }),
    },
  };
  return { ort, created };
}

const tokenizerJson = { model: { vocab: { '<s>': 1, '</s>': 2, '▁hello': 3 } } };

describe('loadMoonshineStreaming', () => {
  it('does not retry WASM after a request is aborted during WebGPU graph creation', async () => {
    const controller = new AbortController();
    const { ort, created } = fakeOrt();
    const create = vi.mocked(ort.InferenceSession.create);
    create.mockImplementation(async () => {
      controller.abort();
      throw new Error('aborted GPU creation');
    });
    await expect(loadMoonshineStreaming({
      ...ASSETS, ort, encoderBackend: 'webgpu', signal: controller.signal,
      fetchJson: async () => tokenizerJson,
    })).rejects.toThrow(/abort/i);
    expect(create).toHaveBeenCalledTimes(4);
    expect(created).toEqual([]);
  });

  it('creates the four graph sessions, builds the transcriber, and releases on dispose', async () => {
    const { ort, created } = fakeOrt();
    const loaded = await loadMoonshineStreaming({
      ...ASSETS,
      ort,
      fetchJson: async () => tokenizerJson,
    });
    expect(created.map((row) => row.url)).toEqual([
      'blob:chunked',
      'blob:whole',
      'blob:prefill',
      'blob:step',
    ]);
    expect(created[0]!.options?.executionProviders).toEqual(['wasm']);
    expect(loaded.transcriber).toBeInstanceOf(MoonshineStreamingTranscriber);
    expect(loaded.session.committedEncoderFrames).toBe(0);
    await loaded.dispose();
    expect(created.every((row) => row.released)).toBe(true);
  });

  it('selects webgpu providers when requested', async () => {
    const { ort, created } = fakeOrt();
    const loaded = await loadMoonshineStreaming({
      ...ASSETS,
      ort,
      encoderBackend: 'webgpu',
      fetchJson: async () => tokenizerJson,
    });
    expect(created.every((row) => row.options?.executionProviders?.join(',') === 'webgpu')).toBe(
      true,
    );
    expect(loaded.resolvedBackend).toBe('webgpu');
    await loaded.dispose();
  });

  it('waits for all WebGPU graph creates, releases partial successes, then reports the WASM fallback', async () => {
    const requests: Array<{ url: string; backend: string; rejected: boolean }> = [];
    const released: string[] = [];
    const ort: MoonshineOrtLike = {
      Tensor: class {
        constructor(
          readonly type: string,
          readonly data: ArrayBufferView,
          readonly dims: readonly number[],
        ) {}
      } as unknown as MoonshineOrtLike['Tensor'],
      InferenceSession: {
        create: vi.fn(async (url: string, options?: Record<string, unknown>) => {
          const backend = (options?.executionProviders as string[] | undefined)?.[0] ?? 'unknown';
          const rejects = backend === 'webgpu' && url === 'blob:step';
          requests.push({ url, backend, rejected: rejects });
          await new Promise((resolve) =>
            setTimeout(resolve, rejects ? 2 : backend === 'webgpu' ? 12 : 0),
          );
          if (rejects) throw new Error('WebGPU graph unsupported');
          return {
            run: async () => ({}),
            release: () => {
              released.push(url);
            },
          };
        }),
      },
    };

    const loaded = await loadMoonshineStreaming({
      ...ASSETS,
      ort,
      encoderBackend: 'webgpu',
      fetchJson: async () => tokenizerJson,
    });

    expect(requests.filter((request) => request.backend === 'webgpu')).toHaveLength(4);
    expect(released.sort()).toEqual(['blob:chunked', 'blob:prefill', 'blob:whole']);
    expect(requests.filter((request) => request.backend === 'wasm')).toHaveLength(4);
    expect(loaded.resolvedBackend).toBe('wasm');
    expect(loaded.resolvedEncoderBackend).toBe('wasm');
    expect(loaded.backendFallbackReason).toContain('WebGPU graph unsupported');
    await loaded.dispose();
  });

  it('releases created sessions when the tokenizer fetch fails', async () => {
    const { ort, created } = fakeOrt();
    await expect(
      loadMoonshineStreaming({
        ...ASSETS,
        ort,
        fetchJson: async () => {
          throw new Error('tokenizer unavailable');
        },
      }),
    ).rejects.toThrow('tokenizer unavailable');
    expect(created.length).toBe(4);
    expect(created.every((row) => row.released)).toBe(true);
  });
});

describe('moonshine local adapter streaming routing', () => {
  const offlineFiles = [
    new File(['enc'], 'encoder_model.onnx'),
    new File(['kv'], 'decoder_cross_kv_model.onnx'),
    new File(['dec'], 'decoder_with_past_model.onnx'),
    new File(['tok'], 'tokenizer.json'),
    new File(['{}'], 'generation_config.json'),
    new File(['{}'], 'export-receipt.json'),
  ];
  const streamingFiles = [
    new File(['chunked'], 'encoder_chunked_model.onnx'),
    new File(['prefill'], 'decoder_prefill_model.onnx'),
    new File(['step'], 'decoder_step_model.onnx'),
  ];

  it('exposes streaming asset URLs when the folder carries the streaming graphs', async () => {
    const entries = moonshineBuiltInLocalModelAdapter.createEntries([
      ...offlineFiles,
      ...streamingFiles,
    ] as unknown as readonly File[]);
    const resolved = await moonshineBuiltInLocalModelAdapter.resolveEntries({
      entries,
      modelId: 'moonshine-streaming-tiny',
    });
    const streaming = (resolved.builtInLoadOptions as Record<string, unknown>)['streaming'] as
      | Record<string, string>
      | undefined;
    expect(streaming).toBeDefined();
    expect(streaming!['chunkedEncoderUrl']).toMatch(/^blob:/);
    expect(streaming!['prefillUrl']).toMatch(/^blob:/);
    expect(streaming!['stepUrl']).toMatch(/^blob:/);
    // The whole-utterance encoder doubles as the final-flush graph.
    expect(streaming!['encoderUrl']).toBe(
      (resolved.builtInLoadOptions as Record<string, unknown>)['encoderUrl'],
    );
    await Promise.all(resolved.assetHandles.map(async (handle) => handle.dispose()));
  });

  it('omits streaming assets for offline-only folders', async () => {
    const entries = moonshineBuiltInLocalModelAdapter.createEntries(
      offlineFiles as unknown as readonly File[],
    );
    const resolved = await moonshineBuiltInLocalModelAdapter.resolveEntries({
      entries,
      modelId: 'moonshine-tiny',
    });
    expect((resolved.builtInLoadOptions as Record<string, unknown>)['streaming']).toBeUndefined();
    await Promise.all(resolved.assetHandles.map(async (handle) => handle.dispose()));
  });
});
