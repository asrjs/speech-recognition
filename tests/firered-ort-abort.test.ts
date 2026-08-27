import { afterEach, describe, expect, it, vi } from 'vitest';

import { AssetLoadAbortedError } from '../src/io/abort.js';
import { MemoryAssetCache } from '../src/runtime/firered-vad/core/asset-cache.js';
import { loadBinaryResource } from '../src/runtime/firered-vad/core/loader.js';

const originalFetch = globalThis.fetch;

const ortHarness = vi.hoisted(() => {
  const sessions: Array<{ release: ReturnType<typeof vi.fn> }> = [];
  let createChain: Promise<void> = Promise.resolve();
  return {
    sessions,
    reset() {
      sessions.length = 0;
      createChain = Promise.resolve();
    },
    enqueueCreate(create: () => Promise<{ release: ReturnType<typeof vi.fn> }>) {
      const run = createChain.then(create);
      createChain = run.then(
        () => undefined,
        () => undefined,
      );
      return run;
    },
  };
});

vi.mock('onnxruntime-web', () => ({
  env: { wasm: {}, versions: { common: '1.24.1' } },
  Tensor: class Tensor {},
  InferenceSession: {
    create: vi.fn(() =>
      ortHarness.enqueueCreate(async () => {
        const session = { release: vi.fn(async () => undefined) };
        ortHarness.sessions.push(session);
        return session;
      }),
    ),
  },
}));

afterEach(() => {
  vi.restoreAllMocks();
  globalThis.fetch = originalFetch;
  ortHarness.reset();
});

describe('FireRed loadBinaryResource abort', () => {
  it('stops a mid-download fetch and skips cache writes', async () => {
    const cache = new MemoryAssetCache();
    const set = vi.spyOn(cache, 'set');
    const controller = new AbortController();

    globalThis.fetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const abortError = () => {
        const error = new Error('Aborted');
        error.name = 'AbortError';
        return error;
      };
      if (init?.signal?.aborted) {
        throw abortError();
      }
      await new Promise<never>((_resolve, reject) => {
        const fail = () => reject(abortError());
        init?.signal?.addEventListener('abort', fail);
        queueMicrotask(() => controller.abort());
      });
    }) as typeof fetch;

    await expect(
      loadBinaryResource('https://example.com/fireredvad.onnx', cache, controller.signal),
    ).rejects.toBeInstanceOf(AssetLoadAbortedError);
    expect(set).not.toHaveBeenCalled();
  });
});

describe('FireRed ORT session abort', () => {
  it('releases sibling sessions when abort is observed after the first create', async () => {
    const { createOrtFireRedBackend } = await import('../src/runtime/firered-vad/core/backend.js');
    const signal = { aborted: false };
    const originalCreate = (await import('onnxruntime-web')).InferenceSession.create;
    vi.mocked(originalCreate).mockImplementation(() =>
      ortHarness.enqueueCreate(async () => {
        const session = { release: vi.fn(async () => undefined) };
        ortHarness.sessions.push(session);
        if (ortHarness.sessions.length > 1) {
          signal.aborted = true;
        }
        return session;
      }),
    );

    globalThis.fetch = vi.fn(async () =>
      new Response(new Uint8Array([1, 2, 3]), { status: 200 }),
    ) as typeof fetch;

    await expect(
      createOrtFireRedBackend({
        cacheAssets: false,
        signal,
        modelUrls: {
          vadUrl: 'https://example.com/vad.onnx',
          streamVadWithCacheUrl: 'https://example.com/stream.onnx',
          aedUrl: 'https://example.com/aed.onnx',
        },
      }),
    ).rejects.toBeInstanceOf(AssetLoadAbortedError);

    expect(ortHarness.sessions.length).toBeGreaterThanOrEqual(1);
    for (const session of ortHarness.sessions) {
      expect(session.release).toHaveBeenCalled();
    }
  });
});
