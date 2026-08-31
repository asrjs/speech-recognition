import { afterEach, describe, expect, it, vi } from 'vitest';

import { BlobAssetHandle, UrlAssetHandle } from '../src/io/handles.js';
import type { AssetCache, AssetProgressEvent } from '../src/types/index.js';

const originalFetch = globalThis.fetch;

afterEach(() => {
  vi.restoreAllMocks();
  globalThis.fetch = originalFetch;
});

describe('UrlAssetHandle', () => {
  it('falls back to main revision when a Hugging Face revision returns 404', async () => {
    const calls: string[] = [];
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      calls.push(url);
      if (calls.length === 1) {
        return new Response('missing', { status: 404, statusText: 'Not Found' });
      }
      return new Response(new Uint8Array([1, 2, 3]), {
        status: 200,
        headers: { 'content-type': 'application/octet-stream' },
      });
    }) as typeof fetch;

    const handle = new UrlAssetHandle(
      {
        id: 'hf:test',
        provider: 'huggingface',
        repoId: 'ysdede/parakeet-tdt-0.6b-v3-onnx',
        revision: 'feature/missing',
        filename: 'encoder-model.fp16.onnx',
        cacheKey: 'hf:test',
      },
      'https://huggingface.co/ysdede/parakeet-tdt-0.6b-v3-onnx/resolve/feature%2Fmissing/encoder-model.fp16.onnx',
    );

    const bytes = await handle.readBytes();
    expect(Array.from(bytes)).toEqual([1, 2, 3]);
    expect(calls).toHaveLength(2);
    expect(calls[0]).toContain('/resolve/feature%2Fmissing/');
    expect(calls[1]).toContain('/resolve/main/');
  });

  it('returns a single streamed response chunk without an extra materialization copy', async () => {
    const payload = new Uint8Array([11, 22, 33, 44]);
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(payload);
        controller.close();
      },
    });
    globalThis.fetch = vi.fn(
      async () =>
        new Response(stream, {
          status: 200,
          headers: { 'content-type': 'application/octet-stream' },
        }),
    ) as typeof fetch;

    const handle = new UrlAssetHandle(
      {
        id: 'url:single-chunk',
        provider: 'url',
        url: 'https://example.com/single-chunk.onnx',
      },
      'https://example.com/single-chunk.onnx',
    );

    const bytes = await handle.readBytes();

    expect(bytes).toBe(payload);
    expect(Array.from(bytes)).toEqual([11, 22, 33, 44]);
  });

  it('continues with network when cache read throws and evicts the broken key', async () => {
    const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const cache: AssetCache = {
      get: vi.fn(async () => {
        throw new Error('cache blob invalid');
      }),
      set: vi.fn(async () => undefined),
      delete: vi.fn(async () => undefined),
    };

    globalThis.fetch = vi.fn(async () => {
      return new Response(new Uint8Array([9, 8, 7]), {
        status: 200,
        headers: { 'content-type': 'application/octet-stream' },
      });
    }) as typeof fetch;

    const handle = new UrlAssetHandle(
      {
        id: 'url:test',
        provider: 'url',
        url: 'https://example.com/model.onnx',
        cacheKey: 'cache:broken',
      },
      'https://example.com/model.onnx',
      cache,
    );

    const bytes = await handle.readBytes();
    expect(Array.from(bytes)).toEqual([9, 8, 7]);
    expect(cache.get).toHaveBeenCalledTimes(1);
    expect(cache.delete).toHaveBeenCalledTimes(1);
    expect(cache.set).toHaveBeenCalledTimes(1);
    expect(consoleWarnSpy).toHaveBeenCalledWith(
      '[assets] Cache read failed for "cache:broken". Falling back to network.',
      expect.any(Error)
    );
    consoleWarnSpy.mockRestore();
  });

  it('reads fallback cache keys and migrates hits to the primary key', async () => {
    const cache: AssetCache = {
      get: vi.fn(async (key: string) =>
        key === 'cache:legacy'
          ? {
              bytes: new Uint8Array([4, 5, 6]),
              contentType: 'application/octet-stream',
            }
          : null,
      ),
      set: vi.fn(async () => undefined),
      delete: vi.fn(async () => undefined),
    };
    const fetchSpy = vi.fn();
    globalThis.fetch = fetchSpy as unknown as typeof fetch;

    const handle = new UrlAssetHandle(
      {
        id: 'url:test',
        provider: 'url',
        url: 'https://example.com/model.onnx',
        cacheKey: 'cache:primary',
        cacheKeyFallbacks: ['cache:legacy'],
      },
      'https://example.com/model.onnx',
      cache,
    );

    const bytes = await handle.readBytes();
    expect(Array.from(bytes)).toEqual([4, 5, 6]);
    expect(cache.get).toHaveBeenCalledWith('cache:primary');
    expect(cache.get).toHaveBeenCalledWith('cache:legacy');
    expect(cache.set).toHaveBeenCalledWith('cache:primary', {
      bytes,
      contentType: 'application/octet-stream',
    });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('returns remote URL locators without materializing the full payload', async () => {
    const fetchSpy = vi.fn();
    globalThis.fetch = fetchSpy as unknown as typeof fetch;

    const handle = new UrlAssetHandle(
      {
        id: 'hf:large-data',
        provider: 'huggingface',
        repoId: 'ysdede/parakeet-tdt-0.6b-v3-onnx',
        revision: 'feat/fp16-canonical-v3',
        filename: 'encoder-model.onnx.data',
      },
      'https://huggingface.co/ysdede/parakeet-tdt-0.6b-v3-onnx/resolve/feat%2Ffp16-canonical-v3/encoder-model.onnx.data',
    );

    const locator = await handle.getLocator('url');
    expect(locator).toContain('/resolve/feat%2Ffp16-canonical-v3/encoder-model.onnx.data');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('streams network chunks without retaining them when caching is disabled', async () => {
    const chunks = [new Uint8Array([1, 2]), new Uint8Array([3, 4])];
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(chunk);
        }
        controller.close();
      },
    });
    globalThis.fetch = vi.fn(
      async () =>
        new Response(stream, {
          status: 200,
          headers: { 'content-type': 'application/octet-stream' },
        }),
    ) as typeof fetch;

    const handle = new UrlAssetHandle(
      {
        id: 'url:stream-no-cache',
        provider: 'url',
        url: 'https://example.com/model.onnx',
      },
      'https://example.com/model.onnx',
    );

    const received: Uint8Array[] = [];
    for await (const chunk of handle.openStream()) {
      received.push(chunk);
    }

    expect(received).toHaveLength(2);
    expect(received.map((chunk) => Array.from(chunk))).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  it('assembles streamed chunks before writing a requested cache entry', async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array([5, 6]));
        controller.enqueue(new Uint8Array([7, 8]));
        controller.close();
      },
    });
    const cache: AssetCache = {
      get: vi.fn(async () => null),
      set: vi.fn(async () => undefined),
    };
    globalThis.fetch = vi.fn(
      async () =>
        new Response(stream, {
          status: 200,
          headers: { 'content-type': 'application/octet-stream' },
        }),
    ) as typeof fetch;

    const handle = new UrlAssetHandle(
      {
        id: 'url:stream-cache',
        provider: 'url',
        url: 'https://example.com/model.onnx',
        cacheKey: 'cache:stream-cache',
      },
      'https://example.com/model.onnx',
      cache,
    );

    const received: number[] = [];
    for await (const chunk of handle.openStream()) {
      received.push(...chunk);
    }

    expect(received).toEqual([5, 6, 7, 8]);
    expect(cache.set).toHaveBeenCalledWith('cache:stream-cache', {
      bytes: expect.any(Uint8Array),
      contentType: 'application/octet-stream',
    });
    expect(Array.from((cache.set as ReturnType<typeof vi.fn>).mock.calls[0]![1].bytes)).toEqual([
      5, 6, 7, 8,
    ]);
  });

  it('can materialize remote URL locators as blob URLs with progress', async () => {
    const progressEvents: AssetProgressEvent[] = [];
    globalThis.fetch = vi.fn(async () => {
      return new Response(new Uint8Array([1, 2, 3, 4]), {
        status: 200,
        headers: {
          'content-length': '4',
          'content-type': 'application/octet-stream',
        },
      });
    }) as typeof fetch;

    const handle = new UrlAssetHandle(
      {
        id: 'hf:blob-locator',
        provider: 'huggingface',
        repoId: 'ysdede/parakeet-tdt-0.6b-v3-onnx',
        revision: 'feat/fp16-canonical-v3',
        filename: 'encoder-model.fp16.onnx',
        preferBlobUrl: true,
        onProgress(event) {
          progressEvents.push(event);
        },
      },
      'https://huggingface.co/ysdede/parakeet-tdt-0.6b-v3-onnx/resolve/feat%2Ffp16-canonical-v3/encoder-model.fp16.onnx',
    );

    const locator = await handle.getLocator('url');
    expect(locator).toMatch(/^blob:/);
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    expect(progressEvents.at(-1)).toMatchObject({ loaded: 4, done: true, source: 'network' });
    handle.dispose();
  });

  it('reports cache provenance when a blob URL locator is materialized from cache', async () => {
    const progressEvents: AssetProgressEvent[] = [];
    const cache: AssetCache = {
      get: vi.fn(async () => null),
      set: vi.fn(async () => undefined),
      getBlob: vi.fn(async () => new Blob([new Uint8Array([1, 2, 3, 4])])),
      setBlob: vi.fn(async () => undefined),
      delete: vi.fn(async () => undefined),
    };
    const fetchSpy = vi.fn();
    globalThis.fetch = fetchSpy as unknown as typeof fetch;

    const handle = new UrlAssetHandle(
      {
        id: 'url:warm-blob-cache',
        provider: 'url',
        url: 'https://example.com/encoder_model.onnx',
        preferBlobUrl: true,
        cacheKey: 'cache:warm-blob',
        onProgress(event) {
          progressEvents.push(event);
        },
      },
      'https://example.com/encoder_model.onnx',
      cache,
    );

    const locator = await handle.getLocator('url');

    expect(locator).toMatch(/^blob:/);
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(progressEvents).toEqual([
      expect.objectContaining({ loaded: 4, total: 4, done: true, source: 'cache' }),
    ]);
    handle.dispose();
  });

  it('uses blob cache methods when materializing blob URL locators', async () => {
    const cache: AssetCache = {
      get: vi.fn(async () => null),
      set: vi.fn(async () => undefined),
      getBlob: vi.fn(async () => null),
      setBlob: vi.fn(async () => undefined),
      delete: vi.fn(async () => undefined),
    };

    globalThis.fetch = vi.fn(async () => {
      return new Response(new Uint8Array([5, 6, 7, 8]), {
        status: 200,
        headers: {
          'content-length': '4',
          'content-type': 'application/octet-stream',
        },
      });
    }) as typeof fetch;

    const handle = new UrlAssetHandle(
      {
        id: 'url:blob-cache',
        provider: 'url',
        url: 'https://example.com/encoder_model.onnx.data',
        preferBlobUrl: true,
        cacheKey: 'cache:blob',
      },
      'https://example.com/encoder_model.onnx.data',
      cache,
    );

    const locator = await handle.getLocator('url');

    expect(locator).toMatch(/^blob:/);
    expect(cache.getBlob).toHaveBeenCalledWith('cache:blob');
    expect(cache.setBlob).toHaveBeenCalledTimes(1);
    expect(cache.set).not.toHaveBeenCalled();
    handle.dispose();
  });

  it('materializes a single blob URL for concurrent getLocator callers', async () => {
    const created: string[] = [];
    const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockImplementation(() => {
      const url = `blob:concurrent-${created.length}`;
      created.push(url);
      return url;
    });
    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

    globalThis.fetch = vi.fn(async () => {
      return new Response(new Uint8Array([1, 2, 3, 4]), {
        status: 200,
        headers: { 'content-type': 'application/octet-stream' },
      });
    }) as typeof fetch;

    const handle = new UrlAssetHandle(
      {
        id: 'url:concurrent-locator',
        provider: 'url',
        url: 'https://example.com/encoder.onnx',
        preferBlobUrl: true,
      },
      'https://example.com/encoder.onnx',
    );

    const locators = await Promise.all([handle.getLocator('url'), handle.getLocator('url')]);

    expect(locators[0]).toBe(locators[1]);
    expect(locators[0]).toBe('blob:concurrent-0');
    expect(createObjectURL).toHaveBeenCalledTimes(1);

    handle.dispose();
    expect(revokeObjectURL).toHaveBeenCalledTimes(1);
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:concurrent-0');
  });

  it('does not leak a blob URL when dispose races an in-flight getLocator', async () => {
    const created: string[] = [];
    const revoked: string[] = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation(() => {
      const url = `blob:inflight-${created.length}`;
      created.push(url);
      return url;
    });
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation((url) => {
      revoked.push(String(url));
    });

    let releaseFetch!: () => void;
    const fetchGate = new Promise<void>((resolve) => {
      releaseFetch = resolve;
    });
    globalThis.fetch = vi.fn(async () => {
      await fetchGate;
      return new Response(new Uint8Array([9, 8, 7]), {
        status: 200,
        headers: { 'content-type': 'application/octet-stream' },
      });
    }) as typeof fetch;

    const handle = new UrlAssetHandle(
      {
        id: 'url:dispose-during-locator',
        provider: 'url',
        url: 'https://example.com/decoder.onnx',
        preferBlobUrl: true,
      },
      'https://example.com/decoder.onnx',
    );

    const pending = handle.getLocator('url');
    handle.dispose();
    releaseFetch();

    await expect(pending).rejects.toThrow(
      'Asset handle "url:dispose-during-locator" has been disposed; blob URL locators cannot be created.',
    );
    expect(created.every((url) => revoked.includes(url))).toBe(true);
    await expect(handle.getLocator('url')).rejects.toThrow(/has been disposed/);
  });

  it('stops a mid-download fetch, skips cache writes, and disposes the handle', async () => {
    const cache: AssetCache = {
      get: vi.fn(async () => null),
      set: vi.fn(async () => undefined),
      delete: vi.fn(async () => undefined),
    };
    const controller = new AbortController();
    const progressEvents: AssetProgressEvent[] = [];
    let pullCount = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(ctrl) {
        pullCount += 1;
        if (pullCount === 2) {
          controller.abort();
        }
        ctrl.enqueue(new Uint8Array([pullCount]));
      },
    });
    const fetchSpy = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(init?.signal).toBe(controller.signal);
      return new Response(stream, {
        status: 200,
        headers: { 'content-type': 'application/octet-stream', 'content-length': '1000' },
      });
    });
    globalThis.fetch = fetchSpy as unknown as typeof fetch;

    const handle = new UrlAssetHandle(
      {
        id: 'url:abort-mid-download',
        provider: 'url',
        url: 'https://example.com/model.onnx',
        cacheKey: 'cache:abort-mid-download',
        signal: controller.signal,
        onProgress(event) {
          progressEvents.push(event);
        },
      },
      'https://example.com/model.onnx',
      cache,
    );

    await expect(handle.readBytes()).rejects.toMatchObject({
      name: 'AssetLoadAbortedError',
      code: 'asset-load-aborted',
    });
    expect(cache.set).not.toHaveBeenCalled();
    expect(progressEvents.some((event) => event.done === true && !event.aborted)).toBe(false);
    expect(progressEvents.at(-1)).toMatchObject({
      aborted: true,
      done: true,
    });
    await expect(handle.getLocator('url')).rejects.toMatchObject({
      name: 'AssetLoadAbortedError',
    });
  });
});

describe('BlobAssetHandle', () => {
  it('revokes the blob URL on dispose and refuses later locators', async () => {
    const created: string[] = [];
    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    vi.spyOn(URL, 'createObjectURL').mockImplementation(() => {
      const url = `blob:memory-${created.length}`;
      created.push(url);
      return url;
    });

    const handle = new BlobAssetHandle(
      {
        id: 'blob:local-encoder',
        provider: 'blob',
        blob: new Blob([new Uint8Array([1, 2])]),
      },
      new Blob([new Uint8Array([1, 2])]),
    );

    const locator = await handle.getLocator('url');
    expect(locator).toBe('blob:memory-0');
    handle.dispose();
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:memory-0');
    await expect(handle.getLocator('url')).rejects.toThrow(
      'Asset handle "blob:local-encoder" has been disposed; blob URL locators cannot be created.',
    );
    expect(created).toHaveLength(1);
    handle.dispose();
  });
});
