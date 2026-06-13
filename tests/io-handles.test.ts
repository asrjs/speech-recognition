import { afterEach, describe, expect, it, vi } from 'vitest';

import { UrlAssetHandle } from '../src/io/handles.js';
import type { AssetCache } from '../src/types/index.js';

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

  it('continues with network when cache read throws and evicts the broken key', async () => {
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

  it('can materialize remote URL locators as blob URLs with progress', async () => {
    const progressEvents: number[] = [];
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
          progressEvents.push(event.loaded);
        },
      },
      'https://huggingface.co/ysdede/parakeet-tdt-0.6b-v3-onnx/resolve/feat%2Ffp16-canonical-v3/encoder-model.fp16.onnx',
    );

    const locator = await handle.getLocator('url');
    expect(locator).toMatch(/^blob:/);
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    expect(progressEvents[progressEvents.length - 1]).toBe(4);
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
});
