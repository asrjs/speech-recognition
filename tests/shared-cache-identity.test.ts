import { afterEach, describe, expect, it, vi } from 'vitest';

import { UrlAssetHandle } from '../src/io/handles.js';
import type { AssetCache, AssetCacheValue } from '../src/types/index.js';

class MemoryAssetCache implements AssetCache {
  readonly values = new Map<string, AssetCacheValue>();
  readonly blobs = new Map<string, Blob>();
  readonly writes: string[] = [];
  readonly blobWrites: string[] = [];
  readonly reads: string[] = [];

  async get(key: string): Promise<AssetCacheValue | null> {
    this.reads.push(key);
    return this.values.get(key) ?? null;
  }

  async set(key: string, value: AssetCacheValue): Promise<void> {
    this.writes.push(key);
    this.values.set(key, value);
  }

  async getBlob(key: string): Promise<Blob | null> {
    this.reads.push(key);
    return this.blobs.get(key) ?? null;
  }

  async setBlob(key: string, blob: Blob): Promise<void> {
    this.blobWrites.push(key);
    this.blobs.set(key, blob);
  }
}

const originalFetch = globalThis.fetch;

afterEach(() => {
  vi.restoreAllMocks();
  globalThis.fetch = originalFetch;
});

describe('shared Hugging Face cache identity', () => {
  it('reuses one family-written main artifact through the runtime helper key', async () => {
    const cache = new MemoryAssetCache();
    const fetchSpy = vi.fn(async () => new Response(new Uint8Array([1, 2, 3])));
    globalThis.fetch = fetchSpy as typeof fetch;
    const url = 'https://huggingface.co/acme/asr/resolve/main/encoder.onnx';
    const family = new UrlAssetHandle(
      {
        id: 'family',
        provider: 'huggingface',
        repoId: 'acme/asr',
        filename: 'encoder.onnx',
        cacheKey: 'huggingface:acme/asr:main:encoder.onnx',
      },
      url,
      cache,
    );
    const helper = new UrlAssetHandle(
      {
        id: 'helper',
        provider: 'huggingface',
        repoId: 'acme/asr',
        filename: 'encoder.onnx',
        cacheKey: 'hf-acme/asr-main--encoder.onnx',
      },
      url,
      cache,
    );

    expect(Array.from(await family.readBytes())).toEqual([1, 2, 3]);
    expect(Array.from(await helper.readBytes())).toEqual([1, 2, 3]);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(cache.writes).toEqual(['hf-asset-v1:["acme/asr","main","encoder.onnx"]']);
  });

  it('unifies nested files across family, helper, and exact URL request shapes', async () => {
    const cache = new MemoryAssetCache();
    const fetchSpy = vi.fn(async () => new Response(new Uint8Array([2, 4, 6])));
    globalThis.fetch = fetchSpy as typeof fetch;
    const url = 'https://huggingface.co/acme/asr/resolve/main/onnx/encoder.onnx';
    const family = new UrlAssetHandle(
      {
        id: 'family',
        provider: 'huggingface',
        repoId: 'acme/asr',
        filename: 'onnx/encoder.onnx',
        cacheKey: 'huggingface:acme/asr:main:onnx/encoder.onnx',
      },
      url,
      cache,
    );
    const helper = new UrlAssetHandle(
      {
        id: 'helper',
        provider: 'huggingface',
        repoId: 'acme/asr',
        subfolder: 'onnx',
        filename: 'encoder.onnx',
        cacheKey: 'hf-acme/asr-main-onnx-encoder.onnx',
      },
      url,
      cache,
    );
    const urlRequest = new UrlAssetHandle(
      { id: 'whisper-url', provider: 'url', url, cacheKey: `url:${url}` },
      url,
      cache,
    );

    expect(Array.from(await family.readBytes())).toEqual([2, 4, 6]);
    expect(Array.from(await helper.readBytes())).toEqual([2, 4, 6]);
    expect(Array.from(await urlRequest.readBytes())).toEqual([2, 4, 6]);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(cache.writes).toEqual(['hf-asset-v1:["acme/asr","main","onnx/encoder.onnx"]']);
  });

  it('reuses a runtime-helper-written main artifact through the family key', async () => {
    const cache = new MemoryAssetCache();
    const fetchSpy = vi.fn(async () => new Response(new Uint8Array([3, 2, 1])));
    globalThis.fetch = fetchSpy as typeof fetch;
    const url = 'https://huggingface.co/acme/asr/resolve/main/decoder.onnx';
    const helper = new UrlAssetHandle(
      {
        id: 'helper',
        provider: 'huggingface',
        repoId: 'acme/asr',
        filename: 'decoder.onnx',
        cacheKey: 'hf-acme/asr-main--decoder.onnx',
      },
      url,
      cache,
    );
    const family = new UrlAssetHandle(
      {
        id: 'family',
        provider: 'huggingface',
        repoId: 'acme/asr',
        filename: 'decoder.onnx',
        cacheKey: 'huggingface:acme/asr:main:decoder.onnx',
      },
      url,
      cache,
    );

    expect(Array.from(await helper.readBytes())).toEqual([3, 2, 1]);
    expect(Array.from(await family.readBytes())).toEqual([3, 2, 1]);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(cache.writes).toHaveLength(1);
  });

  it('reads a legacy family row without copying it to the canonical key', async () => {
    const cache = new MemoryAssetCache();
    const legacyKey = 'huggingface:acme/asr:main:encoder.onnx';
    cache.values.set(legacyKey, { bytes: new Uint8Array([4, 5]), contentType: 'application/onnx' });
    const fetchSpy = vi.fn();
    globalThis.fetch = fetchSpy as unknown as typeof fetch;
    const handle = new UrlAssetHandle(
      {
        id: 'helper',
        provider: 'huggingface',
        repoId: 'acme/asr',
        filename: 'encoder.onnx',
        cacheKey: 'hf-acme/asr-main--encoder.onnx',
      },
      'https://huggingface.co/acme/asr/resolve/main/encoder.onnx',
      cache,
    );

    expect(Array.from(await handle.readBytes())).toEqual([4, 5]);
    expect(cache.writes).toEqual([]);
    expect(cache.values.has(legacyKey)).toBe(true);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it.each([
    'huggingface:acme/asr:main:onnx/encoder.onnx',
    'huggingface:acme/asr:main:onnx:encoder.onnx',
  ])('reads legacy nested family spelling %s without copying', async (legacyKey) => {
    const cache = new MemoryAssetCache();
    cache.values.set(legacyKey, { bytes: new Uint8Array([8, 9]) });
    const fetchSpy = vi.fn();
    globalThis.fetch = fetchSpy as unknown as typeof fetch;
    const url = 'https://huggingface.co/acme/asr/resolve/main/onnx/encoder.onnx';
    const handle = new UrlAssetHandle(
      {
        id: 'helper',
        provider: 'huggingface',
        repoId: 'acme/asr',
        subfolder: 'onnx',
        filename: 'encoder.onnx',
        cacheKey: 'hf-acme/asr-main-onnx-encoder.onnx',
      },
      url,
      cache,
    );

    expect(Array.from(await handle.readBytes())).toEqual([8, 9]);
    expect(cache.writes).toEqual([]);
    expect(cache.values.has(legacyKey)).toBe(true);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('migrates a recognized alias when explicitly supplied as a caller fallback', async () => {
    const cache = new MemoryAssetCache();
    const legacyKey = 'huggingface:acme/asr:main:encoder.onnx';
    cache.values.set(legacyKey, { bytes: new Uint8Array([5, 4]) });
    const fetchSpy = vi.fn();
    globalThis.fetch = fetchSpy as unknown as typeof fetch;
    const handle = new UrlAssetHandle(
      {
        id: 'helper',
        provider: 'huggingface',
        repoId: 'acme/asr',
        filename: 'encoder.onnx',
        cacheKey: 'hf-acme/asr-main--encoder.onnx',
        cacheKeyFallbacks: [legacyKey],
      },
      'https://huggingface.co/acme/asr/resolve/main/encoder.onnx',
      cache,
    );

    expect(Array.from(await handle.readBytes())).toEqual([5, 4]);
    expect(cache.writes).toEqual(['hf-asset-v1:["acme/asr","main","encoder.onnx"]']);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('migrates an explicitly requested legacy blob fallback', async () => {
    const cache = new MemoryAssetCache();
    const url = 'https://huggingface.co/acme/asr/resolve/main/encoder.onnx';
    const legacyKey = `url:${url}`;
    const blob = new Blob([new Uint8Array([2, 3])]);
    cache.blobs.set(legacyKey, blob);
    const fetchSpy = vi.fn();
    globalThis.fetch = fetchSpy as unknown as typeof fetch;
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:migrated-asset');
    const handle = new UrlAssetHandle(
      {
        id: 'helper',
        provider: 'huggingface',
        repoId: 'acme/asr',
        filename: 'encoder.onnx',
        cacheKey: 'hf-acme/asr-main--encoder.onnx',
        cacheKeyFallbacks: [legacyKey],
        preferBlobUrl: true,
      },
      url,
      cache,
    );

    expect(await handle.getLocator('url')).toBe('blob:migrated-asset');
    expect(cache.blobWrites).toEqual(['hf-asset-v1:["acme/asr","main","encoder.onnx"]']);
    expect(cache.blobs.get(legacyKey)).toBe(blob);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('reads a trusted Whisper URL blob alias without copying it', async () => {
    const cache = new MemoryAssetCache();
    const url = 'https://huggingface.co/acme/asr/resolve/main/encoder.onnx';
    const blob = new Blob([new Uint8Array([7, 8])], { type: 'application/onnx' });
    cache.blobs.set(`url:${url}`, blob);
    const fetchSpy = vi.fn();
    globalThis.fetch = fetchSpy as unknown as typeof fetch;
    const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:cached-asset');
    const handle = new UrlAssetHandle(
      { id: `url:${url}`, provider: 'url', url, cacheKey: `url:${url}`, preferBlobUrl: true },
      url,
      cache,
    );

    expect(await handle.getLocator('url')).toBe('blob:cached-asset');
    expect(cache.blobWrites).toEqual([]);
    expect(cache.blobs.get(`url:${url}`)).toBe(blob);
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(createObjectURL).toHaveBeenCalledWith(blob);
  });

  it('keeps signed/query URLs opaque instead of stripping URL parameters', async () => {
    const cache = new MemoryAssetCache();
    const url = 'https://huggingface.co/acme/asr/resolve/main/encoder.onnx?token=secret';
    const fetchSpy = vi.fn(async () => new Response(new Uint8Array([6])));
    globalThis.fetch = fetchSpy as typeof fetch;
    const handle = new UrlAssetHandle(
      { id: 'signed', provider: 'url', url, cacheKey: `url:${url}` },
      url,
      cache,
    );

    await handle.readBytes();
    expect(cache.writes).toEqual([`url:${url}`]);
    expect(cache.reads[0]).toBe(`url:${url}`);
    expect(cache.reads).not.toContain('hf-asset-v1:["acme/asr","main","encoder.onnx"]');
  });

  it('leaves custom keys, strict revisions, quantizations, and shards isolated', async () => {
    const cache = new MemoryAssetCache();
    const fetchSpy = vi.fn(async () => new Response(new Uint8Array([9])));
    globalThis.fetch = fetchSpy as typeof fetch;
    const base = 'https://huggingface.co/acme/asr/resolve/main/';
    const custom = new UrlAssetHandle(
      {
        id: 'custom',
        provider: 'huggingface',
        repoId: 'acme/asr',
        filename: 'encoder.onnx',
        cacheKey: 'caller-owned-key',
      },
      `${base}encoder.onnx`,
      cache,
    );
    await custom.readBytes();
    expect(cache.writes.at(-1)).toBe('caller-owned-key');

    cache.reads.length = 0;
    const pinned = new UrlAssetHandle(
      {
        id: 'pinned',
        provider: 'huggingface',
        repoId: 'acme/asr',
        filename: 'encoder.onnx',
        revision: 'release/1',
        cacheKey: 'huggingface:acme/asr:release/1:encoder.onnx',
      },
      'https://huggingface.co/acme/asr/resolve/release%2F1/encoder.onnx',
      cache,
    );
    await pinned.readBytes();
    expect(cache.reads).toContain('huggingface:acme/asr:release/1:encoder.onnx');
    expect(cache.reads).not.toContain('hf-asset-v1:["acme/asr","main","encoder.onnx"]');

    const files = ['encoder.onnx', 'encoder.int8.onnx', 'encoder.onnx.data'];
    for (const filename of files) {
      await new UrlAssetHandle(
        {
          id: filename,
          provider: 'huggingface',
          repoId: 'acme/asr',
          filename,
          cacheKey: `huggingface:acme/asr:main:${filename}`,
        },
        `${base}${filename}`,
        cache,
      ).readBytes();
    }
    const identities = cache.writes.filter((key) => key.startsWith('hf-asset-v1:'));
    expect(new Set(identities).size).toBe(3);
  });
});
