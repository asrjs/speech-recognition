import { describe, expect, it, vi } from 'vitest';
import { createBlobAssetProvider, createCompositeAssetProvider } from '../src/io/providers.js';
import type { AssetProvider, AssetRequest, ResolvedAssetHandle } from '../src/types/index.js';

describe('createBlobAssetProvider', () => {
  it('resolves a Blob request as readable text', async () => {
    const provider = createBlobAssetProvider();
    const request: AssetRequest = {
      id: 'test-blob',
      blob: new Blob(['test text'], { type: 'text/plain' }),
    };

    expect(provider.canResolve(request)).toBe(true);

    const handle = await provider.resolve(request);

    expect(await handle.readText()).toBe('test text');
    expect(handle.contentType).toBe('text/plain');
  });

  it('resolves raw bytes using the requested content type', async () => {
    const provider = createBlobAssetProvider();
    const request: AssetRequest = {
      id: 'test-bytes',
      bytes: new Uint8Array([1, 2, 3]),
      contentType: 'application/octet-stream',
    };

    expect(provider.canResolve(request)).toBe(true);

    const handle = await provider.resolve(request);

    expect(Array.from(await handle.readBytes())).toEqual([1, 2, 3]);
    expect(handle.contentType).toBe('application/octet-stream');
  });

  it('resolves browser-style file handles', async () => {
    const provider = createBlobAssetProvider();
    const request: AssetRequest = {
      id: 'test-file-handle',
      fileHandle: {
        getFile: async () => new Blob(['file text'], { type: 'text/csv' }),
      },
    };

    expect(provider.canResolve(request)).toBe(true);

    const handle = await provider.resolve(request);

    expect(await handle.readText()).toBe('file text');
    expect(handle.contentType).toBe('text/csv');
  });

  it('rejects unresolvable requests', async () => {
    const provider = createBlobAssetProvider();
    const request: AssetRequest = {
      id: 'test-invalid',
      url: 'https://example.com/asset',
    };

    expect(provider.canResolve(request)).toBe(false);
    await expect(provider.resolve(request)).rejects.toThrowError(
      'Blob asset provider cannot resolve "test-invalid".',
    );
  });
});

describe('createCompositeAssetProvider', () => {
  it('reports resolvability when any child provider can resolve', () => {
    const first: AssetProvider = {
      canResolve: vi.fn().mockReturnValue(false),
      resolve: vi.fn(),
    };
    const second: AssetProvider = {
      canResolve: vi.fn().mockReturnValue(true),
      resolve: vi.fn(),
    };
    const request: AssetRequest = { id: 'test' };

    const composite = createCompositeAssetProvider([first, second]);

    expect(composite.canResolve(request)).toBe(true);
    expect(first.canResolve).toHaveBeenCalledWith(request);
    expect(second.canResolve).toHaveBeenCalledWith(request);
  });

  it('uses the first provider that can resolve a request', async () => {
    const handle = {} as ResolvedAssetHandle;
    const first: AssetProvider = {
      canResolve: vi.fn().mockReturnValue(false),
      resolve: vi.fn(),
    };
    const second: AssetProvider = {
      canResolve: vi.fn().mockReturnValue(true),
      resolve: vi.fn().mockResolvedValue(handle),
    };
    const third: AssetProvider = {
      canResolve: vi.fn().mockReturnValue(true),
      resolve: vi.fn(),
    };
    const request: AssetRequest = { id: 'test' };

    const composite = createCompositeAssetProvider([first, second, third]);

    await expect(composite.resolve(request)).resolves.toBe(handle);
    expect(second.resolve).toHaveBeenCalledWith(request);
    expect(third.resolve).not.toHaveBeenCalled();
  });

  it('throws when no provider can resolve a request', async () => {
    const provider: AssetProvider = {
      canResolve: vi.fn().mockReturnValue(false),
      resolve: vi.fn(),
    };
    const request: AssetRequest = { id: 'missing' };

    const composite = createCompositeAssetProvider([provider]);

    await expect(composite.resolve(request)).rejects.toThrowError(
      'No asset provider can resolve "missing".',
    );
    expect(provider.resolve).not.toHaveBeenCalled();
  });
});
