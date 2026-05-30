import { describe, expect, it } from 'vitest';
import { createBlobAssetProvider } from '../src/io/providers.js';

describe('createBlobAssetProvider', () => {
  it('resolves a Blob to a valid BlobAssetHandle that can be read', async () => {
    const provider = createBlobAssetProvider();

    const request: any = {
      id: 'test-blob',
      blob: new Blob(['test text'], { type: 'text/plain' })
    };

    expect(provider.canResolve(request)).toBe(true);

    const handle = await provider.resolve(request);
    expect(handle).toBeDefined();

    expect(await handle.readText()).toBe('test text');
    expect(handle.contentType).toBe('text/plain');
  });

  it('resolves raw bytes to a BlobAssetHandle', async () => {
    const provider = createBlobAssetProvider();

    const request: any = {
      id: 'test-bytes',
      bytes: new Uint8Array([1, 2, 3]),
      contentType: 'application/octet-stream'
    };

    expect(provider.canResolve(request)).toBe(true);

    const handle = await provider.resolve(request);

    const readBytes = await handle.readBytes();
    expect(Array.from(readBytes)).toEqual([1, 2, 3]);
  });

  it('resolves fileHandle returning a Blob to a BlobAssetHandle', async () => {
    const provider = createBlobAssetProvider();

    const request: any = {
      id: 'test-file-handle-blob',
      fileHandle: {
        getFile: async () => new Blob(['file text'], { type: 'text/csv' })
      }
    };

    expect(provider.canResolve(request)).toBe(true);

    const handle = await provider.resolve(request);

    expect(await handle.readText()).toBe('file text');
    expect(handle.contentType).toBe('text/csv');
  });

  it('resolves fileHandle returning raw data to a BlobAssetHandle', async () => {
    const provider = createBlobAssetProvider();

    const request: any = {
      id: 'test-file-handle-buffer',
      fileHandle: {
        getFile: async () => new Uint8Array([7, 8, 9])
      },
      contentType: 'application/custom'
    };

    expect(provider.canResolve(request)).toBe(true);

    const handle = await provider.resolve(request);

    const readBytes = await handle.readBytes();
    expect(Array.from(readBytes)).toEqual([7, 8, 9]);
    expect(handle.contentType).toBe('application/custom');
  });

  it('rejects unresolvable requests', async () => {
    const provider = createBlobAssetProvider();

    const request: any = {
      id: 'test-invalid',
      url: 'https://example.com/asset'
    };

    expect(provider.canResolve(request)).toBe(false);

    await expect(provider.resolve(request)).rejects.toThrowError(
      'Blob asset provider cannot resolve "test-invalid".'
    );
  });
});
