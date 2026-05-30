import { describe, expect, it, vi } from 'vitest';
import { createCompositeAssetProvider } from '../src/io/providers.js';
import type { AssetProvider, AssetRequest, ResolvedAssetHandle } from '../src/types/index.js';

describe('createCompositeAssetProvider', () => {
  it('should return true for canResolve if any provider can resolve', () => {
    const provider1: AssetProvider = {
      canResolve: vi.fn().mockReturnValue(false),
      resolve: vi.fn(),
    };
    const provider2: AssetProvider = {
      canResolve: vi.fn().mockReturnValue(true),
      resolve: vi.fn(),
    };

    const composite = createCompositeAssetProvider([provider1, provider2]);
    const request: AssetRequest = { id: 'test' };

    expect(composite.canResolve(request)).toBe(true);
    expect(provider1.canResolve).toHaveBeenCalledWith(request);
    expect(provider2.canResolve).toHaveBeenCalledWith(request);
  });

  it('should return false for canResolve if no provider can resolve', () => {
    const provider1: AssetProvider = {
      canResolve: vi.fn().mockReturnValue(false),
      resolve: vi.fn(),
    };

    const composite = createCompositeAssetProvider([provider1]);
    const request: AssetRequest = { id: 'test' };

    expect(composite.canResolve(request)).toBe(false);
    expect(provider1.canResolve).toHaveBeenCalledWith(request);
  });

  it('should resolve using the first provider that can resolve', async () => {
    const handle = {} as ResolvedAssetHandle;
    const provider1: AssetProvider = {
      canResolve: vi.fn().mockReturnValue(false),
      resolve: vi.fn(),
    };
    const provider2: AssetProvider = {
      canResolve: vi.fn().mockReturnValue(true),
      resolve: vi.fn().mockResolvedValue(handle),
    };
    const provider3: AssetProvider = {
      canResolve: vi.fn().mockReturnValue(true),
      resolve: vi.fn(),
    };

    const composite = createCompositeAssetProvider([provider1, provider2, provider3]);
    const request: AssetRequest = { id: 'test' };

    const result = await composite.resolve(request);

    expect(result).toBe(handle);
    expect(provider1.canResolve).toHaveBeenCalledWith(request);
    expect(provider2.canResolve).toHaveBeenCalledWith(request);
    expect(provider2.resolve).toHaveBeenCalledWith(request);
    expect(provider3.resolve).not.toHaveBeenCalled();
  });

  it('should throw an error if no provider can resolve during resolve', async () => {
    const provider1: AssetProvider = {
      canResolve: vi.fn().mockReturnValue(false),
      resolve: vi.fn(),
    };

    const composite = createCompositeAssetProvider([provider1]);
    const request: AssetRequest = { id: 'test-id' };

    await expect(composite.resolve(request)).rejects.toThrowError(
      'No asset provider can resolve "test-id".',
    );
    expect(provider1.canResolve).toHaveBeenCalledWith(request);
    expect(provider1.resolve).not.toHaveBeenCalled();
  });
});
