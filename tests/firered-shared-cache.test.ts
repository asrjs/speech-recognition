import { describe, expect, it, vi } from 'vitest';

import { SharedAssetCacheBridge } from '../src/runtime/firered-vad/core/shared-asset-cache.js';
import type {
  FireRedAssetCache,
  FireRedAssetCacheValue,
} from '../src/runtime/firered-vad/types.js';

function cache(initial: Record<string, FireRedAssetCacheValue> = {}) {
  const values = new Map(Object.entries(initial));
  const get = vi.fn(async (key: string) => values.get(key) ?? null);
  const set = vi.fn(async (key: string, value: FireRedAssetCacheValue) => {
    values.set(key, value);
  });
  return { get, set, values };
}

describe('FireRed shared asset cache bridge', () => {
  it('uses the shared cache hit without consulting the legacy cache', async () => {
    const value = { bytes: new Uint8Array([1]) };
    const shared = cache({ 'firered:model.onnx': value });
    const legacy = vi.fn(async (): Promise<FireRedAssetCache | null> => null);
    const bridge = new SharedAssetCacheBridge(shared, legacy);

    expect(await bridge.get('firered:model.onnx')).toBe(value);
    expect(legacy).not.toHaveBeenCalled();
  });

  it('reads an existing legacy hit without copying it into either cache', async () => {
    const value = { bytes: new Uint8Array([2]) };
    const shared = cache();
    const old = cache({ 'firered:model.onnx': value });
    const legacy = vi.fn(async (): Promise<FireRedAssetCache> => old);
    const bridge = new SharedAssetCacheBridge(shared, legacy);

    expect(await bridge.get('firered:model.onnx')).toBe(value);
    expect(old.get).toHaveBeenCalledWith('firered:model.onnx');
    expect(shared.set).not.toHaveBeenCalled();
  });

  it('writes misses to the shared cache only, retaining the namespaced key', async () => {
    const shared = cache();
    const old = cache();
    const legacy = vi.fn(async (): Promise<FireRedAssetCache> => old);
    const bridge = new SharedAssetCacheBridge(shared, legacy);
    const value = { bytes: new Uint8Array([3]) };

    expect(await bridge.get('firered:model.onnx')).toBeNull();
    await bridge.set('firered:model.onnx', value);

    expect(shared.set).toHaveBeenCalledWith('firered:model.onnx', value);
    expect(old.set).not.toHaveBeenCalled();
  });
});
