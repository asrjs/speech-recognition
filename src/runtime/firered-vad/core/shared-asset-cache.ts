import type { AssetCache } from '../../../types/index.js';
import type { FireRedAssetCache, FireRedAssetCacheValue } from '../types.js';

/** Routes FireRed assets through the shared ASRJS cache and reads old FireRed entries when present. */
export class SharedAssetCacheBridge implements FireRedAssetCache {
  constructor(
    private readonly shared: AssetCache,
    private readonly legacy: () => Promise<FireRedAssetCache | null>,
  ) {}

  async get(key: string): Promise<FireRedAssetCacheValue | null> {
    const value = await this.shared.get(key);
    if (value) return value;
    return (await this.legacy())?.get(key) ?? null;
  }

  async set(key: string, value: FireRedAssetCacheValue): Promise<void> {
    await this.shared.set(key, value);
  }
}
