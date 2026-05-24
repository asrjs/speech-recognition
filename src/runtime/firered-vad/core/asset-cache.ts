import type { FireRedAssetCache, FireRedAssetCacheValue } from '../types.js';

const DB_NAME = 'firered-vad-web-cache';
const STORE_NAME = 'asset-cache';
const DB_VERSION = 2;

function hasIndexedDb(): boolean {
  return typeof indexedDB !== 'undefined';
}

function isNotFoundIndexedDbError(error: unknown): boolean {
  if (typeof error !== 'object' || error === null) {
    return false;
  }
  const candidate = error as { readonly name?: unknown };
  return candidate.name === 'NotFoundError';
}

interface IdbStoreLike {
  get(key: string): { onsuccess: (() => void) | null; onerror: (() => void) | null; result?: Blob };
  put(
    value: Blob,
    key: string,
  ): { onsuccess: (() => void) | null; onerror: (() => void) | null; result?: IDBValidKey };
}

interface IdbDbLike {
  readonly version?: number;
  readonly objectStoreNames: { contains(name: string): boolean };
  createObjectStore(name: string): unknown;
  transaction(names: string[], mode: 'readonly' | 'readwrite'): { objectStore(name: string): IdbStoreLike };
  close?(): void;
}

function ensureStore(db: IdbDbLike): void {
  if (!db.objectStoreNames.contains(STORE_NAME)) {
    db.createObjectStore(STORE_NAME);
  }
}

async function openDb(version = DB_VERSION): Promise<IdbDbLike> {
  if (!hasIndexedDb()) {
    throw new Error('IndexedDB is unavailable in this environment.');
  }

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, version);
    request.onerror = () => reject(new Error('Failed to open IndexedDB.'));
    request.onupgradeneeded = () => {
      const db = request.result as unknown as IdbDbLike;
      ensureStore(db);
    };
    request.onsuccess = async () => {
      const db = request.result as unknown as IdbDbLike;
      if (db.objectStoreNames.contains(STORE_NAME)) {
        resolve(db);
        return;
      }

      db.close?.();
      try {
        const reopened = await openDb((db.version ?? version) + 1);
        resolve(reopened);
      } catch (error) {
        reject(error);
      }
    };
  });
}

export class MemoryAssetCache implements FireRedAssetCache {
  private readonly values = new Map<string, FireRedAssetCacheValue>();

  async get(key: string): Promise<FireRedAssetCacheValue | null> {
    return this.values.get(key) ?? null;
  }

  async set(key: string, value: FireRedAssetCacheValue): Promise<void> {
    this.values.set(key, value);
  }
}

export class IndexedDbAssetCache implements FireRedAssetCache {
  private dbPromise: Promise<IdbDbLike> | null = null;

  private resetDb(): void {
    this.dbPromise = null;
  }

  private getDb(): Promise<IdbDbLike> {
    if (!this.dbPromise) {
      this.dbPromise = openDb();
    }
    return this.dbPromise;
  }

  async get(key: string): Promise<FireRedAssetCacheValue | null> {
    if (!hasIndexedDb()) {
      return null;
    }
    const db = await this.getDb();
    let blob: Blob | undefined;
    try {
      blob = await new Promise<Blob | undefined>((resolve, reject) => {
        const request = db.transaction([STORE_NAME], 'readonly').objectStore(STORE_NAME).get(key);
        request.onerror = () => reject(new Error('Failed to read from IndexedDB.'));
        request.onsuccess = () => resolve(request.result);
      });
    } catch (error) {
      if (isNotFoundIndexedDbError(error)) {
        this.resetDb();
        return null;
      }
      throw error;
    }
    if (!blob) {
      return null;
    }
    const value: FireRedAssetCacheValue = {
      bytes: new Uint8Array(await blob.arrayBuffer()),
    };
    if (blob.type) {
      (value as { contentType: string }).contentType = blob.type;
    }
    return value;
  }

  async set(key: string, value: FireRedAssetCacheValue): Promise<void> {
    if (!hasIndexedDb()) {
      return;
    }
    const db = await this.getDb();
    const bytes = value.bytes.slice();
    const blob = new Blob([bytes.buffer], { type: value.contentType ?? 'application/octet-stream' });
    try {
      await new Promise<void>((resolve, reject) => {
        const request = db.transaction([STORE_NAME], 'readwrite').objectStore(STORE_NAME).put(blob, key);
        request.onerror = () => reject(new Error('Failed to write to IndexedDB.'));
        request.onsuccess = () => resolve();
      });
    } catch (error) {
      if (isNotFoundIndexedDbError(error)) {
        this.resetDb();
        return;
      }
      throw error;
    }
  }
}

let defaultBrowserCache: IndexedDbAssetCache | null = null;
const defaultMemoryCache = new MemoryAssetCache();

export function getDefaultAssetCache(preferPersistent = true): FireRedAssetCache {
  if (preferPersistent && hasIndexedDb()) {
    if (!defaultBrowserCache) {
      defaultBrowserCache = new IndexedDbAssetCache();
    }
    return defaultBrowserCache;
  }
  return defaultMemoryCache;
}
