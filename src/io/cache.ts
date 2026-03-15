import type { AssetCache, AssetCacheValue } from '../types/index.js';

interface IdbDatabaseLike {
  readonly version?: number;
  readonly objectStoreNames: {
    contains(name: string): boolean;
  };
  createObjectStore(name: string): unknown;
  transaction(names: string[], mode: 'readonly' | 'readwrite'): IdbTransactionLike;
  close?(): void;
}

interface IdbTransactionLike {
  objectStore(name: string): IdbObjectStoreLike;
}

interface IdbObjectStoreLike {
  get(key: string): { onsuccess: (() => void) | null; onerror: (() => void) | null; result?: Blob };
  put(
    blob: Blob,
    key: string,
  ): { onsuccess: (() => void) | null; onerror: (() => void) | null; result?: IDBValidKey };
}

function hasIndexedDb(): boolean {
  return typeof indexedDB !== 'undefined';
}

const CACHE_DB_NAME = 'asrjs-cache-db';
const CACHE_STORE_NAME = 'asset-cache';
const CACHE_DB_VERSION = 2;

function ensureStore(db: IdbDatabaseLike): void {
  if (!db.objectStoreNames.contains(CACHE_STORE_NAME)) {
    db.createObjectStore(CACHE_STORE_NAME);
  }
}

async function openDb(version = CACHE_DB_VERSION): Promise<IdbDatabaseLike> {
  if (!hasIndexedDb()) {
    throw new Error('IndexedDB is unavailable in this environment.');
  }

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(CACHE_DB_NAME, version);
    request.onerror = () => reject(new Error('Error opening IndexedDB.'));
    request.onsuccess = async () => {
      const db = request.result as unknown as IdbDatabaseLike;
      if (db.objectStoreNames.contains(CACHE_STORE_NAME)) {
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
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result as unknown as IdbDatabaseLike;
      ensureStore(db);
    };
  });
}

export class IndexedDbAssetCache implements AssetCache {
  private dbPromise: Promise<IdbDatabaseLike> | null = null;

  private resetDb(): void {
    this.dbPromise = null;
  }

  private getDb(): Promise<IdbDatabaseLike> {
    if (!this.dbPromise) {
      this.dbPromise = openDb();
    }

    return this.dbPromise;
  }

  async get(key: string): Promise<AssetCacheValue | null> {
    if (!hasIndexedDb()) {
      return null;
    }

    const db = await this.getDb();
    let blob: Blob | undefined;
    try {
      blob = await new Promise<Blob | undefined>((resolve, reject) => {
        const transaction = db.transaction([CACHE_STORE_NAME], 'readonly');
        const store = transaction.objectStore(CACHE_STORE_NAME);
        const request = store.get(key);
        request.onerror = () => reject(new Error('Error reading from IndexedDB.'));
        request.onsuccess = () => resolve(request.result);
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === 'NotFoundError') {
        this.resetDb();
        return null;
      }
      throw error;
    }

    if (!blob) {
      return null;
    }

    return {
      bytes: new Uint8Array(await blob.arrayBuffer()),
      contentType: blob.type || undefined,
    };
  }

  async set(key: string, value: AssetCacheValue): Promise<void> {
    if (!hasIndexedDb()) {
      return;
    }

    const db = await this.getDb();
    const blob = new Blob([value.bytes.slice().buffer], {
      type: value.contentType ?? 'application/octet-stream',
    });

    try {
      await new Promise<void>((resolve, reject) => {
        const transaction = db.transaction([CACHE_STORE_NAME], 'readwrite');
        const store = transaction.objectStore(CACHE_STORE_NAME);
        const request = store.put(blob, key);
        request.onerror = () => reject(new Error('Error writing to IndexedDB.'));
        request.onsuccess = () => resolve();
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === 'NotFoundError') {
        this.resetDb();
        return;
      }
      throw error;
    }
  }
}

export class MemoryAssetCache implements AssetCache {
  private readonly values = new Map<string, AssetCacheValue>();

  async get(key: string): Promise<AssetCacheValue | null> {
    return this.values.get(key) ?? null;
  }

  async set(key: string, value: AssetCacheValue): Promise<void> {
    this.values.set(key, value);
  }
}

let defaultIndexedDbCache: IndexedDbAssetCache | null = null;

export function getDefaultIndexedDbAssetCache(): IndexedDbAssetCache | null {
  if (!hasIndexedDb()) {
    return null;
  }

  if (!defaultIndexedDbCache) {
    defaultIndexedDbCache = new IndexedDbAssetCache();
  }

  return defaultIndexedDbCache;
}
