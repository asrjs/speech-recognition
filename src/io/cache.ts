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
  delete(key: string): { onsuccess: (() => void) | null; onerror: (() => void) | null };
}

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
    const blob = await this.getBlob(key);
    if (!blob) {
      return null;
    }

    try {
      return {
        bytes: new Uint8Array(await blob.arrayBuffer()),
        contentType: blob.type || undefined,
      };
    } catch (error) {
      if (isNotFoundIndexedDbError(error)) {
        await this.delete(key);
        return null;
      }
      throw error;
    }
  }

  async getBlob(key: string): Promise<Blob | null> {
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
      if (isNotFoundIndexedDbError(error)) {
        this.resetDb();
        return null;
      }
      throw error;
    }

    if (!blob) {
      return null;
    }

    return blob;
  }

  async set(key: string, value: AssetCacheValue): Promise<void> {
    await this.setBlob(
      key,
      new Blob([value.bytes.slice().buffer], {
        type: value.contentType ?? 'application/octet-stream',
      }),
    );
  }

  async setBlob(key: string, blob: Blob): Promise<void> {
    if (!hasIndexedDb()) {
      return;
    }

    const db = await this.getDb();

    try {
      await new Promise<void>((resolve, reject) => {
        const transaction = db.transaction([CACHE_STORE_NAME], 'readwrite');
        const store = transaction.objectStore(CACHE_STORE_NAME);
        const request = store.put(blob, key);
        request.onerror = () => reject(new Error('Error writing to IndexedDB.'));
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

  async delete(key: string): Promise<void> {
    if (!hasIndexedDb()) {
      return;
    }

    const db = await this.getDb();
    try {
      await new Promise<void>((resolve, reject) => {
        const transaction = db.transaction([CACHE_STORE_NAME], 'readwrite');
        const store = transaction.objectStore(CACHE_STORE_NAME);
        const request = store.delete(key);
        request.onerror = () => reject(new Error('Error deleting from IndexedDB.'));
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

export class MemoryAssetCache implements AssetCache {
  private readonly values = new Map<string, AssetCacheValue>();
  private readonly blobs = new Map<string, Blob>();

  async get(key: string): Promise<AssetCacheValue | null> {
    const value = this.values.get(key);
    if (value) return value;
    const blob = this.blobs.get(key);
    if (!blob) return null;
    return {
      bytes: new Uint8Array(await blob.arrayBuffer()),
      contentType: blob.type || undefined,
    };
  }

  async set(key: string, value: AssetCacheValue): Promise<void> {
    this.values.set(key, value);
    this.blobs.delete(key);
  }

  async getBlob(key: string): Promise<Blob | null> {
    const blob = this.blobs.get(key);
    if (blob) return blob;
    const value = this.values.get(key);
    if (!value) return null;
    return new Blob([value.bytes.slice().buffer], {
      type: value.contentType ?? 'application/octet-stream',
    });
  }

  async setBlob(key: string, blob: Blob): Promise<void> {
    this.blobs.set(key, blob);
    this.values.delete(key);
  }

  async delete(key: string): Promise<void> {
    this.values.delete(key);
    this.blobs.delete(key);
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
