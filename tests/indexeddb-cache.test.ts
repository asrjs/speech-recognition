import { afterEach, describe, expect, it } from 'vitest';

import { IndexedDbAssetCache } from '../src/io/cache.js';

type OpenRequestLike = {
  result: unknown;
  onsuccess: ((event: unknown) => void) | null;
  onerror: (() => void) | null;
  onupgradeneeded: ((event: unknown) => void) | null;
};

type FakeStore = {
  get: (key: string) => {
    onsuccess: (() => void) | null;
    onerror: (() => void) | null;
    result?: Blob;
  };
  put: (
    blob: Blob,
    key: string,
  ) => { onsuccess: (() => void) | null; onerror: (() => void) | null; result?: IDBValidKey };
};

const originalIndexedDb = globalThis.indexedDB;

function createEmptyRequest(result: unknown): OpenRequestLike {
  return {
    result,
    onsuccess: null,
    onerror: null,
    onupgradeneeded: null,
  };
}

function flushAsync(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

afterEach(() => {
  if (originalIndexedDb) {
    globalThis.indexedDB = originalIndexedDb;
  } else {
    delete (globalThis as Record<string, unknown>).indexedDB;
  }
});

describe('IndexedDbAssetCache', () => {
  it('recovers from a stale database that is missing the asset-cache store', async () => {
    let hasStore = false;
    const store: FakeStore = {
      get: () => {
        const request = {
          onsuccess: null as (() => void) | null,
          onerror: null as (() => void) | null,
          result: undefined as Blob | undefined,
        };
        queueMicrotask(() => {
          request.onsuccess?.();
        });
        return request;
      },
      put: () => {
        const request = {
          onsuccess: null as (() => void) | null,
          onerror: null as (() => void) | null,
          result: 'ok' as IDBValidKey,
        };
        queueMicrotask(() => {
          request.onsuccess?.();
        });
        return request;
      },
    };

    const db = {
      version: 2,
      objectStoreNames: {
        contains(name: string) {
          return name === 'asset-cache' ? hasStore : false;
        },
      },
      createObjectStore(name: string) {
        if (name === 'asset-cache') {
          hasStore = true;
        }
        return {};
      },
      transaction(names: string[]) {
        if (!hasStore || !names.includes('asset-cache')) {
          throw new DOMException('missing store', 'NotFoundError');
        }
        return {
          objectStore() {
            return store;
          },
        };
      },
      close() {
        return undefined;
      },
    };

    const versions: number[] = [];
    globalThis.indexedDB = {
      open(_name: string, version?: number) {
        versions.push(version ?? 0);
        const request = createEmptyRequest(db);
        queueMicrotask(() => {
          if (versions.length === 2) {
            request.onupgradeneeded?.({ target: { result: db } });
          }
          request.onsuccess?.({});
        });
        return request as unknown as IDBOpenDBRequest;
      },
    } as IDBFactory;

    const cache = new IndexedDbAssetCache();
    const value = await cache.get('missing-key');
    await flushAsync();

    expect(value).toBeNull();
    expect(versions).toEqual([2, 3]);
  });
});
