import {
  isAssetLoadAbortedError,
  throwIfAssetAborted,
  toAssetLoadAbortedError,
  toFetchAbortSignal,
  throwAssetLoadAborted,
} from './abort.js';
import type {
  AssetCache,
  AssetCacheValue,
  AssetProgressEvent,
  AssetRequest,
  ResolvedAssetHandle,
} from '../types/index.js';

interface CacheReadHit {
  readonly key: string;
  readonly value: AssetCacheValue;
}

interface BlobCacheReadHit {
  readonly key: string;
  readonly blob: Blob;
}

function emitAssetProgress(
  request: AssetRequest,
  event: AssetProgressEvent,
): void {
  if (event.aborted) {
    request.onProgress?.({
      id: request.id,
      ...event,
      done: true,
      aborted: true,
    });
    return;
  }
  if (request.signal?.aborted) {
    return;
  }
  request.onProgress?.({
    id: request.id,
    ...event,
  });
}

async function* bytesToStream(bytes: Uint8Array): AsyncIterable<Uint8Array> {
  yield bytes;
}

async function* readReadableStream(
  stream: ReadableStream<Uint8Array>,
  signal?: AssetRequest['signal'],
): AsyncIterable<Uint8Array> {
  const reader = stream.getReader();

  try {
    while (true) {
      throwIfAssetAborted(signal, 'download');
      const { done, value } = await reader.read();
      if (done) {
        return;
      }
      throwIfAssetAborted(signal, 'download');
      if (value) {
        yield value;
      }
    }
  } catch (error) {
    try {
      await reader.cancel();
    } catch {
      // already closed or cancelled
    }
    throwAssetLoadAborted(error, 'download');
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // cancel() already released the lock
    }
  }
}

async function streamToBytes(
  iterable: AsyncIterable<Uint8Array>,
  onChunk?: (chunk: Uint8Array, loaded: number) => void,
): Promise<Uint8Array> {
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  for await (const chunk of iterable) {
    chunks.push(chunk);
    loaded += chunk.byteLength;
    onChunk?.(chunk, loaded);
  }

  if (chunks.length === 1) {
    return chunks[0]!;
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }

  return bytes;
}

async function streamToBlob(
  iterable: AsyncIterable<Uint8Array>,
  contentType: string | undefined,
  onChunk?: (chunk: Uint8Array, loaded: number) => void,
): Promise<Blob> {
  const chunks: BlobPart[] = [];
  let loaded = 0;

  for await (const chunk of iterable) {
    chunks.push(chunk.slice().buffer as ArrayBuffer);
    loaded += chunk.byteLength;
    onChunk?.(chunk, loaded);
  }

  return new Blob(chunks, {
    type: contentType ?? 'application/octet-stream',
  });
}

async function readCache(
  cache: AssetCache | undefined,
  key: string | undefined,
  fallbackKeys: readonly string[] = [],
): Promise<CacheReadHit | null> {
  if (!cache || !key) {
    return null;
  }

  const keys = [key, ...fallbackKeys.filter((candidate) => candidate && candidate !== key)];
  for (const candidateKey of keys) {
    try {
      const value = await cache.get(candidateKey);
      if (value) {
        return { key: candidateKey, value };
      }
    } catch (error) {
      console.warn(
        `[assets] Cache read failed for "${candidateKey}". Falling back to network.`,
        error,
      );
      try {
        await cache.delete?.(candidateKey);
      } catch {
        // best-effort cache eviction only
      }
    }
  }

  return null;
}

async function writeCache(
  cache: AssetCache | undefined,
  key: string | undefined,
  value: AssetCacheValue,
): Promise<void> {
  if (!cache || !key) {
    return;
  }

  try {
    await cache.set(key, value);
  } catch (error) {
    console.warn(`[assets] Cache write failed for "${key}". Continuing without cache.`, error);
  }
}

async function readBlobCache(
  cache: AssetCache | undefined,
  key: string | undefined,
  fallbackKeys: readonly string[] = [],
): Promise<BlobCacheReadHit | null> {
  if (!cache?.getBlob || !key) {
    return null;
  }

  const keys = [key, ...fallbackKeys.filter((candidate) => candidate && candidate !== key)];
  for (const candidateKey of keys) {
    try {
      const blob = await cache.getBlob(candidateKey);
      if (blob) {
        return { key: candidateKey, blob };
      }
    } catch (error) {
      console.warn(
        `[assets] Blob cache read failed for "${candidateKey}". Falling back to network.`,
        error,
      );
      try {
        await cache.delete?.(candidateKey);
      } catch {
        // best-effort cache eviction only
      }
    }
  }

  return null;
}

async function writeBlobCache(
  cache: AssetCache | undefined,
  key: string | undefined,
  blob: Blob,
): Promise<void> {
  if (!cache?.setBlob || !key) {
    return;
  }

  try {
    await cache.setBlob(key, blob);
  } catch (error) {
    console.warn(`[assets] Blob cache write failed for "${key}". Continuing without cache.`, error);
  }
}

async function migrateBlobCacheHit(
  cache: AssetCache | undefined,
  primaryKey: string | undefined,
  hit: BlobCacheReadHit,
): Promise<void> {
  if (!cache?.setBlob || !primaryKey || hit.key === primaryKey) {
    return;
  }

  await writeBlobCache(cache, primaryKey, hit.blob);
}

async function migrateCacheHit(
  cache: AssetCache | undefined,
  primaryKey: string | undefined,
  hit: CacheReadHit,
): Promise<void> {
  if (!cache || !primaryKey || hit.key === primaryKey) {
    return;
  }

  await writeCache(cache, primaryKey, hit.value);
}

function formatRepoPath(repoId: string): string {
  return String(repoId || '')
    .split('/')
    .map((part) => encodeURIComponent(part))
    .join('/');
}

function buildHuggingFaceResolveUrl(
  request: AssetRequest,
  revisionOverride?: string,
): string | null {
  if (!request.repoId || !request.filename) {
    return null;
  }

  const revision = revisionOverride ?? request.revision ?? 'main';
  const encodedRevision = encodeURIComponent(revision);
  const encodedSubfolder = request.subfolder
    ? request.subfolder
        .split('/')
        .map((part) => encodeURIComponent(part))
        .join('/')
    : '';
  const encodedFilename = request.filename
    .split('/')
    .map((part) => encodeURIComponent(part))
    .join('/');

  const pathParts = [formatRepoPath(request.repoId), 'resolve', encodedRevision];
  if (encodedSubfolder) {
    pathParts.push(encodedSubfolder);
  }
  pathParts.push(encodedFilename);
  return `https://huggingface.co/${pathParts.join('/')}`;
}

function getFetchCandidateUrls(request: AssetRequest, primaryUrl: string): readonly string[] {
  const urls = [primaryUrl];
  if (request.provider !== 'huggingface') {
    return urls;
  }

  const revision = request.revision ?? 'main';
  if (revision === 'main') {
    return urls;
  }

  const fallbackUrl = buildHuggingFaceResolveUrl(request, 'main');
  if (fallbackUrl && fallbackUrl !== primaryUrl) {
    urls.push(fallbackUrl);
  }
  return urls;
}

async function fetchWithCandidates(request: AssetRequest, primaryUrl: string): Promise<Response> {
  const candidateUrls = getFetchCandidateUrls(request, primaryUrl);
  let lastStatus: number | null = null;
  let lastStatusText = '';
  const fetchSignal = toFetchAbortSignal(request.signal);
  const fetchInit = fetchSignal ? { signal: fetchSignal } : undefined;

  for (let index = 0; index < candidateUrls.length; index += 1) {
    throwIfAssetAborted(request.signal, 'download');
    const candidateUrl = candidateUrls[index]!;
    let response: Response;
    try {
      response = await fetch(candidateUrl, fetchInit);
    } catch (error) {
      throwAssetLoadAborted(error, 'download');
    }
    if (response.ok) {
      return response;
    }

    lastStatus = response.status;
    lastStatusText = response.statusText;
    const hasNext = index < candidateUrls.length - 1;
    if (!hasNext || response.status !== 404) {
      break;
    }
  }

  throw new Error(
    `Failed to fetch asset "${request.id}": ${lastStatus ?? 0} ${lastStatusText}`.trim(),
  );
}

function createDisposedAssetHandleError(request: AssetRequest): Error {
  return new Error(
    `Asset handle "${request.id}" has been disposed; blob URL locators cannot be created.`,
  );
}

export class BlobAssetHandle implements ResolvedAssetHandle {
  private locatorUrl: string | null = null;
  private disposed = false;

  constructor(
    readonly request: AssetRequest,
    private readonly blob: Blob,
  ) {}

  get contentType(): string | undefined {
    return this.blob.type || this.request.contentType || undefined;
  }

  get sizeBytes(): number | undefined {
    return this.blob.size;
  }

  openStream(): AsyncIterable<Uint8Array> {
    const stream = this.blob.stream();
    return readReadableStream(stream as ReadableStream<Uint8Array>, this.request.signal);
  }

  async readBytes(): Promise<Uint8Array> {
    throwIfAssetAborted(this.request.signal, 'download');
    const bytes = new Uint8Array(await this.blob.arrayBuffer());
    throwIfAssetAborted(this.request.signal, 'download');
    return bytes;
  }

  async readText(): Promise<string> {
    return await this.blob.text();
  }

  async readJson<T>(): Promise<T> {
    return JSON.parse(await this.readText()) as T;
  }

  async getLocator(target: 'url' | 'path'): Promise<string | null> {
    if (target === 'path') {
      return null;
    }
    if (this.disposed) {
      throw createDisposedAssetHandleError(this.request);
    }
    if (!this.locatorUrl) {
      this.locatorUrl = URL.createObjectURL(this.blob);
    }

    return this.locatorUrl;
  }

  dispose(): void {
    this.disposed = true;
    if (this.locatorUrl) {
      URL.revokeObjectURL(this.locatorUrl);
      this.locatorUrl = null;
    }
  }
}

export class UrlAssetHandle implements ResolvedAssetHandle {
  private blobUrl: string | null = null;
  private bytesPromise: Promise<Uint8Array> | null = null;
  private blobPromise: Promise<Blob> | null = null;
  private disposed = false;

  constructor(
    readonly request: AssetRequest,
    private readonly url: string,
    private readonly cache?: AssetCache,
  ) {}

  get contentType(): string | undefined {
    return this.request.contentType;
  }

  async *openStream(): AsyncIterable<Uint8Array> {
    throwIfAssetAborted(this.request.signal, 'download');
    const cached = await readCache(
      this.cache,
      this.request.cacheKey,
      this.request.cacheKeyFallbacks,
    );
    if (cached) {
      await migrateCacheHit(this.cache, this.request.cacheKey, cached);
      throwIfAssetAborted(this.request.signal, 'download');
      emitAssetProgress(this.request, {
        id: this.request.id,
        loaded: cached.value.bytes.byteLength,
        total: cached.value.bytes.byteLength,
        done: true,
        source: 'cache',
      });
      yield* bytesToStream(cached.value.bytes);
      return;
    }

    try {
      const response = await fetchWithCandidates(this.request, this.url);

      const totalHeader = response.headers.get('content-length');
      const total = totalHeader ? Number.parseInt(totalHeader, 10) : undefined;
      const body = response.body;
      if (!body) {
        const bytes = new Uint8Array(await response.arrayBuffer());
        throwIfAssetAborted(this.request.signal, 'download');
        await writeCache(this.cache, this.request.cacheKey, {
          bytes,
          contentType: response.headers.get('content-type') || undefined,
        });
        emitAssetProgress(this.request, {
          id: this.request.id,
          loaded: bytes.byteLength,
          total,
          done: true,
          source: 'network',
        });
        yield bytes;
        return;
      }

      // `openStream()` is the large-asset path. Retaining every chunk here
      // would double the live payload for callers that did not request a
      // cache write, so only collect when there is an actual cache target.
      const cacheChunks: Uint8Array[] | null = this.cache && this.request.cacheKey ? [] : null;
      let loaded = 0;
      for await (const chunk of readReadableStream(
        body as ReadableStream<Uint8Array>,
        this.request.signal,
      )) {
        cacheChunks?.push(chunk);
        loaded += chunk.byteLength;
        emitAssetProgress(this.request, {
          id: this.request.id,
          loaded,
          total,
          source: 'network',
        });
        yield chunk;
      }

      throwIfAssetAborted(this.request.signal, 'download');
      emitAssetProgress(this.request, {
        id: this.request.id,
        loaded,
        total,
        done: true,
        source: 'network',
      });

      if (cacheChunks) {
        let bytes: Uint8Array;
        if (cacheChunks.length === 1) {
          bytes = cacheChunks[0]!;
        } else {
          bytes = new Uint8Array(loaded);
          let offset = 0;
          for (const chunk of cacheChunks) {
            bytes.set(chunk, offset);
            offset += chunk.byteLength;
          }
        }
        await writeCache(this.cache, this.request.cacheKey, {
          bytes,
          contentType: response.headers.get('content-type') || undefined,
        });
      }
    } catch (error) {
      this.failDownload(error);
    }
  }

  async readBytes(): Promise<Uint8Array> {
    if (!this.bytesPromise) {
      this.bytesPromise = (async () => {
        throwIfAssetAborted(this.request.signal, 'download');
        const cached = await readCache(
          this.cache,
          this.request.cacheKey,
          this.request.cacheKeyFallbacks,
        );
        if (cached) {
          await migrateCacheHit(this.cache, this.request.cacheKey, cached);
          throwIfAssetAborted(this.request.signal, 'download');
          emitAssetProgress(this.request, {
            id: this.request.id,
            loaded: cached.value.bytes.byteLength,
            total: cached.value.bytes.byteLength,
            done: true,
            source: 'cache',
          });
          return cached.value.bytes;
        }

        try {
          const response = await fetchWithCandidates(this.request, this.url);

          const totalHeader = response.headers.get('content-length');
          const total = totalHeader ? Number.parseInt(totalHeader, 10) : undefined;
          const body = response.body;
          const bytes = body
            ? await streamToBytes(
                readReadableStream(body as ReadableStream<Uint8Array>, this.request.signal),
                (_chunk, loaded) => {
                  emitAssetProgress(this.request, {
                    id: this.request.id,
                    loaded,
                    total,
                    source: 'network',
                  });
                },
              )
            : new Uint8Array(await response.arrayBuffer());

          throwIfAssetAborted(this.request.signal, 'download');

          emitAssetProgress(this.request, {
            id: this.request.id,
            loaded: bytes.byteLength,
            total,
            done: true,
            source: 'network',
          });

          await writeCache(this.cache, this.request.cacheKey, {
            bytes,
            contentType: response.headers.get('content-type') || undefined,
          });

          return bytes;
        } catch (error) {
          this.bytesPromise = null;
          this.failDownload(error);
        }
      })();
    }

    return this.bytesPromise;
  }

  async readText(): Promise<string> {
    return new TextDecoder().decode(await this.readBytes());
  }

  async readJson<T>(): Promise<T> {
    return JSON.parse(await this.readText()) as T;
  }

  async getLocator(target: 'url' | 'path'): Promise<string | null> {
    if (target === 'path') {
      return null;
    }

    throwIfAssetAborted(this.request.signal, 'download');

    if (this.disposed) {
      throw createDisposedAssetHandleError(this.request);
    }

    if (/^https?:\/\//i.test(this.url) && !this.request.preferBlobUrl) {
      return this.url;
    }

    if (!this.blobUrl) {
      const blob = await this.readBlob();
      if (this.disposed) {
        throw createDisposedAssetHandleError(this.request);
      }
      if (!this.blobUrl) {
        const blobUrl = URL.createObjectURL(blob);
        if (this.disposed) {
          URL.revokeObjectURL(blobUrl);
          throw createDisposedAssetHandleError(this.request);
        }
        this.blobUrl = blobUrl;
      }
    }

    return this.blobUrl;
  }

  private async readBlob(): Promise<Blob> {
    if (!this.blobPromise) {
      this.blobPromise = (async () => {
        throwIfAssetAborted(this.request.signal, 'download');
        const cached = await readBlobCache(
          this.cache,
          this.request.cacheKey,
          this.request.cacheKeyFallbacks,
        );
        if (cached) {
          await migrateBlobCacheHit(this.cache, this.request.cacheKey, cached);
          throwIfAssetAborted(this.request.signal, 'download');
          emitAssetProgress(this.request, {
            id: this.request.id,
            loaded: cached.blob.size,
            total: cached.blob.size,
            done: true,
            source: 'cache',
          });
          return cached.blob;
        }

        try {
          const response = await fetchWithCandidates(this.request, this.url);
          const contentType = response.headers.get('content-type') || this.request.contentType;
          const totalHeader = response.headers.get('content-length');
          const total = totalHeader ? Number.parseInt(totalHeader, 10) : undefined;
          const body = response.body;
          const blob = body
            ? await streamToBlob(
                readReadableStream(body as ReadableStream<Uint8Array>, this.request.signal),
                contentType || undefined,
                (_chunk, loaded) => {
                  emitAssetProgress(this.request, {
                    id: this.request.id,
                    loaded,
                    total,
                    source: 'network',
                  });
                },
              )
            : await response.blob();

          throwIfAssetAborted(this.request.signal, 'download');

          emitAssetProgress(this.request, {
            id: this.request.id,
            loaded: blob.size,
            total,
            done: true,
            source: 'network',
          });

          await writeBlobCache(this.cache, this.request.cacheKey, blob);

          return blob;
        } catch (error) {
          this.blobPromise = null;
          this.failDownload(error);
        }
      })();
    }

    return this.blobPromise;
  }

  private failDownload(error: unknown): never {
    const aborted = toAssetLoadAbortedError(error, 'download');
    if (isAssetLoadAbortedError(aborted)) {
      emitAssetProgress(this.request, {
        loaded: 0,
        done: true,
        aborted: true,
        source: 'network',
      });
      this.bytesPromise = null;
      this.blobPromise = null;
      this.dispose();
      throw aborted;
    }
    throw error;
  }

  dispose(): void {
    this.disposed = true;
    if (this.blobUrl) {
      URL.revokeObjectURL(this.blobUrl);
      this.blobUrl = null;
    }
  }
}
