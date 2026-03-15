import type {
  AssetCache,
  AssetCacheValue,
  AssetRequest,
  ResolvedAssetHandle,
} from '../types/index.js';

async function* bytesToStream(bytes: Uint8Array): AsyncIterable<Uint8Array> {
  yield bytes;
}

async function* readReadableStream(stream: ReadableStream<Uint8Array>): AsyncIterable<Uint8Array> {
  const reader = stream.getReader();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        return;
      }
      if (value) {
        yield value;
      }
    }
  } finally {
    reader.releaseLock();
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

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }

  return bytes;
}

async function readCache(
  cache: AssetCache | undefined,
  key: string | undefined,
): Promise<AssetCacheValue | null> {
  if (!cache || !key) {
    return null;
  }

  return cache.get(key);
}

async function writeCache(
  cache: AssetCache | undefined,
  key: string | undefined,
  value: AssetCacheValue,
): Promise<void> {
  if (!cache || !key) {
    return;
  }

  await cache.set(key, value);
}

export class BlobAssetHandle implements ResolvedAssetHandle {
  private locatorUrl: string | null = null;

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
    return readReadableStream(stream as ReadableStream<Uint8Array>);
  }

  async readBytes(): Promise<Uint8Array> {
    return new Uint8Array(await this.blob.arrayBuffer());
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

    if (!this.locatorUrl) {
      this.locatorUrl = URL.createObjectURL(this.blob);
    }

    return this.locatorUrl;
  }

  dispose(): void {
    if (this.locatorUrl) {
      URL.revokeObjectURL(this.locatorUrl);
      this.locatorUrl = null;
    }
  }
}

export class UrlAssetHandle implements ResolvedAssetHandle {
  private blobUrl: string | null = null;
  private bytesPromise: Promise<Uint8Array> | null = null;

  constructor(
    readonly request: AssetRequest,
    private readonly url: string,
    private readonly cache?: AssetCache,
  ) {}

  get contentType(): string | undefined {
    return this.request.contentType;
  }

  async *openStream(): AsyncIterable<Uint8Array> {
    const cached = await readCache(this.cache, this.request.cacheKey);
    if (cached) {
      this.request.onProgress?.({
        id: this.request.id,
        loaded: cached.bytes.byteLength,
        total: cached.bytes.byteLength,
        done: true,
      });
      yield* bytesToStream(cached.bytes);
      return;
    }

    const response = await fetch(this.url);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch asset "${this.request.id}": ${response.status} ${response.statusText}`,
      );
    }

    const totalHeader = response.headers.get('content-length');
    const total = totalHeader ? Number.parseInt(totalHeader, 10) : undefined;
    const body = response.body;
    if (!body) {
      const bytes = new Uint8Array(await response.arrayBuffer());
      await writeCache(this.cache, this.request.cacheKey, {
        bytes,
        contentType: response.headers.get('content-type') || undefined,
      });
      this.request.onProgress?.({
        id: this.request.id,
        loaded: bytes.byteLength,
        total,
        done: true,
      });
      yield bytes;
      return;
    }

    const chunks: Uint8Array[] = [];
    let loaded = 0;
    for await (const chunk of readReadableStream(body as ReadableStream<Uint8Array>)) {
      chunks.push(chunk);
      loaded += chunk.byteLength;
      this.request.onProgress?.({
        id: this.request.id,
        loaded,
        total,
      });
      yield chunk;
    }

    this.request.onProgress?.({
      id: this.request.id,
      loaded,
      total,
      done: true,
    });

    if (this.cache && this.request.cacheKey) {
      const bytes = new Uint8Array(loaded);
      let offset = 0;
      for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
      }
      await writeCache(this.cache, this.request.cacheKey, {
        bytes,
        contentType: response.headers.get('content-type') || undefined,
      });
    }
  }

  async readBytes(): Promise<Uint8Array> {
    if (!this.bytesPromise) {
      this.bytesPromise = (async () => {
        const cached = await readCache(this.cache, this.request.cacheKey);
        if (cached) {
          this.request.onProgress?.({
            id: this.request.id,
            loaded: cached.bytes.byteLength,
            total: cached.bytes.byteLength,
            done: true,
          });
          return cached.bytes;
        }

        const response = await fetch(this.url);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch asset "${this.request.id}": ${response.status} ${response.statusText}`,
          );
        }

        const totalHeader = response.headers.get('content-length');
        const total = totalHeader ? Number.parseInt(totalHeader, 10) : undefined;
        const body = response.body;
        const bytes = body
          ? await streamToBytes(
              readReadableStream(body as ReadableStream<Uint8Array>),
              (_chunk, loaded) => {
                this.request.onProgress?.({
                  id: this.request.id,
                  loaded,
                  total,
                });
              },
            )
          : new Uint8Array(await response.arrayBuffer());

        this.request.onProgress?.({
          id: this.request.id,
          loaded: bytes.byteLength,
          total,
          done: true,
        });

        await writeCache(this.cache, this.request.cacheKey, {
          bytes,
          contentType: response.headers.get('content-type') || undefined,
        });

        return bytes;
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

    if (!this.blobUrl) {
      const bytes = await this.readBytes();
      const blob = new Blob([bytes.slice().buffer], {
        type: this.request.contentType ?? 'application/octet-stream',
      });
      this.blobUrl = URL.createObjectURL(blob);
    }

    return this.blobUrl;
  }

  dispose(): void {
    if (this.blobUrl) {
      URL.revokeObjectURL(this.blobUrl);
      this.blobUrl = null;
    }
  }
}
