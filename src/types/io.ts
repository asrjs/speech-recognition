/** Progress update emitted while an asset provider streams or downloads a model artifact. */
export interface AssetProgressEvent {
  readonly loaded: number;
  readonly total?: number;
  readonly id?: string;
  readonly done?: boolean;
}

/** Minimal browser file-handle shape used by local-entry loaders without depending on the full File System Access API. */
export interface LocalAssetFileHandleLike {
  getFile(): Promise<File | Blob>;
}

/**
 * Logical description of an asset to load.
 *
 * The runtime resolves this through an {@link AssetProvider} rather than
 * hardcoding fetch, filesystem, or cache behavior inside model families.
 */
export interface AssetRequest {
  readonly id: string;
  readonly kind?: 'binary' | 'text' | 'json';
  readonly provider?: 'huggingface' | 'url' | 'path' | 'blob' | 'bytes' | 'file-handle';
  readonly url?: string;
  readonly path?: string;
  readonly repoId?: string;
  readonly revision?: string;
  readonly filename?: string;
  readonly subfolder?: string;
  readonly blob?: Blob;
  readonly bytes?: Uint8Array;
  readonly fileHandle?: LocalAssetFileHandleLike;
  readonly cacheKey?: string;
  readonly contentType?: string;
  readonly onProgress?: (event: AssetProgressEvent) => void;
}

/** Cache payload stored for a resolved asset. */
export interface AssetCacheValue {
  readonly bytes: Uint8Array;
  readonly contentType?: string;
}

/** Persistent or in-memory cache used by asset providers. */
export interface AssetCache {
  get(key: string): Promise<AssetCacheValue | null>;
  set(key: string, value: AssetCacheValue): Promise<void>;
  delete?(key: string): Promise<void>;
}

/**
 * Resolved asset handle returned by an {@link AssetProvider}.
 *
 * Handles are stream-first so large model files can be consumed incrementally.
 * Implementations may lazily materialize temporary URLs or paths and must clean
 * them up on {@link dispose}.
 */
export interface ResolvedAssetHandle {
  readonly request: AssetRequest;
  readonly contentType?: string;
  readonly sizeBytes?: number;
  openStream(): AsyncIterable<Uint8Array>;
  readBytes(): Promise<Uint8Array>;
  readText(): Promise<string>;
  readJson<T>(): Promise<T>;
  getLocator(target: 'url' | 'path'): Promise<string | null>;
  dispose(): Promise<void> | void;
}

/** Environment-aware resolver for model assets such as ONNX files, tokenizers, and configs. */
export interface AssetProvider {
  canResolve(request: AssetRequest): boolean;
  resolve(request: AssetRequest): Promise<ResolvedAssetHandle>;
}
