import {
  createBuiltInSpeechRuntime,
  loadBuiltInSpeechModel,
  type BuiltInSpeechModelHandle,
  type CreateBuiltInSpeechRuntimeOptions,
  type LoadBuiltInSpeechModelOptions,
} from './builtins.js';
import type {
  AudioInputLike,
  BaseTranscriptionOptions,
  TranscriptResponse,
  TranscriptResponseFlavor,
} from '../types/index.js';
import type { DefaultSpeechRuntime } from './session.js';

/**
 * App-facing convenience handle returned by `loadSpeechModel()`.
 *
 * This is a thin alias over the built-in model handle so consumers can start
 * from the root package without learning the built-ins namespace first.
 */
export type LoadedSpeechModel<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
> = BuiltInSpeechModelHandle<TLoadOptions, TTranscriptionOptions, TNative>;

/** Root-level convenience options for loading a built-in speech model. */
export type LoadSpeechModelOptions<TLoadOptions = unknown> =
  LoadBuiltInSpeechModelOptions<TLoadOptions>;

/**
 * Loads a built-in speech model, creates a ready session, and returns a small
 * handle with `transcribe()` and `dispose()`.
 *
 * Advanced callers can still pass an explicit runtime, or bypass this helper
 * entirely and use `createSpeechRuntime().loadModel(...)` directly.
 */
export async function loadSpeechModel<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
>(
  options: LoadSpeechModelOptions<TLoadOptions>,
): Promise<LoadedSpeechModel<TLoadOptions, TTranscriptionOptions, TNative>> {
  return loadBuiltInSpeechModel<TLoadOptions, TTranscriptionOptions, TNative>(options);
}

export interface TranscribeSpeechOptions<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TFlavor extends TranscriptResponseFlavor = 'canonical',
> extends LoadSpeechModelOptions<TLoadOptions> {
  readonly transcribeOptions?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor };
}

/**
 * One-shot high-level helper for app code.
 *
 * It automatically loads a built-in model, runs a single transcription, and
 * disposes model resources when done.
 */
export async function transcribeSpeech<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
  TFlavor extends TranscriptResponseFlavor = 'canonical',
>(
  input: AudioInputLike,
  options: TranscribeSpeechOptions<TLoadOptions, TTranscriptionOptions, TFlavor>,
): Promise<TranscriptResponse<TNative, TFlavor>> {
  const { transcribeOptions, ...loadOptions } = options;
  const loaded = await loadSpeechModel<TLoadOptions, TTranscriptionOptions, TNative>(loadOptions);

  try {
    return await loaded.transcribe(input, transcribeOptions);
  } finally {
    await loaded.dispose();
  }
}

export interface SpeechPipelineOptions extends CreateBuiltInSpeechRuntimeOptions {
  readonly runtime?: DefaultSpeechRuntime;
  /**
   * When true (default), the pipeline caches loaded models by a stable request key.
   * If a request cannot be stably serialized and `cacheKey` is not provided,
   * the pipeline still works but treats that request as non-cacheable.
   */
  readonly cacheModels?: boolean;
}

export interface SpeechPipelineModelRequest<TLoadOptions = unknown>
  extends Omit<
    LoadSpeechModelOptions<TLoadOptions>,
    'runtime' | 'hooks' | 'useManifestSources'
  > {
  readonly cacheKey?: string;
  readonly forceReload?: boolean;
}

export interface SpeechPipelineTranscribeRequest<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TFlavor extends TranscriptResponseFlavor = 'canonical',
> extends SpeechPipelineModelRequest<TLoadOptions> {
  readonly transcribeOptions?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor };
}

export interface SpeechPipeline {
  readonly runtime: DefaultSpeechRuntime;
  readonly cacheModels: boolean;
  loadModel<
    TLoadOptions = unknown,
    TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
    TNative = unknown,
  >(
    request: SpeechPipelineModelRequest<TLoadOptions>,
  ): Promise<LoadedSpeechModel<TLoadOptions, TTranscriptionOptions, TNative>>;
  transcribe<
    TLoadOptions = unknown,
    TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
    TNative = unknown,
    TFlavor extends TranscriptResponseFlavor = 'canonical',
  >(
    input: AudioInputLike,
    request: SpeechPipelineTranscribeRequest<TLoadOptions, TTranscriptionOptions, TFlavor>,
  ): Promise<TranscriptResponse<TNative, TFlavor>>;
  listLoadedModels(): readonly string[];
  disposeModel(requestOrCacheKey: string | SpeechPipelineModelRequest<unknown>): Promise<void>;
  dispose(): Promise<void>;
}

type UnknownLoadedModelHandle = BuiltInSpeechModelHandle<unknown, BaseTranscriptionOptions, unknown>;

function canonicalizeCacheValue(value: unknown, seen = new WeakSet<object>()): unknown {
  if (value === null || value === undefined) {
    return value;
  }

  const valueType = typeof value;
  if (valueType === 'string' || valueType === 'number' || valueType === 'boolean') {
    return value;
  }

  if (valueType === 'bigint') {
    return value.toString();
  }

  if (valueType === 'function' || valueType === 'symbol') {
    throw new Error('Non-serializable value.');
  }

  if (Array.isArray(value)) {
    return value.map((item) => canonicalizeCacheValue(item, seen));
  }

  if (!(value instanceof Date) && valueType === 'object') {
    const record = value as Record<string, unknown>;
    const prototype = Object.getPrototypeOf(record);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new Error('Unsupported object prototype.');
    }

    if (seen.has(record)) {
      throw new Error('Circular reference.');
    }
    seen.add(record);

    const normalized: Record<string, unknown> = {};
    for (const key of Object.keys(record).sort()) {
      const normalizedValue = canonicalizeCacheValue(record[key], seen);
      if (normalizedValue !== undefined) {
        normalized[key] = normalizedValue;
      }
    }
    return normalized;
  }

  if (value instanceof Date) {
    return value.toISOString();
  }

  throw new Error('Unsupported cache key value.');
}

function resolveAutomaticCacheKey(request: SpeechPipelineModelRequest<unknown>): string | null {
  const { cacheKey, forceReload, onProgress, ...cacheInput } = request;
  void forceReload;
  void onProgress;

  if (cacheKey) {
    return cacheKey;
  }

  try {
    const normalized = canonicalizeCacheValue(cacheInput) as Record<string, unknown>;
    return `model:${JSON.stringify(normalized)}`;
  } catch {
    return null;
  }
}

class DefaultSpeechPipeline implements SpeechPipeline {
  readonly runtime: DefaultSpeechRuntime;
  readonly cacheModels: boolean;

  private readonly ownsRuntime: boolean;
  private readonly hooks: CreateBuiltInSpeechRuntimeOptions['hooks'];
  private readonly useManifestSources: boolean;
  private readonly handles = new Map<string, UnknownLoadedModelHandle>();
  private readonly inflight = new Map<string, Promise<UnknownLoadedModelHandle>>();
  private disposed = false;

  constructor(options: SpeechPipelineOptions = {}) {
    this.ownsRuntime = !options.runtime;
    this.runtime = options.runtime ?? createBuiltInSpeechRuntime(options);
    this.cacheModels = options.cacheModels ?? true;
    this.hooks = options.hooks;
    this.useManifestSources = options.useManifestSources ?? true;
  }

  async loadModel<
    TLoadOptions = unknown,
    TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
    TNative = unknown,
  >(
    request: SpeechPipelineModelRequest<TLoadOptions>,
  ): Promise<LoadedSpeechModel<TLoadOptions, TTranscriptionOptions, TNative>> {
    this.assertNotDisposed();

    const cacheKey = this.cacheModels
      ? resolveAutomaticCacheKey(request as SpeechPipelineModelRequest<unknown>)
      : null;

    if (!cacheKey) {
      return (await this.createModelHandle(request)) as LoadedSpeechModel<
        TLoadOptions,
        TTranscriptionOptions,
        TNative
      >;
    }

    if (request.forceReload) {
      await this.disposeModel(cacheKey);
    }

    const existing = this.handles.get(cacheKey);
    if (existing) {
      return existing as LoadedSpeechModel<TLoadOptions, TTranscriptionOptions, TNative>;
    }

    const inflight = this.inflight.get(cacheKey);
    if (inflight) {
      return (await inflight) as LoadedSpeechModel<TLoadOptions, TTranscriptionOptions, TNative>;
    }

    const created = this.createModelHandle(request);
    this.inflight.set(cacheKey, created);

    try {
      const handle = await created;
      this.handles.set(cacheKey, handle);
      return handle as LoadedSpeechModel<TLoadOptions, TTranscriptionOptions, TNative>;
    } finally {
      this.inflight.delete(cacheKey);
    }
  }

  async transcribe<
    TLoadOptions = unknown,
    TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
    TNative = unknown,
    TFlavor extends TranscriptResponseFlavor = 'canonical',
  >(
    input: AudioInputLike,
    request: SpeechPipelineTranscribeRequest<TLoadOptions, TTranscriptionOptions, TFlavor>,
  ): Promise<TranscriptResponse<TNative, TFlavor>> {
    this.assertNotDisposed();

    const { transcribeOptions, ...modelRequest } = request;
    const cacheKey = this.cacheModels
      ? resolveAutomaticCacheKey(modelRequest as SpeechPipelineModelRequest<unknown>)
      : null;

    if (!cacheKey) {
      const handle = await this.createModelHandle(modelRequest);
      try {
        return (await handle.transcribe(
          input,
          transcribeOptions,
        )) as TranscriptResponse<TNative, TFlavor>;
      } finally {
        await handle.dispose();
      }
    }

    const handle = await this.loadModel<TLoadOptions, TTranscriptionOptions, TNative>(modelRequest);
    return await handle.transcribe<TFlavor>(input, transcribeOptions);
  }

  listLoadedModels(): readonly string[] {
    return [...this.handles.keys()];
  }

  async disposeModel(requestOrCacheKey: string | SpeechPipelineModelRequest<unknown>): Promise<void> {
    const cacheKey =
      typeof requestOrCacheKey === 'string'
        ? requestOrCacheKey
        : resolveAutomaticCacheKey(requestOrCacheKey);

    if (!cacheKey) {
      return;
    }

    const existing = this.handles.get(cacheKey);
    if (existing) {
      await existing.dispose();
      this.handles.delete(cacheKey);
    }

    const inflight = this.inflight.get(cacheKey);
    if (inflight) {
      this.inflight.delete(cacheKey);
      const result = await Promise.resolve(inflight).catch(() => null);
      if (result) {
        await result.dispose();
      }
    }
  }

  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }
    this.disposed = true;

    const inflightResults = await Promise.allSettled(this.inflight.values());
    this.inflight.clear();

    if (!this.ownsRuntime) {
      const uniqueHandles = new Set<UnknownLoadedModelHandle>(this.handles.values());
      for (const result of inflightResults) {
        if (result.status === 'fulfilled') {
          uniqueHandles.add(result.value);
        }
      }
      this.handles.clear();
      await Promise.all([...uniqueHandles].map((handle) => Promise.resolve(handle.dispose())));
      return;
    }

    this.handles.clear();
    await this.runtime.dispose();
  }

  private async createModelHandle<TLoadOptions>(
    request: SpeechPipelineModelRequest<TLoadOptions>,
  ): Promise<UnknownLoadedModelHandle> {
    const { cacheKey, forceReload, ...loadOptions } = request;
    void cacheKey;
    void forceReload;

    const handle = await loadBuiltInSpeechModel({
      ...loadOptions,
      runtime: this.runtime,
      hooks: this.hooks,
      useManifestSources: this.useManifestSources,
    });
    return handle as UnknownLoadedModelHandle;
  }

  private assertNotDisposed(): void {
    if (this.disposed) {
      throw new Error('Speech pipeline is disposed.');
    }
  }
}

/**
 * Creates a model-agnostic high-level speech pipeline with optional model caching.
 *
 * This is designed for app code that wants one surface for loading and
 * transcribing across multiple model families/presets while preserving direct,
 * low-level runtime/model APIs for advanced workflows.
 */
export function createSpeechPipeline(options: SpeechPipelineOptions = {}): SpeechPipeline {
  return new DefaultSpeechPipeline(options);
}
