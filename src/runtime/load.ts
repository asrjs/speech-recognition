import {
  createBuiltInSpeechRuntime,
  loadBuiltInSpeechModel,
  type BuiltInSpeechModelHandle,
  type CreateBuiltInSpeechRuntimeOptions,
  type LoadBuiltInSpeechModelOptions,
} from './builtins.js';
import { PcmAudioBuffer } from '../audio/index.js';
import {
  createDefaultModelInferenceLimits,
  planWindowedTranscription,
  transcribeWithWindowing,
  withResolvedTranscriptDetail,
} from '../pipeline/index.js';
import type {
  AudioInputLike,
  BaseTranscriptionOptions,
  MonoPcmInput,
  SpeechBatchSession,
  SpeechSession,
  TranscriptResponse,
  TranscriptResponseFlavor,
} from '../types/index.js';
import { NotImplementedSpeechFeatureError } from './errors.js';
import type { DefaultSpeechRuntime } from './session.js';

/**
 * App-facing convenience handle returned by `loadSpeechModel()`.
 *
 * This is a thin alias over the built-in model handle so consumers can start
 * from the root package without learning the built-ins namespace first.
 */
export interface LoadedSpeechModel<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
> extends BuiltInSpeechModelHandle<TLoadOptions, TTranscriptionOptions, TNative> {
  /** True when the loaded session exposes mixed-length batch execution. */
  readonly supportsBatch: boolean;
  transcribeBatch<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    inputs: readonly AudioInputLike[],
    options?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor },
  ): Promise<readonly TranscriptResponse<TNative, TFlavor>[]>;
  transcribeMonoPcm<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    pcm: MonoPcmInput,
    sampleRate: number,
    options?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor },
  ): Promise<TranscriptResponse<TNative, TFlavor>>;
}

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
  const handle = await loadBuiltInSpeechModel<TLoadOptions, TTranscriptionOptions, TNative>(
    options,
  );
  return createLoadedSpeechModelHandle(handle);
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

/**
 * One-shot convenience helper for callers starting from raw mono PCM.
 *
 * This keeps the root high-level API explicit about sample rate without forcing
 * app code to manually wrap audio in `PcmAudioBuffer`.
 */
export async function transcribeSpeechFromMonoPcm<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
  TFlavor extends TranscriptResponseFlavor = 'canonical',
>(
  pcm: MonoPcmInput,
  sampleRate: number,
  options: TranscribeSpeechOptions<TLoadOptions, TTranscriptionOptions, TFlavor>,
): Promise<TranscriptResponse<TNative, TFlavor>> {
  return transcribeSpeech<TLoadOptions, TTranscriptionOptions, TNative, TFlavor>(
    createMonoPcmAudioBuffer(pcm, sampleRate),
    options,
  );
}

/**
 * One-shot batch helper for model families that expose a batch-capable
 * session. The helper preserves the same response-flavor contract as
 * `transcribeSpeech()` and disposes the loaded model after the batch.
 */
export async function transcribeSpeechBatch<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
  TFlavor extends TranscriptResponseFlavor = 'canonical',
>(
  inputs: readonly AudioInputLike[],
  options: TranscribeSpeechOptions<TLoadOptions, TTranscriptionOptions, TFlavor>,
): Promise<readonly TranscriptResponse<TNative, TFlavor>[]> {
  const { transcribeOptions, ...loadOptions } = options;
  const loaded = await loadSpeechModel<TLoadOptions, TTranscriptionOptions, TNative>(loadOptions);

  try {
    return await loaded.transcribeBatch(inputs, transcribeOptions);
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

export interface SpeechPipelineModelRequest<TLoadOptions = unknown> extends Omit<
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
  transcribeMonoPcm<
    TLoadOptions = unknown,
    TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
    TNative = unknown,
    TFlavor extends TranscriptResponseFlavor = 'canonical',
  >(
    pcm: MonoPcmInput,
    sampleRate: number,
    request: SpeechPipelineTranscribeRequest<TLoadOptions, TTranscriptionOptions, TFlavor>,
  ): Promise<TranscriptResponse<TNative, TFlavor>>;
  transcribeBatch<
    TLoadOptions = unknown,
    TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
    TNative = unknown,
    TFlavor extends TranscriptResponseFlavor = 'canonical',
  >(
    inputs: readonly AudioInputLike[],
    request: SpeechPipelineTranscribeRequest<TLoadOptions, TTranscriptionOptions, TFlavor>,
  ): Promise<readonly TranscriptResponse<TNative, TFlavor>[]>;
  listLoadedModels(): readonly string[];
  disposeModel(requestOrCacheKey: string | SpeechPipelineModelRequest<unknown>): Promise<void>;
  /** Dispose all loaded models to free GPU memory without deleting IndexedDB cache. */
  flushAllModels(): Promise<void>;
  dispose(): Promise<void>;
}

type UnknownLoadedModelHandle = LoadedSpeechModel<unknown, BaseTranscriptionOptions, unknown>;

function createMonoPcmAudioBuffer(pcm: MonoPcmInput, sampleRate: number): PcmAudioBuffer {
  return PcmAudioBuffer.fromMono(pcm, sampleRate);
}

function assertBatchSession<TTranscriptionOptions extends BaseTranscriptionOptions, TNative>(
  session: SpeechSession<TTranscriptionOptions, TNative>,
  modelId: string,
): SpeechBatchSession<TTranscriptionOptions, TNative> {
  if (typeof session.transcribeBatch !== 'function') {
    throw new NotImplementedSpeechFeatureError(
      `Model "${modelId}" does not expose a batch transcription capability.`,
      { feature: 'transcribeBatch', modelId },
    );
  }
  return session as SpeechBatchSession<TTranscriptionOptions, TNative>;
}

function assertBatchInputsAreDirect<TOptions extends BaseTranscriptionOptions>(
  inputs: readonly AudioInputLike[],
  options: TOptions | undefined,
  inference: ReturnType<typeof createDefaultModelInferenceLimits>,
  modelId: string,
): void {
  if (options?.windowing === 'disabled') {
    return;
  }
  for (const [index, input] of inputs.entries()) {
    const decision = planWindowedTranscription(input, options, inference);
    if (decision.shouldWindow) {
      throw new NotImplementedSpeechFeatureError(
        `Batch transcription does not support automatic long-audio windowing for model "${modelId}"; transcribe long inputs individually or set windowing to "disabled".`,
        {
          feature: 'transcribeBatch',
          modelId,
          inputIndex: index,
          durationSeconds: decision.audio.durationSeconds,
        },
      );
    }
  }
}

export function createLoadedSpeechModelHandle<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
>(
  handle: BuiltInSpeechModelHandle<TLoadOptions, TTranscriptionOptions, TNative>,
): LoadedSpeechModel<TLoadOptions, TTranscriptionOptions, TNative> {
  async function transcribeAudio<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor },
  ): Promise<TranscriptResponse<TNative, TFlavor>> {
    const resolvedOptions = withResolvedTranscriptDetail(options);
    if (resolvedOptions?.responseFlavor === 'native') {
      return handle.transcribe(input, resolvedOptions);
    }

    const inference =
      handle.model.info.inference ??
      createDefaultModelInferenceLimits({
        family: handle.model.info.family,
        modelId: handle.model.info.modelId,
      });
    const decision = planWindowedTranscription(input, resolvedOptions, inference);
    if (!decision.shouldWindow) {
      return handle.transcribe(input, resolvedOptions);
    }

    const canonical = await transcribeWithWindowing({
      input: decision.audio,
      options: { ...(resolvedOptions ?? {}), responseFlavor: 'canonical' } as TTranscriptionOptions,
      inference,
      transcribeWindow: async (windowInput, windowOptions) =>
        (await handle.transcribe(windowInput, {
          ...windowOptions,
          responseFlavor: 'canonical',
        } as TTranscriptionOptions & {
          readonly responseFlavor: 'canonical';
        })) as TranscriptResponse<TNative, 'canonical'>,
    });

    if (resolvedOptions?.responseFlavor === 'canonical+native') {
      return { canonical } as TranscriptResponse<TNative, TFlavor>;
    }
    return canonical as TranscriptResponse<TNative, TFlavor>;
  }

  const supportsBatch = typeof handle.session.transcribeBatch === 'function';

  async function transcribeBatchAudio<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    inputs: readonly AudioInputLike[],
    options?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor },
  ): Promise<readonly TranscriptResponse<TNative, TFlavor>[]> {
    const batchSession = assertBatchSession(handle.session, handle.model.info.modelId);
    const resolvedOptions = withResolvedTranscriptDetail(options);
    const inference =
      handle.model.info.inference ??
      createDefaultModelInferenceLimits({
        family: handle.model.info.family,
        modelId: handle.model.info.modelId,
      });
    assertBatchInputsAreDirect(inputs, resolvedOptions, inference, handle.model.info.modelId);
    return batchSession.transcribeBatch(inputs, resolvedOptions);
  }

  return {
    runtime: handle.runtime,
    model: handle.model,
    session: handle.session,
    supportsBatch,
    transcribe: transcribeAudio,
    transcribeBatch: transcribeBatchAudio,
    async transcribeMonoPcm<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
      pcm: MonoPcmInput,
      sampleRate: number,
      options?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor },
    ): Promise<TranscriptResponse<TNative, TFlavor>> {
      return transcribeAudio(createMonoPcmAudioBuffer(pcm, sampleRate), options);
    },
    async dispose(): Promise<void> {
      await handle.dispose();
    },
  };
}

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
  private readonly generations = new Map<string, number>();
  private readonly pendingDisposals = new Map<string, Promise<void>>();
  private readonly handleDisposals = new WeakMap<UnknownLoadedModelHandle, Promise<void>>();
  private disposed = false;
  private disposePromise: Promise<void> | null = null;
  private flushPromise: Promise<void> | null = null;

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

    if (this.flushPromise) {
      await this.flushPromise;
      this.assertNotDisposed();
    }

    const cacheKey = this.cacheModels
      ? resolveAutomaticCacheKey(request as SpeechPipelineModelRequest<unknown>)
      : null;

    if (!cacheKey) {
      const handle = await this.createModelHandle(request);
      if (this.disposed) {
        await this.disposeHandle(handle);
        throw this.createInvalidationError();
      }
      return handle as LoadedSpeechModel<
        TLoadOptions,
        TTranscriptionOptions,
        TNative
      >;
    }

    if (request.forceReload) {
      await this.disposeModel(cacheKey);
      this.assertNotDisposed();
    }

    const generation = this.getGeneration(cacheKey);
    const existing = this.handles.get(cacheKey);
    if (existing) {
      return existing as LoadedSpeechModel<TLoadOptions, TTranscriptionOptions, TNative>;
    }

    const inflight = this.inflight.get(cacheKey);
    if (inflight) {
      const handle = await inflight;
      if (!this.isGenerationCurrent(cacheKey, generation)) {
        await this.disposeHandle(handle);
        throw this.createInvalidationError();
      }
      return handle as LoadedSpeechModel<TLoadOptions, TTranscriptionOptions, TNative>;
    }

    const created = this.createModelHandle(request);
    this.inflight.set(cacheKey, created);

    try {
      const handle = await created;
      if (!this.isGenerationCurrent(cacheKey, generation)) {
        await this.disposeHandle(handle);
        throw this.createInvalidationError();
      }
      this.handles.set(cacheKey, handle);
      return handle as LoadedSpeechModel<TLoadOptions, TTranscriptionOptions, TNative>;
    } finally {
      if (this.inflight.get(cacheKey) === created) {
        this.inflight.delete(cacheKey);
      }
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
        return (await handle.transcribe(input, transcribeOptions)) as TranscriptResponse<
          TNative,
          TFlavor
        >;
      } finally {
        await this.disposeHandle(handle);
      }
    }

    const handle = await this.loadModel<TLoadOptions, TTranscriptionOptions, TNative>(modelRequest);
    return await handle.transcribe<TFlavor>(input, transcribeOptions);
  }

  async transcribeMonoPcm<
    TLoadOptions = unknown,
    TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
    TNative = unknown,
    TFlavor extends TranscriptResponseFlavor = 'canonical',
  >(
    pcm: MonoPcmInput,
    sampleRate: number,
    request: SpeechPipelineTranscribeRequest<TLoadOptions, TTranscriptionOptions, TFlavor>,
  ): Promise<TranscriptResponse<TNative, TFlavor>> {
    return this.transcribe<TLoadOptions, TTranscriptionOptions, TNative, TFlavor>(
      createMonoPcmAudioBuffer(pcm, sampleRate),
      request,
    );
  }

  async transcribeBatch<
    TLoadOptions = unknown,
    TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
    TNative = unknown,
    TFlavor extends TranscriptResponseFlavor = 'canonical',
  >(
    inputs: readonly AudioInputLike[],
    request: SpeechPipelineTranscribeRequest<TLoadOptions, TTranscriptionOptions, TFlavor>,
  ): Promise<readonly TranscriptResponse<TNative, TFlavor>[]> {
    this.assertNotDisposed();

    const { transcribeOptions, ...modelRequest } = request;
    const cacheKey = this.cacheModels
      ? resolveAutomaticCacheKey(modelRequest as SpeechPipelineModelRequest<unknown>)
      : null;

    if (!cacheKey) {
      const handle = await this.createModelHandle(modelRequest);
      try {
        return (await handle.transcribeBatch(
          inputs,
          transcribeOptions,
        )) as readonly TranscriptResponse<TNative, TFlavor>[];
      } finally {
        await this.disposeHandle(handle);
      }
    }

    const handle = await this.loadModel<TLoadOptions, TTranscriptionOptions, TNative>(modelRequest);
    return handle.transcribeBatch<TFlavor>(inputs, transcribeOptions);
  }

  listLoadedModels(): readonly string[] {
    return [...this.handles.keys()];
  }

  async disposeModel(
    requestOrCacheKey: string | SpeechPipelineModelRequest<unknown>,
  ): Promise<void> {
    const cacheKey =
      typeof requestOrCacheKey === 'string'
        ? requestOrCacheKey
        : resolveAutomaticCacheKey(requestOrCacheKey);

    if (!cacheKey) {
      return;
    }

    if (this.disposed) {
      await this.disposePromise;
      return;
    }

    this.invalidateGeneration(cacheKey);

    const existing = this.handles.get(cacheKey);
    if (existing) {
      this.handles.delete(cacheKey);
    }

    const inflight = this.inflight.get(cacheKey);
    if (inflight) {
      this.inflight.delete(cacheKey);
    }

    const previousDisposal = this.pendingDisposals.get(cacheKey);
    const disposal = (async () => {
      const errors: unknown[] = [];

      if (previousDisposal) {
        try {
          await previousDisposal;
        } catch (error) {
          errors.push(error);
        }
      }

      if (existing) {
        try {
          await this.disposeHandle(existing);
        } catch (error) {
          errors.push(error);
        }
      }

      if (inflight) {
        const result = await Promise.resolve(inflight).catch(() => null);
        if (result) {
          try {
            await this.disposeHandle(result);
          } catch (error) {
            errors.push(error);
          }
        }
      }

      if (errors.length > 0) {
        throw errors[0];
      }
    })();
    this.pendingDisposals.set(cacheKey, disposal);

    try {
      await disposal;
    } finally {
      if (this.pendingDisposals.get(cacheKey) === disposal) {
        this.pendingDisposals.delete(cacheKey);
      }
    }
  }

  /** Dispose all loaded models to free GPU memory without deleting IndexedDB cache.
   *  The runtime stays alive — models can be reloaded from IndexedDB immediately.
   *  Use this between audio files to prevent VRAM accumulation from cache-key changes. */
  async flushAllModels(): Promise<void> {
    if (this.disposed) {
      await this.disposePromise;
      return;
    }
    if (this.flushPromise) {
      await this.flushPromise;
      return;
    }

    const flush = this.flushAllModelsInternal();
    this.flushPromise = flush;
    try {
      await flush;
    } finally {
      if (this.flushPromise === flush) {
        this.flushPromise = null;
      }
    }
  }

  async dispose(): Promise<void> {
    if (this.disposePromise) {
      await this.disposePromise;
      return;
    }
    if (this.disposed) {
      return;
    }

    this.disposed = true;
    const disposal = this.disposeInternal();
    this.disposePromise = disposal;
    await disposal;
  }

  private async flushAllModelsInternal(): Promise<void> {
    const keys = new Set<string>([
      ...this.handles.keys(),
      ...this.inflight.keys(),
    ]);
    for (const cacheKey of keys) {
      this.invalidateGeneration(cacheKey);
    }

    const cachedHandles = [...this.handles.values()];
    const inflightLoads = [...this.inflight.values()];
    const pendingDisposals = [...this.pendingDisposals.values()];
    this.handles.clear();
    this.inflight.clear();

    const inflightResults = await Promise.allSettled(inflightLoads);
    const uniqueHandles = new Set<UnknownLoadedModelHandle>(cachedHandles);

    for (const result of inflightResults) {
      if (result.status === 'fulfilled') {
        uniqueHandles.add(result.value);
      }
    }

    await Promise.allSettled(pendingDisposals);
    await Promise.all(
      [...uniqueHandles].map(async (handle) => {
        try {
          await this.disposeHandle(handle);
        } catch {
          /* best-effort */
        }
      }),
    );
  }

  private async disposeInternal(): Promise<void> {
    if (this.flushPromise) {
      await this.flushPromise;
    }

    const keys = new Set<string>([
      ...this.handles.keys(),
      ...this.inflight.keys(),
    ]);
    for (const cacheKey of keys) {
      this.invalidateGeneration(cacheKey);
    }

    const cachedHandles = [...this.handles.values()];
    const inflightLoads = [...this.inflight.values()];
    const pendingDisposals = [...this.pendingDisposals.values()];
    this.handles.clear();
    this.inflight.clear();

    const inflightResults = await Promise.allSettled(inflightLoads);
    await Promise.allSettled(pendingDisposals);

    if (!this.ownsRuntime) {
      const uniqueHandles = new Set<UnknownLoadedModelHandle>(cachedHandles);
      for (const result of inflightResults) {
        if (result.status === 'fulfilled') {
          uniqueHandles.add(result.value);
        }
      }
      await Promise.all(
        [...uniqueHandles].map(async (handle) => {
          await this.disposeHandle(handle);
        }),
      );
      return;
    }

    await this.runtime.dispose();
  }

  private getGeneration(cacheKey: string): number {
    return this.generations.get(cacheKey) ?? 0;
  }

  private invalidateGeneration(cacheKey: string): void {
    this.generations.set(cacheKey, this.getGeneration(cacheKey) + 1);
  }

  private isGenerationCurrent(cacheKey: string, generation: number): boolean {
    return !this.disposed && this.getGeneration(cacheKey) === generation;
  }

  private createInvalidationError(): Error {
    return this.disposed
      ? new Error('Speech pipeline is disposed.')
      : new Error('Speech model load was disposed or invalidated before it completed.');
  }

  private disposeHandle(handle: UnknownLoadedModelHandle): Promise<void> {
    const existing = this.handleDisposals.get(handle);
    if (existing) {
      return existing;
    }

    const disposal = Promise.resolve().then(() => handle.dispose());
    this.handleDisposals.set(handle, disposal);
    return disposal;
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
    return createLoadedSpeechModelHandle(handle) as unknown as UnknownLoadedModelHandle;
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
