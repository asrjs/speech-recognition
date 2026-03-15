import type { AudioInputLike } from './audio.js';
import type { AssetCache, AssetProvider } from './io.js';
import type { ModelArchitectureDescriptor } from './architecture.js';
import type { ModelClassification } from './classification.js';
import type {
  BackendCapabilities,
  BackendProbeContext,
  BackendSelectionCriteria,
  ExecutionBackend,
} from './backend.js';
import type {
  TranscriptDetailLevel,
  TranscriptMetrics,
  TranscriptResponse,
  TranscriptResponseFlavor,
} from './transcript.js';
import type { StreamingSessionOptions, StreamingTranscriber } from './streaming.js';

export interface AbortSignalLike {
  readonly aborted: boolean;
}

/** Runtime-level progress update used for model loading and other long-running operations. */
export interface RuntimeProgressEvent {
  readonly phase: string;
  readonly loaded?: number;
  readonly total?: number;
  readonly file?: string;
  readonly percent?: number;
  readonly loadedMiB?: number;
  readonly totalMiB?: number;
  readonly isComplete?: boolean;
  readonly modelId?: string;
  readonly backendId?: string;
  readonly message?: string;
}

export type RuntimeProgressCallback = (event: RuntimeProgressEvent) => void;

/** Stable stage labels used for per-transcription progress callbacks. */
export type TranscriptionProgressStage =
  | 'start'
  | 'preprocess'
  | 'encode'
  | 'decode'
  | 'postprocess'
  | 'complete';

/** Per-transcription progress update emitted while a session is processing audio. */
export interface TranscriptionProgressEvent {
  readonly stage: TranscriptionProgressStage;
  readonly progress: number;
  readonly elapsedMs?: number;
  readonly remainingMs?: number;
  readonly completedUnits?: number;
  readonly totalUnits?: number;
  readonly modelId?: string;
  readonly backendId?: string;
  readonly message?: string;
  readonly metrics?: TranscriptMetrics;
}

export type TranscriptionProgressCallback = (event: TranscriptionProgressEvent) => void;

/** Optional logger hooks accepted by the runtime. */
export interface RuntimeLogger {
  debug?(message: string, meta?: Record<string, unknown>): void;
  info?(message: string, meta?: Record<string, unknown>): void;
  warn?(message: string, meta?: Record<string, unknown>): void;
  error?(message: string, meta?: Record<string, unknown>): void;
}

/** Optional runtime hooks for logging and progress reporting. */
export interface SpeechRuntimeHooks {
  readonly logger?: RuntimeLogger;
  readonly onProgress?: RuntimeProgressCallback;
}

/** Session-wide options that are stable across model families. */
export interface BaseSessionOptions {
  readonly locale?: string;
}

/** Stable cross-family transcription options understood by all sessions. */
export interface BaseTranscriptionOptions {
  readonly detail?: TranscriptDetailLevel;
  readonly responseFlavor?: TranscriptResponseFlavor;
  readonly language?: string;
  readonly timeOffsetSeconds?: number;
  readonly chunkLengthSeconds?: number;
  readonly strideLengthSeconds?: number;
  readonly signal?: AbortSignalLike | null;
  readonly onProgress?: TranscriptionProgressCallback;
}

/** Metadata describing a loaded model instance. */
export interface SpeechModelInfo {
  readonly family: string;
  readonly modelId: string;
  readonly classification: ModelClassification;
  readonly preset?: string;
  readonly architecture?: ModelArchitectureDescriptor;
  readonly description?: string;
  readonly nativeOutputName?: string;
}

/** Explicit model-family load request. */
export interface FamilyModelLoadRequest<TLoadOptions = unknown> {
  readonly family: string;
  readonly preset?: never;
  readonly modelId: string;
  readonly classification?: Partial<ModelClassification>;
  readonly backend?: string;
  readonly options?: TLoadOptions;
  readonly selectionCriteria?: BackendSelectionCriteria;
  readonly resolvedPreset?: string;
}

/** Branded preset load request that resolves into a technical family request. */
export interface PresetModelLoadRequest<TLoadOptions = unknown> {
  readonly family?: never;
  readonly preset: string;
  readonly modelId?: string;
  readonly classification?: Partial<ModelClassification>;
  readonly backend?: string;
  readonly options?: TLoadOptions;
  readonly selectionCriteria?: BackendSelectionCriteria;
}

/** Discriminated union for model loading through either a family or a preset. */
export type ModelLoadRequest<TLoadOptions = unknown> =
  | FamilyModelLoadRequest<TLoadOptions>
  | PresetModelLoadRequest<TLoadOptions>;

/** Shared IO context exposed to runtime, preset, and model factory implementations. */
export interface SpeechRuntimeIoContext {
  readonly assetProvider?: AssetProvider;
  readonly assetCache?: AssetCache;
}

/** Context object passed to preset resolution. */
export interface SpeechPresetFactoryContext extends SpeechRuntimeIoContext {
  readonly runtime: SpeechRuntime;
  readonly hooks: SpeechRuntimeHooks;
}

/** Context object passed to model-family creation. */
export interface SpeechModelFactoryContext extends SpeechRuntimeIoContext {
  readonly runtime: SpeechRuntime;
  readonly backend: ExecutionBackend;
  readonly hooks: SpeechRuntimeHooks;
}

/** Per-model inference session. Sessions own backend resources and must be disposed explicitly. */
export interface SpeechSession<
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
> {
  transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor },
  ): Promise<TranscriptResponse<TNative, TFlavor>>;
  dispose(): Promise<void> | void;
}

/** Loaded model instance produced by a model family. */
export interface SpeechModel<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
> {
  readonly info: SpeechModelInfo;
  readonly backend: ExecutionBackend;
  readonly loadOptions?: TLoadOptions;
  createSession(
    options?: BaseSessionOptions,
  ): Promise<SpeechSession<TTranscriptionOptions, TNative>>;
  createStreamingTranscriber?(options?: StreamingSessionOptions): Promise<StreamingTranscriber>;
  dispose(): Promise<void> | void;
}

/** Technical implementation family, such as `nemo-tdt` or `whisper-seq2seq`. */
export interface SpeechModelFactory<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
> {
  readonly family: string;
  readonly classification?: Partial<ModelClassification>;
  supports(modelId: string): boolean;
  matchesClassification?(classification: Partial<ModelClassification>): boolean;
  createModel(
    request: FamilyModelLoadRequest<TLoadOptions>,
    context: SpeechModelFactoryContext,
  ): Promise<SpeechModel<TLoadOptions, TTranscriptionOptions, TNative>>;
}

/** Thin branded layer that resolves a preset into a family-specific load request. */
export interface SpeechPresetFactory<TPresetOptions = unknown, TResolvedLoadOptions = unknown> {
  readonly preset: string;
  supports(modelId?: string): boolean;
  resolveModelRequest(
    request: PresetModelLoadRequest<TPresetOptions>,
    context: SpeechPresetFactoryContext,
  ): Promise<FamilyModelLoadRequest<TResolvedLoadOptions>>;
}

/**
 * Top-level runtime orchestration surface.
 *
 * The runtime owns registration, backend selection, model loading, and loaded
 * model lifecycle, but not model-family execution logic itself.
 */
export interface SpeechRuntime extends SpeechRuntimeIoContext {
  registerBackend(backend: ExecutionBackend): this;
  registerModelFamily(factory: SpeechModelFactory<any, any, any>): this;
  registerPreset(factory: SpeechPresetFactory<any, any>): this;
  listBackends(): readonly ExecutionBackend[];
  listModelFamilies(): readonly SpeechModelFactory<any, any, any>[];
  listPresets(): readonly SpeechPresetFactory<any, any>[];
  probeBackends(context?: BackendProbeContext): Promise<readonly BackendCapabilities[]>;
  selectBackend(criteria?: BackendSelectionCriteria): Promise<ExecutionBackend>;
  loadModel<
    TLoadOptions = unknown,
    TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
    TNative = unknown,
  >(
    request: ModelLoadRequest<TLoadOptions>,
  ): Promise<SpeechModel<TLoadOptions, TTranscriptionOptions, TNative>>;
  dispose(): Promise<void> | void;
}
