import type { AudioInputLike } from './audio.js';
import type { ModelArchitectureDescriptor } from './architecture.js';
import type { ModelClassification } from './classification.js';
import type {
  BackendCapabilities,
  BackendProbeContext,
  BackendSelectionCriteria,
  ExecutionBackend
} from './backend.js';
import type {
  TranscriptDetailLevel,
  TranscriptResponse,
  TranscriptResponseFlavor
} from './transcript.js';
import type { StreamingSessionOptions, StreamingTranscriber } from './streaming.js';

export interface AbortSignalLike {
  readonly aborted: boolean;
}

export interface RuntimeProgressEvent {
  readonly phase: string;
  readonly loaded?: number;
  readonly total?: number;
  readonly modelId?: string;
  readonly backendId?: string;
  readonly message?: string;
}

export type RuntimeProgressCallback = (event: RuntimeProgressEvent) => void;

export interface RuntimeLogger {
  debug?(message: string, meta?: Record<string, unknown>): void;
  info?(message: string, meta?: Record<string, unknown>): void;
  warn?(message: string, meta?: Record<string, unknown>): void;
  error?(message: string, meta?: Record<string, unknown>): void;
}

export interface SpeechRuntimeHooks {
  readonly logger?: RuntimeLogger;
  readonly onProgress?: RuntimeProgressCallback;
}

export interface BaseSessionOptions {
  readonly locale?: string;
}

export interface BaseTranscriptionOptions {
  readonly detail?: TranscriptDetailLevel;
  readonly responseFlavor?: TranscriptResponseFlavor;
  readonly language?: string;
  readonly timeOffsetSeconds?: number;
  readonly chunkLengthSeconds?: number;
  readonly strideLengthSeconds?: number;
  readonly signal?: AbortSignalLike | null;
}

export interface SpeechModelInfo {
  readonly family: string;
  readonly modelId: string;
  readonly classification: ModelClassification;
  readonly architecture?: ModelArchitectureDescriptor;
  readonly description?: string;
  readonly nativeOutputName?: string;
}

export interface ModelLoadRequest<TLoadOptions = unknown> {
  readonly family?: string;
  readonly modelId: string;
  readonly classification?: Partial<ModelClassification>;
  readonly backend?: string;
  readonly options?: TLoadOptions;
  readonly selectionCriteria?: BackendSelectionCriteria;
}

export interface SpeechModelFactoryContext {
  readonly runtime: SpeechRuntime;
  readonly backend: ExecutionBackend;
  readonly hooks: SpeechRuntimeHooks;
}

export interface SpeechSession<
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown
> {
  transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor }
  ): Promise<TranscriptResponse<TNative, TFlavor>>;
  dispose(): Promise<void> | void;
}

export interface SpeechModel<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown
> {
  readonly info: SpeechModelInfo;
  readonly backend: ExecutionBackend;
  readonly loadOptions?: TLoadOptions;
  createSession(options?: BaseSessionOptions): Promise<SpeechSession<TTranscriptionOptions, TNative>>;
  createStreamingTranscriber?(options?: StreamingSessionOptions): Promise<StreamingTranscriber>;
  dispose(): Promise<void> | void;
}

export interface SpeechModelFactory<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown
> {
  readonly family: string;
  readonly classification?: Partial<ModelClassification>;
  supports(modelId: string): boolean;
  matchesClassification?(classification: Partial<ModelClassification>): boolean;
  createModel(
    request: ModelLoadRequest<TLoadOptions>,
    context: SpeechModelFactoryContext
  ): Promise<SpeechModel<TLoadOptions, TTranscriptionOptions, TNative>>;
}

export interface SpeechRuntime {
  registerBackend(backend: ExecutionBackend): this;
  registerModelFamily(factory: SpeechModelFactory<any, any, any>): this;
  listBackends(): readonly ExecutionBackend[];
  listModelFamilies(): readonly SpeechModelFactory<any, any, any>[];
  probeBackends(context?: BackendProbeContext): Promise<readonly BackendCapabilities[]>;
  selectBackend(criteria?: BackendSelectionCriteria): Promise<ExecutionBackend>;
  loadModel<
    TLoadOptions = unknown,
    TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
    TNative = unknown
  >(request: ModelLoadRequest<TLoadOptions>): Promise<SpeechModel<TLoadOptions, TTranscriptionOptions, TNative>>;
}
