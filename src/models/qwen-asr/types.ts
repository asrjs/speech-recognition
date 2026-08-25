import type {
  AssetProvider,
  AudioBufferLike,
  BaseTranscriptionOptions,
  ModelClassification,
  SpeechRuntimeHooks,
  TranscriptMetrics,
  TranscriptWarning,
} from '../../types/index.js';
import type { TextTokenizer, TokenizerSpec } from '../../tokenizers/index.js';

export type QwenExecutionBackend = 'webgpu' | 'wasm';
export type QwenCacheOutputLocation = 'cpu' | 'gpu-buffer';

/** The fixed graph contract used by the browser ONNX conversion. */
export interface Qwen3AsrGraphContract {
  readonly numLayers: number;
  readonly numKvHeads: number;
  readonly headDim: number;
  readonly hiddenSize: number;
  readonly vocabularySize: number;
  readonly audioWindowFrames: number;
  readonly audioTokensPerWindow: number;
  readonly audioFramesMultiple: number;
  readonly batchSize: number;
  readonly pastSeedLength: number;
  readonly pastSeedValue: number;
  readonly pastSeedAttentionMask: number;
  readonly eosTokenIds: readonly number[];
  readonly padTokenId: number;
  readonly audioPadTokenId: number;
  readonly audioStartTokenId: number;
  readonly audioEndTokenId: number;
  readonly imStartTokenId: number;
  readonly imEndTokenId: number;
  readonly logitsOutputLocation: 'cpu';
  readonly cacheOutputLocation: QwenCacheOutputLocation;
}

export interface Qwen3AsrModelConfig {
  readonly ecosystem: 'qwen';
  readonly architecture: 'qwen3-asr';
  readonly processorArchitecture: 'qwen3-asr-mel';
  readonly encoderArchitecture: 'qwen3-asr-audio-encoder';
  readonly decoderArchitecture: 'qwen3-asr-qwen3-decoder';
  readonly sampleRate: number;
  readonly melBins: number;
  readonly hopLength: number;
  readonly nFft: number;
  readonly minInputSamples: number;
  readonly maxInputDurationSec: number;
  readonly languages: readonly string[];
  readonly tokenizer: TokenizerSpec;
  readonly graph: Qwen3AsrGraphContract;
}

export interface Qwen3AsrDirectArtifacts {
  readonly encoderUrl: string;
  readonly decoderUrl: string;
  readonly tokenizerUrl: string;
  readonly encoderDataUrl?: string;
  readonly decoderDataUrl?: string;
  /** ONNX external-data `location` values, normally the data-file basenames. */
  readonly encoderDataPath?: string;
  readonly decoderDataPath?: string;
  readonly processorConfigUrl?: string;
  readonly manifestUrl?: string;
}

export interface Qwen3AsrHuggingFaceSource {
  readonly kind: 'huggingface';
  readonly repoId: string;
  readonly revision?: string;
  readonly cacheKeyFallbackRevisions?: readonly string[];
  readonly encoderPath?: string;
  readonly decoderPath?: string;
  readonly tokenizerPath?: string;
  readonly encoderDataPath?: string;
  readonly decoderDataPath?: string;
  readonly encoderBackend?: QwenExecutionBackend;
  readonly decoderBackend?: QwenExecutionBackend;
  readonly cacheOutputLocation?: QwenCacheOutputLocation;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export interface Qwen3AsrDirectArtifactSource {
  readonly kind: 'direct';
  readonly artifacts: Qwen3AsrDirectArtifacts;
  readonly encoderBackend?: QwenExecutionBackend;
  readonly decoderBackend?: QwenExecutionBackend;
  readonly cacheOutputLocation?: QwenCacheOutputLocation;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export type Qwen3AsrArtifactSource =
  | Qwen3AsrDirectArtifactSource
  | Qwen3AsrHuggingFaceSource;

export interface Qwen3AsrModelOptions {
  readonly config?: Partial<Qwen3AsrModelConfig>;
  readonly source?: Qwen3AsrArtifactSource;
}

export interface Qwen3AsrTranscriptionOptions extends BaseTranscriptionOptions {
  readonly context?: string;
  readonly maxNewTokens?: number;
  readonly returnSpecialTokens?: boolean;
  readonly cacheOutputLocation?: QwenCacheOutputLocation;
}

export interface Qwen3AsrNativeToken {
  readonly index: number;
  readonly id: number;
  readonly text: string;
  readonly special: boolean;
}

export interface Qwen3AsrNativeSegment {
  readonly index: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
}

export interface Qwen3AsrNativeTranscript {
  readonly utteranceText: string;
  readonly isFinal: boolean;
  readonly language?: string;
  readonly rawText?: string;
  readonly tokens?: readonly Qwen3AsrNativeToken[];
  readonly segments?: readonly Qwen3AsrNativeSegment[];
  readonly metrics?: TranscriptMetrics;
  readonly warnings?: readonly TranscriptWarning[];
}

export interface Qwen3AsrFeatureResult {
  readonly features: Float32Array;
  readonly inputFeaturesMask: Int32Array;
  readonly nMels: number;
  readonly frameCount: number;
  readonly validFrameCount: number;
  readonly durationSeconds: number;
  readonly sampleRate: number;
}

export interface Qwen3AsrModelDependencies {
  readonly executor?: Qwen3AsrExecutor;
  readonly tokenizer?: TextTokenizer;
  readonly featureProcessor?: {
    process(audio: AudioBufferLike): Qwen3AsrFeatureResult;
  };
  readonly assetProvider?: AssetProvider;
  readonly runtimeHooks?: SpeechRuntimeHooks;
}

export interface Qwen3AsrExecutor {
  ready?(): Promise<void> | void;
  transcribe(
    audio: AudioBufferLike,
    options: Qwen3AsrTranscriptionOptions,
    context: {
      readonly modelId: string;
      readonly classification: ModelClassification;
      readonly config: Qwen3AsrModelConfig;
    },
  ): Promise<Qwen3AsrNativeTranscript>;
  dispose(): Promise<void> | void;
}
