import type {
  AssetProvider,
  AudioBufferLike,
  BaseTranscriptionOptions,
  SpeechRuntimeHooks,
} from '../../types/index.js';
import type {
  NemoDecodeContext,
  NemoFeatureExtractor,
  NemoModelConfig,
  NemoModelDependencies,
  NemoModelOptions,
  NemoNativeToken,
  NemoNativeTranscript,
  NemoNativeWord,
} from '../nemo-common/index.js';

export interface NemoAedPromptSettings {
  /**
   * Source language of the input audio. Canary 180M Flash supports `en`, `de`,
   * `es`, and `fr` in the current preset.
   */
  readonly sourceLanguage: string;
  /**
   * Target language for output text. Set equal to `sourceLanguage` for ASR, or
   * change it for speech translation.
   */
  readonly targetLanguage: string;
  /**
   * Optional free-form decoder context slot supported by the prompt format.
   * The current runtime keeps this empty because Canary text-context prompting
   * is not restored yet.
   */
  readonly decoderContext: string;
  /** Canary `canary2` prompt emotion slot, usually `<|emo:undefined|>`. */
  readonly emotion: string;
  /** Whether to request punctuation and capitalization output. */
  readonly punctuate: boolean;
  /** Whether to request inverse text normalization. */
  readonly inverseTextNormalization: boolean;
  /** Whether to request timestamp generation. */
  readonly timestamps: boolean;
  /** Whether to request diarization-aware prompting. */
  readonly diarize: boolean;
}

export interface NemoAedModelConfig extends NemoModelConfig {
  readonly ecosystem: 'nemo';
  readonly architecture: 'nemo-aed';
  readonly decoderArchitecture: 'transformer-decoder';
  readonly encoderArchitecture: string;
  readonly encoderHiddenSize?: number;
  readonly decoderHiddenSize?: number;
  readonly encoderOutputSize?: number;
  readonly maxTargetPositions: number;
  readonly promptFormat: string;
  readonly promptDefaults: readonly NemoAedPromptSettings[];
}

export type NemoAedQuantization = 'int8' | 'fp32' | 'fp16';
export type NemoAedPreprocessorBackend = 'js' | 'onnx';

export interface NemoAedDirectArtifacts {
  readonly encoderUrl: string;
  readonly decoderUrl: string;
  readonly tokenizerUrl: string;
  readonly preprocessorUrl?: string;
  readonly configUrl?: string;
  readonly encoderDataUrl?: string;
  readonly decoderDataUrl?: string;
  readonly encoderFilename?: string;
  readonly decoderFilename?: string;
  readonly tokenizerFilename?: string;
  readonly preprocessorFilename?: string;
  readonly configFilename?: string;
}

export interface NemoAedDirectArtifactSource {
  readonly kind: 'direct';
  readonly artifacts: NemoAedDirectArtifacts;
  readonly preprocessorBackend?: NemoAedPreprocessorBackend;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export interface NemoAedHuggingFaceSource {
  readonly kind: 'huggingface';
  readonly repoId: string;
  readonly revision?: string;
  readonly encoderQuant?: NemoAedQuantization;
  readonly decoderQuant?: NemoAedQuantization;
  readonly preprocessorName?: 'nemo80' | 'nemo128';
  readonly tokenizerName?: string;
  readonly configName?: string;
  readonly preprocessorBackend?: NemoAedPreprocessorBackend;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export type NemoAedArtifactSource = NemoAedDirectArtifactSource | NemoAedHuggingFaceSource;

export interface NemoAedModelOptions extends NemoModelOptions<NemoAedModelConfig> {
  readonly source?: NemoAedArtifactSource;
}

export interface NemoAedNativeToken extends NemoNativeToken {
  readonly logProb?: number;
  readonly special?: boolean;
}

export interface NemoAedNativeTranscript extends Omit<NemoNativeTranscript, 'words' | 'tokens'> {
  readonly language?: string;
  readonly words?: readonly NemoNativeWord[];
  readonly tokens?: readonly NemoAedNativeToken[];
  readonly prompt?: {
    readonly settings: NemoAedPromptSettings;
    readonly ids: readonly number[];
    readonly pieces: readonly string[];
  };
  readonly debug?: {
    readonly tokenIds?: readonly number[];
    readonly promptIds?: readonly number[];
    readonly logProbs?: readonly number[];
  };
}

export interface NemoAedTranscriptionOptions extends BaseTranscriptionOptions {
  /** Shortcut for selecting the source language prompt token. */
  readonly sourceLanguage?: string;
  /** Shortcut for selecting the target language prompt token. */
  readonly targetLanguage?: string;
  /** Optional decoder context text slot. Currently restricted to the empty default. */
  readonly decoderContext?: string;
  /** Optional Canary emotion token value. */
  readonly emotion?: string;
  /** Request punctuation and capitalization in the generated text. */
  readonly punctuate?: boolean;
  /** Request inverse text normalization behavior. */
  readonly inverseTextNormalization?: boolean;
  /** Request timestamp output if the runtime/backend supports it. */
  readonly timestamps?: boolean;
  /** Request diarization-aware prompting. */
  readonly diarize?: boolean;
  /** Upper bound on newly generated decoder tokens. */
  readonly maxNewTokens?: number;
  readonly returnTokenIds?: boolean;
  readonly returnPromptIds?: boolean;
  readonly returnLogProbs?: boolean;
}

export interface NemoAedDecoder {
  decode(
    features: Awaited<ReturnType<NemoFeatureExtractor<NemoAedModelConfig>['compute']>>,
    options: NemoAedTranscriptionOptions,
    context: NemoDecodeContext<NemoAedModelConfig>,
  ): Promise<NemoAedNativeTranscript> | NemoAedNativeTranscript;
}

export interface NemoAedExecutor {
  ready?(): Promise<void> | void;
  transcribe(
    audio: AudioBufferLike,
    options: NemoAedTranscriptionOptions,
    context: NemoDecodeContext<NemoAedModelConfig>,
  ): Promise<NemoAedNativeTranscript>;
  dispose(): Promise<void> | void;
}

export interface NemoAedModelDependencies extends NemoModelDependencies<
  NemoAedModelConfig,
  NemoAedNativeTranscript,
  NemoAedTranscriptionOptions
> {
  readonly featureExtractor?: NemoFeatureExtractor<NemoAedModelConfig>;
  readonly decoder?: NemoAedDecoder;
  readonly executor?: NemoAedExecutor;
  readonly assetProvider?: AssetProvider;
  readonly runtimeHooks?: SpeechRuntimeHooks;
}
