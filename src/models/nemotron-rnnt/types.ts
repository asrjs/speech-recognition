import type {
  AbortSignalLike,
  AssetProvider,
  BaseTranscriptionOptions,
  SpeechRuntimeHooks,
} from '../../types/index.js';
import type {
  NemoDecodeContext,
  NemoModelConfig,
  NemoModelDependencies,
  NemoModelOptions,
  NemoNativeToken,
  NemoNativeTranscript,
  NemoNativeWord,
} from '../nemo-common/index.js';
import type { NemoTdtExecutionBackend, NemoTdtPreprocessorBackend } from '../nemo-tdt/types.js';

/**
 * Prompt/language IDs accepted by the Nemotron 3.5 ASR Streaming 0.6B
 * encoder. These constants match the published NeMo contract: `auto=101`
 * selects language from audio, `en=0` forces English, `tr=18` forces
 * Turkish. The decoder emits `<en-US>`/`<tr-TR>` segments under that
 * choice; downstream token-id 13087 is the per-step blank.
 */
export interface NemotronRnntPromptIds {
  readonly auto: number;
  readonly en: number;
  readonly tr: number;
}

/**
 * Encoder cache geometry for the cache-aware streaming encoder. Values
 * come from the published ONNX contract (24 Conformer layers, 56-frame
 * channel cache, 8-frame time cache, 1024-dim hidden). Library code must
 * allocate the cache tensors exactly once at session start and reuse
 * them across chunks to match the encoder's streaming contract.
 */
export interface NemotronRnntEncoderCache {
  readonly channelLayers: number;
  readonly channelFrames: number;
  readonly channelDim: number;
  readonly timeLayers: number;
  readonly timeFrames: number;
  readonly timeDim: number;
}

export interface NemotronRnntModelConfig extends NemoModelConfig {
  readonly ecosystem: 'nemo';
  readonly architecture: 'nemotron-rnnt';
  readonly decoderArchitecture: 'rnnt';
  readonly encoderArchitecture: 'fastconformer';
  readonly predictionHiddenSize: number;
  readonly predictionLayers: number;
  readonly chunkFrames: number;
  readonly encoderOutputFramesPerChunk: number;
  readonly encoderCache: NemotronRnntEncoderCache;
  readonly promptIds: NemotronRnntPromptIds;
  readonly defaultPromptId: number;
  readonly maxDecodeSteps: number;
  readonly maxOutputTokens: number;
  readonly blankTokenId: number;
}

export type NemotronRnntPreprocessorBackend = NemoTdtPreprocessorBackend;
export type NemotronRnntExecutionBackend = NemoTdtExecutionBackend;

export interface NemotronRnntDirectArtifacts {
  readonly encoderUrl: string;
  readonly decoderUrl: string;
  readonly jointUrl: string;
  readonly tokenizerUrl: string;
  readonly encoderDataUrl?: string;
  readonly decoderDataUrl?: string;
  readonly jointDataUrl?: string;
  readonly encoderFilename?: string;
  readonly decoderFilename?: string;
  readonly jointFilename?: string;
}

export interface NemotronRnntHuggingFaceSource {
  readonly kind: 'huggingface';
  readonly repoId: string;
  readonly revision?: string;
  readonly cacheKeyFallbackRevisions?: readonly string[];
  readonly encoderBackend?: NemotronRnntExecutionBackend;
  readonly preprocessorBackend?: NemotronRnntPreprocessorBackend;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export interface NemotronRnntDirectArtifactSource {
  readonly kind: 'direct';
  readonly artifacts: NemotronRnntDirectArtifacts;
  readonly encoderBackend?: NemotronRnntExecutionBackend;
  readonly preprocessorBackend?: NemotronRnntPreprocessorBackend;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export type NemotronRnntArtifactSource =
  | NemotronRnntHuggingFaceSource
  | NemotronRnntDirectArtifactSource;

export interface NemotronRnntModelOptions extends NemoModelOptions<NemotronRnntModelConfig> {
  readonly source?: NemotronRnntArtifactSource;
}

export type NemotronRnntNativeWord = NemoNativeWord;

export interface NemotronRnntNativeToken extends NemoNativeToken {
  readonly frameIndex?: number;
  readonly logProb?: number;
}

export interface NemotronRnntNativeSpecialToken extends NemotronRnntNativeToken {
  readonly kind: 'lang-segment' | 'control';
}

export interface NemotronRnntNativeTranscript
  extends Omit<NemoNativeTranscript, 'tokens'> {
  readonly rawUtteranceText?: string;
  readonly tokens?: readonly NemotronRnntNativeToken[];
  readonly specialTokens?: readonly NemotronRnntNativeSpecialToken[];
  readonly control?: {
    readonly containsLangSegment: boolean;
    readonly langSegmentTokenIds?: readonly number[];
  };
  readonly debug?: {
    readonly tokenIds?: readonly number[];
    readonly frameIndices?: readonly number[];
    readonly logProbs?: readonly number[];
  };
}

export interface NemotronRnntTranscriptionOptions extends BaseTranscriptionOptions {
  /**
   * Language ID forwarded to the encoder's `lang_id` input. Defaults to
   * the configured default prompt id (auto).
   */
  readonly promptId?: number;
  readonly returnTokenIds?: boolean;
  readonly returnFrameIndices?: boolean;
  readonly returnLogProbs?: boolean;
}

export interface NemotronRnntDecoder {
  decode(
    features: {
      readonly features: Float32Array;
      readonly frameCount: number;
      readonly durationSeconds: number;
    },
    options: NemotronRnntTranscriptionOptions,
    context: NemoDecodeContext<NemotronRnntModelConfig>,
  ): Promise<NemotronRnntNativeTranscript> | NemotronRnntNativeTranscript;
}

export interface NemotronRnntExecutor {
  ready?(): Promise<void> | void;
  transcribe(
    audio: {
      readonly sampleRate: number;
      readonly durationSeconds: number;
      readonly channels?: ReadonlyArray<Float32Array>;
    },
    options: NemotronRnntTranscriptionOptions,
    context: NemoDecodeContext<NemotronRnntModelConfig>,
  ): Promise<NemotronRnntNativeTranscript>;
  dispose(): Promise<void> | void;
}

export interface NemotronRnntModelDependencies extends NemoModelDependencies<
  NemotronRnntModelConfig,
  NemotronRnntNativeTranscript,
  NemotronRnntTranscriptionOptions
> {
  readonly executor?: NemotronRnntExecutor;
  readonly decoder?: NemotronRnntDecoder;
  readonly assetProvider?: AssetProvider;
  readonly runtimeHooks?: SpeechRuntimeHooks;
  readonly signal?: AbortSignalLike | null;
}
