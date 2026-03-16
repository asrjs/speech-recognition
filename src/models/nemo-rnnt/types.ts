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
} from '../nemo-common/index.js';
import type {
  NemoTdtArtifactSource,
  NemoTdtDecoderStateSnapshot,
  NemoTdtExecutionBackend,
  NemoTdtPreprocessorBackend,
  NemoTdtQuantization,
} from '../nemo-tdt/types.js';

export interface NemoRnntModelConfig extends NemoModelConfig {
  readonly ecosystem: 'nemo';
  readonly architecture: 'nemo-rnnt';
  readonly decoderArchitecture: 'rnnt';
  readonly encoderArchitecture: string;
  readonly predictionHiddenSize?: number;
  readonly predictionLayers?: number;
  readonly maxSymbolsPerStep?: number;
}

export type NemoRnntQuantization = NemoTdtQuantization;
export type NemoRnntPreprocessorBackend = NemoTdtPreprocessorBackend;
export type NemoRnntExecutionBackend = NemoTdtExecutionBackend;
export type NemoRnntArtifactSource = NemoTdtArtifactSource;

export interface NemoRnntModelOptions extends NemoModelOptions<NemoRnntModelConfig> {
  readonly source?: NemoRnntArtifactSource;
}

export interface NemoRnntNativeToken extends NemoNativeToken {
  readonly frameIndex?: number;
  readonly logProb?: number;
  readonly tdtStep?: number;
}

export interface NemoRnntNativeSpecialToken extends NemoRnntNativeToken {
  readonly kind: 'eou' | 'eob' | 'control';
}

export interface NemoRnntNativeTranscript extends Omit<NemoNativeTranscript, 'tokens'> {
  readonly rawUtteranceText?: string;
  readonly tokens?: readonly NemoRnntNativeToken[];
  readonly specialTokens?: readonly NemoRnntNativeSpecialToken[];
  readonly control?: {
    readonly containsEou: boolean;
    readonly containsEob: boolean;
    readonly eouTokenId?: number;
    readonly eobTokenId?: number;
  };
  readonly decoderState?: NemoTdtDecoderStateSnapshot;
  readonly debug?: {
    readonly tokenIds?: readonly number[];
    readonly frameIndices?: readonly number[];
    readonly logProbs?: readonly number[];
    readonly tdtSteps?: readonly number[];
  };
}

export interface NemoRnntTranscriptionOptions extends BaseTranscriptionOptions {
  readonly temperature?: number;
  readonly returnTokenIds?: boolean;
  readonly returnFrameIndices?: boolean;
  readonly returnLogProbs?: boolean;
  readonly returnTdtSteps?: boolean;
  readonly returnDecoderState?: boolean;
  readonly incremental?: {
    readonly cacheKey: string;
    readonly prefixSeconds: number;
  } | null;
}

export interface NemoRnntDecoder {
  decode(
    features: Awaited<ReturnType<NemoFeatureExtractor<NemoRnntModelConfig>['compute']>>,
    options: NemoRnntTranscriptionOptions,
    context: NemoDecodeContext<NemoRnntModelConfig>,
  ): Promise<NemoRnntNativeTranscript> | NemoRnntNativeTranscript;
}

export interface NemoRnntExecutor {
  ready?(): Promise<void> | void;
  transcribe(
    audio: AudioBufferLike,
    options: NemoRnntTranscriptionOptions,
    context: NemoDecodeContext<NemoRnntModelConfig>,
  ): Promise<NemoRnntNativeTranscript>;
  dispose(): Promise<void> | void;
}

export interface NemoRnntModelDependencies extends NemoModelDependencies<
  NemoRnntModelConfig,
  NemoRnntNativeTranscript,
  NemoRnntTranscriptionOptions
> {
  readonly featureExtractor?: NemoFeatureExtractor<NemoRnntModelConfig>;
  readonly decoder?: NemoRnntDecoder;
  readonly executor?: NemoRnntExecutor;
  readonly assetProvider?: AssetProvider;
  readonly runtimeHooks?: SpeechRuntimeHooks;
}
