import type { AudioBufferLike, BaseTranscriptionOptions } from '../../types/index.js';
import type {
  NemoDecodeContext,
  NemoFeatureExtractor,
  NemoModelConfig,
  NemoModelDependencies,
  NemoModelOptions,
  NemoNativeToken,
  NemoNativeTranscript
} from '../nemo-common/index.js';

export interface NemoTdtModelConfig extends NemoModelConfig {
  readonly ecosystem: 'nemo';
  readonly architecture: 'nemo-tdt';
  readonly decoderArchitecture: 'tdt';
  readonly encoderArchitecture: string;
  readonly predictionHiddenSize?: number;
  readonly predictionLayers?: number;
  readonly maxSymbolsPerStep?: number;
}

export type NemoTdtQuantization = 'int8' | 'fp32' | 'fp16';
export type NemoTdtPreprocessorBackend = 'js' | 'onnx';

export interface NemoTdtDirectArtifacts {
  readonly encoderUrl: string;
  readonly decoderUrl: string;
  readonly tokenizerUrl: string;
  readonly preprocessorUrl?: string;
  readonly encoderDataUrl?: string;
  readonly decoderDataUrl?: string;
  readonly encoderFilename?: string;
  readonly decoderFilename?: string;
}

export interface NemoTdtDirectArtifactSource {
  readonly kind: 'direct';
  readonly artifacts: NemoTdtDirectArtifacts;
  readonly preprocessorBackend?: NemoTdtPreprocessorBackend;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export interface NemoTdtHuggingFaceSource {
  readonly kind: 'huggingface';
  readonly repoId: string;
  readonly revision?: string;
  readonly encoderQuant?: NemoTdtQuantization;
  readonly decoderQuant?: NemoTdtQuantization;
  readonly preprocessorName?: 'nemo80' | 'nemo128';
  readonly preprocessorBackend?: NemoTdtPreprocessorBackend;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export type NemoTdtArtifactSource =
  | NemoTdtDirectArtifactSource
  | NemoTdtHuggingFaceSource;

export interface NemoTdtModelOptions extends NemoModelOptions<NemoTdtModelConfig> {
  readonly source?: NemoTdtArtifactSource;
}

export interface NemoTdtDecoderStateSnapshot {
  readonly s1: Float32Array;
  readonly s2: Float32Array;
  readonly dims1: readonly number[];
  readonly dims2: readonly number[];
}

export interface NemoTdtNativeToken extends NemoNativeToken {
  readonly frameIndex?: number;
  readonly logProb?: number;
  readonly tdtStep?: number;
}

export interface NemoTdtNativeTranscript extends Omit<NemoNativeTranscript, 'tokens'> {
  readonly tokens?: readonly NemoTdtNativeToken[];
  readonly decoderState?: NemoTdtDecoderStateSnapshot;
  readonly debug?: {
    readonly tokenIds?: readonly number[];
    readonly frameIndices?: readonly number[];
    readonly logProbs?: readonly number[];
    readonly tdtSteps?: readonly number[];
  };
}

export interface NemoTdtTranscriptionOptions extends BaseTranscriptionOptions {
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

export interface NemoTdtDecoder {
  decode(
    features: Awaited<ReturnType<NemoFeatureExtractor<NemoTdtModelConfig>['compute']>>,
    options: NemoTdtTranscriptionOptions,
    context: NemoDecodeContext<NemoTdtModelConfig>
  ): Promise<NemoTdtNativeTranscript> | NemoTdtNativeTranscript;
}

export interface NemoTdtExecutor {
  transcribe(
    audio: AudioBufferLike,
    options: NemoTdtTranscriptionOptions,
    context: NemoDecodeContext<NemoTdtModelConfig>
  ): Promise<NemoTdtNativeTranscript>;
  dispose(): Promise<void> | void;
}

export interface NemoTdtModelDependencies
  extends NemoModelDependencies<NemoTdtModelConfig, NemoTdtNativeTranscript, NemoTdtTranscriptionOptions> {
  readonly featureExtractor?: NemoFeatureExtractor<NemoTdtModelConfig>;
  readonly decoder?: NemoTdtDecoder;
  readonly executor?: NemoTdtExecutor;
}
