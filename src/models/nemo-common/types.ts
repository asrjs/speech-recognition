import type { AudioProcessor, ProcessorFeatureDescriptor } from '../../processors/index.js';
import type { TextTokenizer, TokenizerSpec } from '../../tokenizers/index.js';
import type {
  BaseTranscriptionOptions,
  ModelClassification,
  PrecisionMode,
  TranscriptMeta,
  TranscriptResult
} from '../../types/index.js';

export interface NemoRuntimeHints {
  readonly encoderPrecision?: PrecisionMode;
  readonly decoderPrecision?: PrecisionMode;
  readonly preprocessorBackend?: 'js' | 'onnx';
}

export interface NemoModelConfig {
  readonly ecosystem: 'nemo';
  readonly architecture: string;
  readonly encoderArchitecture: string;
  readonly decoderArchitecture: string;
  readonly sampleRate: number;
  readonly frameShiftSeconds: number;
  readonly subsamplingFactor: number;
  readonly melBins: number;
  readonly vocabularySize?: number;
  readonly languages: readonly string[];
  readonly tokenizer: TokenizerSpec;
}

export interface NemoModelOptions<TConfig extends NemoModelConfig = NemoModelConfig> {
  readonly modelBaseUrl?: string;
  readonly revision?: string;
  readonly config?: Partial<TConfig>;
  readonly runtimeHints?: NemoRuntimeHints;
}

export interface NemoNativeToken {
  readonly index: number;
  readonly id?: number;
  readonly text: string;
  readonly rawText?: string;
  readonly isWordStart?: boolean;
  readonly startTime?: number;
  readonly endTime?: number;
  readonly confidence?: number;
}

export interface NemoNativeWord {
  readonly index: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
  readonly confidence?: number;
}

export interface NemoNativeConfidence {
  readonly utterance?: number;
  readonly wordAverage?: number;
  readonly tokenAverage?: number;
  readonly frameAverage?: number;
  readonly averageLogProb?: number;
  readonly frames?: readonly number[];
}

export interface NemoNativeTranscriptMetrics {
  readonly preprocessMs?: number;
  readonly encodeMs?: number;
  readonly decodeMs?: number;
  readonly tokenizeMs?: number;
  readonly totalMs?: number;
  readonly rtf?: number;
}

export interface NemoNativeTranscript {
  readonly utteranceText: string;
  readonly isFinal: boolean;
  readonly words?: readonly NemoNativeWord[];
  readonly tokens?: readonly NemoNativeToken[];
  readonly confidence?: NemoNativeConfidence;
  readonly metrics?: NemoNativeTranscriptMetrics;
  readonly warnings?: readonly { readonly code: string; readonly message: string }[];
}

export type NemoFeatureDescriptor = ProcessorFeatureDescriptor;
export type NemoTokenizer = TextTokenizer;
export type NemoFeatureExtractor<TConfig extends NemoModelConfig = NemoModelConfig> =
  AudioProcessor<TConfig, ProcessorFeatureDescriptor>;

export interface NemoDecodeContext<TConfig extends NemoModelConfig = NemoModelConfig> {
  readonly modelId: string;
  readonly classification: ModelClassification;
  readonly config: TConfig;
  readonly tokenizer?: NemoTokenizer;
}

export interface NemoTimestampReconstructor<
  TNative extends NemoNativeTranscript = NemoNativeTranscript,
  TOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions
> {
  reconstruct(nativeTranscript: TNative, detail: TOptions['detail']): Pick<TranscriptResult, 'segments' | 'words' | 'tokens'>;
}

export interface NemoConfidenceReconstructor<TNative extends NemoNativeTranscript = NemoNativeTranscript> {
  summarize(nativeTranscript: TNative): Pick<
    TranscriptMeta,
    'averageConfidence' | 'averageWordConfidence' | 'averageTokenConfidence'
  >;
}

export interface NemoModelDependencies<
  TConfig extends NemoModelConfig = NemoModelConfig,
  TNative extends NemoNativeTranscript = NemoNativeTranscript,
  TOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions
> {
  readonly tokenizer?: NemoTokenizer;
  readonly featureExtractor?: NemoFeatureExtractor<TConfig>;
  readonly timestampReconstructor?: NemoTimestampReconstructor<TNative, TOptions>;
  readonly confidenceReconstructor?: NemoConfidenceReconstructor<TNative>;
}
