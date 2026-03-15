import type { TokenizerSpec, TextTokenizer } from '../../tokenizers/index.js';
import type {
  AssetProvider,
  AudioBufferLike,
  BaseTranscriptionOptions,
  ModelClassification,
  SpeechRuntimeHooks,
  TranscriptMetrics,
} from '../../types/index.js';

export interface LasrCtcModelConfig {
  readonly ecosystem: 'lasr';
  readonly architecture: 'lasr-ctc';
  readonly processorArchitecture: 'kaldi-mel';
  readonly encoderArchitecture: 'conformer';
  readonly decoderArchitecture: 'ctc';
  readonly sampleRate: number;
  readonly rawStride: number;
  readonly nMels: number;
  readonly featureHopSeconds: number;
  readonly vocabularySize?: number;
  readonly languages: readonly string[];
  readonly tokenizer: TokenizerSpec;
}

export interface LasrCtcDirectArtifacts {
  readonly modelUrl: string;
  readonly tokenizerUrl: string;
  readonly modelDataUrl?: string;
  readonly modelDataFilename?: string;
}

export interface LasrCtcDirectArtifactSource {
  readonly kind: 'direct';
  readonly artifacts: LasrCtcDirectArtifacts;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export interface LasrCtcHuggingFaceSource {
  readonly kind: 'huggingface';
  readonly repoId: string;
  readonly revision?: string;
  readonly subfolder?: string;
  readonly modelFilename?: string;
  readonly modelDataFilename?: string;
  readonly tokenizerFilename?: string;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export type LasrCtcArtifactSource = LasrCtcDirectArtifactSource | LasrCtcHuggingFaceSource;

export interface LasrCtcModelOptions {
  readonly modelBaseUrl?: string;
  readonly revision?: string;
  readonly source?: LasrCtcArtifactSource;
  readonly config?: Partial<LasrCtcModelConfig>;
}

export interface LasrCtcNativeToken {
  readonly index: number;
  readonly id?: number;
  readonly text: string;
  readonly startTime?: number;
  readonly endTime?: number;
  readonly confidence?: number;
  readonly logitIndex?: number;
}

export interface LasrCtcTokenSpan {
  readonly tokenId: number;
  readonly text: string;
  readonly startFrame: number;
  readonly endFrame: number;
  readonly frameCount: number;
  readonly startTime: number;
  readonly endTime: number;
  readonly duration: number;
  readonly confidence: number;
  readonly averageLogProb: number;
}

export interface LasrCtcUtteranceTiming {
  readonly hasSpeech: boolean;
  readonly startFrame: number | null;
  readonly endFrame: number | null;
  readonly startTime: number;
  readonly endTime: number;
  readonly duration: number;
  readonly confidence: number;
}

export interface LasrCtcSentenceTiming {
  readonly text: string;
  readonly startTokenIndex: number;
  readonly endTokenIndex: number;
  readonly startFrame: number;
  readonly endFrame: number;
  readonly startTime: number;
  readonly endTime: number;
  readonly duration: number;
  readonly confidence: number;
}

export interface LasrCtcNativeWord {
  readonly index: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
  readonly confidence?: number;
}

export interface LasrCtcNativeTranscript {
  readonly utteranceText: string;
  readonly isFinal: boolean;
  readonly words?: readonly LasrCtcNativeWord[];
  readonly tokens?: readonly LasrCtcNativeToken[];
  readonly confidence?: {
    readonly utterance?: number;
    readonly tokenAverage?: number;
    readonly wordAverage?: number;
  };
  readonly metrics?: TranscriptMetrics;
  readonly ctc?: {
    readonly frameIds?: readonly number[];
    readonly collapsedIds?: readonly number[];
    readonly secondsPerFrame?: number;
    readonly utterance?: LasrCtcUtteranceTiming;
    readonly tokenSpans?: readonly LasrCtcTokenSpan[];
    readonly sentences?: readonly LasrCtcSentenceTiming[];
  };
  readonly warnings?: readonly { readonly code: string; readonly message: string }[];
}

export interface LasrCtcTranscriptionOptions extends BaseTranscriptionOptions {
  readonly returnTokenIds?: boolean;
  readonly returnLogitIndices?: boolean;
  readonly returnFrameIds?: boolean;
}

export interface LasrCtcFeatureBatch {
  readonly features: Float32Array;
  readonly frameCount: number;
  readonly featureSize: number;
}

export interface LasrCtcFeaturePreprocessor {
  process(audio: Float32Array): LasrCtcFeatureBatch;
}

export interface LasrCtcExecutorContext {
  readonly modelId: string;
  readonly classification: ModelClassification;
  readonly config: LasrCtcModelConfig;
  readonly tokenizer: TextTokenizer;
}

export interface LasrCtcExecutor {
  ready?(): Promise<void> | void;
  transcribe(
    audio: AudioBufferLike,
    options: LasrCtcTranscriptionOptions,
    context: LasrCtcExecutorContext,
  ): Promise<LasrCtcNativeTranscript>;
  dispose(): Promise<void> | void;
}

export interface LasrCtcModelDependencies {
  readonly tokenizer?: TextTokenizer;
  readonly preprocessor?: LasrCtcFeaturePreprocessor;
  readonly executor?: LasrCtcExecutor;
  readonly assetProvider?: AssetProvider;
  readonly runtimeHooks?: SpeechRuntimeHooks;
}
