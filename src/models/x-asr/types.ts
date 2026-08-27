import type {
  AssetProvider,
  BaseTranscriptionOptions,
  SpeechRuntimeHooks,
  TranscriptMetrics,
} from '../../types/index.js';

export type XAsrChunkVariant = 160 | 480 | 960 | 1920;
export type XAsrTensorType = 'float32' | 'float16' | 'int32' | 'int64';

export interface XAsrStateTensorSpec {
  readonly name: string;
  readonly type: XAsrTensorType;
  readonly dims: readonly number[];
}

/** Explicit graph contract required because ONNX state shapes are model-specific. */
export interface XAsrGraphContract {
  readonly featureInputName?: string;
  readonly encoderOutputName?: string;
  readonly encoderStateInputs: readonly XAsrStateTensorSpec[];
  readonly encoderStateOutputs?: readonly string[];
  readonly encoderFrameSize: number;
  readonly encoderFrameShift: number;
  readonly decoderInputName?: string;
  readonly decoderOutputName?: string;
  readonly decoderContextSize: number;
  readonly joinerEncoderInputName?: string;
  readonly joinerDecoderInputName?: string;
  readonly joinerOutputName?: string;
  /** Encoder input names are model-specific; the deployment manifest must supply these. */
}

export interface XAsrModelConfig {
  readonly ecosystem: 'x-asr';
  readonly architecture: 'zipformer2-streaming-rnnt';
  readonly processorArchitecture: 'kaldi-fbank';
  readonly encoderArchitecture: 'zipformer2';
  readonly decoderArchitecture: 'stateless-rnnt';
  readonly sampleRate: 16000;
  readonly featureDim: 80;
  readonly featureHopSeconds: 0.01;
  readonly rawStride: 1;
  readonly vocabularySize?: number;
  readonly languages: readonly ['zh', 'en'];
  readonly chunkMs: XAsrChunkVariant;
  readonly graph: XAsrGraphContract;
}

export interface XAsrDirectArtifacts {
  readonly encoderUrl: string;
  readonly decoderUrl: string;
  readonly joinerUrl: string;
  readonly tokenizerUrl: string;
  readonly encoderDataUrl?: string;
  readonly decoderDataUrl?: string;
  readonly joinerDataUrl?: string;
  readonly encoderDataFilename?: string;
  readonly decoderDataFilename?: string;
  readonly joinerDataFilename?: string;
}

export type XAsrArtifactSource =
  | {
      readonly kind: 'direct';
      readonly artifacts: XAsrDirectArtifacts;
      readonly wasmPaths?: string;
      readonly cpuThreads?: number;
      readonly enableProfiling?: boolean;
    }
  | {
      readonly kind: 'huggingface';
      readonly repoId: string;
      readonly revision?: string;
      readonly subfolder?: string;
      readonly encoderFilename?: string;
      readonly decoderFilename?: string;
      readonly joinerFilename?: string;
      readonly tokenizerFilename?: string;
      readonly encoderDataFilename?: string;
      readonly decoderDataFilename?: string;
      readonly joinerDataFilename?: string;
      readonly wasmPaths?: string;
      readonly cpuThreads?: number;
      readonly enableProfiling?: boolean;
    };

export interface XAsrModelOptions {
  readonly source?: XAsrArtifactSource;
  readonly config?: Partial<Omit<XAsrModelConfig, 'graph'>> & {
    readonly graph?: Partial<XAsrGraphContract>;
  };
}

export interface XAsrNativeToken {
  readonly index: number;
  readonly id?: number;
  readonly text: string;
  readonly startTime?: number;
  readonly endTime?: number;
  readonly confidence?: number;
  readonly logitIndex?: number;
}

export interface XAsrNativeWord {
  readonly index: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
  readonly confidence?: number;
}

/** Native output owned by the X-ASR RNN-T family, independent of CTC output contracts. */
export interface XAsrNativeTranscript {
  readonly utteranceText: string;
  readonly isFinal: boolean;
  readonly words?: readonly XAsrNativeWord[];
  readonly tokens?: readonly XAsrNativeToken[];
  readonly confidence?: {
    readonly utterance?: number;
    readonly tokenAverage?: number;
    readonly wordAverage?: number;
  };
  readonly metrics?: TranscriptMetrics;
  readonly warnings?: readonly { readonly code: string; readonly message: string }[];
}

export interface XAsrTranscriptionOptions extends BaseTranscriptionOptions {
  readonly returnTokenIds?: boolean;
  readonly returnLogitIndices?: boolean;
  readonly returnFrameIds?: boolean;
}

export interface XAsrModelDependencies {
  readonly executor?: import('./executor.js').XAsrExecutor;
  readonly assetProvider?: AssetProvider;
  readonly runtimeHooks?: SpeechRuntimeHooks;
  readonly signal?: import('../../types/index.js').AbortSignalLike | null;
}
