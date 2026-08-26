import type { AssetProvider, SpeechRuntimeHooks } from '../../types/index.js';
import type {
  LasrCtcNativeTranscript,
  LasrCtcTranscriptionOptions,
} from '../lasr-ctc/types.js';

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

export type XAsrNativeTranscript = LasrCtcNativeTranscript;
export type XAsrTranscriptionOptions = LasrCtcTranscriptionOptions;

export interface XAsrModelDependencies {
  readonly executor?: import('./executor.js').XAsrExecutor;
  readonly assetProvider?: AssetProvider;
  readonly runtimeHooks?: SpeechRuntimeHooks;
}
