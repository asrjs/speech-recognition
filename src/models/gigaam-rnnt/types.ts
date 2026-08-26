import type { AssetProvider, SpeechRuntimeHooks } from '../../types/index.js';
import type { LasrCtcArtifactSource, LasrCtcModelConfig, LasrCtcModelOptions, LasrCtcNativeTranscript, LasrCtcTranscriptionOptions } from '../lasr-ctc/types.js';

export interface GigaAmRnntModelConfig extends Omit<LasrCtcModelConfig, 'ecosystem' | 'architecture' | 'processorArchitecture' | 'encoderArchitecture' | 'decoderArchitecture'> {
  readonly ecosystem: 'gigaam';
  readonly architecture: 'gigaam-rnnt';
  readonly processorArchitecture: 'gigaam-fbank';
  readonly encoderArchitecture: 'gigaam-conformer';
  readonly decoderArchitecture: 'rnnt';
  readonly nFft: 320;
  readonly winLength: 320;
  readonly hopLength: 160;
  readonly featureLayout: 'mel-major';
  readonly predictionHiddenSize: number;
  readonly maxTokensPerFrame: number;
}

export interface GigaAmRnntDirectArtifacts {
  readonly encoderUrl: string;
  readonly decoderUrl: string;
  readonly jointUrl: string;
  readonly tokenizerUrl: string;
  readonly encoderDataUrl?: string;
  readonly decoderDataUrl?: string;
  readonly jointDataUrl?: string;
  readonly encoderDataFilename?: string;
  readonly decoderDataFilename?: string;
  readonly jointDataFilename?: string;
}

export type GigaAmRnntArtifactSource =
  | (Omit<Extract<LasrCtcArtifactSource, { kind: 'direct' }>, 'artifacts'> & { readonly kind: 'direct'; readonly artifacts: GigaAmRnntDirectArtifacts })
  | (Omit<Extract<LasrCtcArtifactSource, { kind: 'huggingface' }>, 'modelFilename' | 'tokenizerFilename'> & {
      readonly kind: 'huggingface';
      readonly encoderFilename?: string;
      readonly decoderFilename?: string;
      readonly jointFilename?: string;
      readonly tokenizerFilename?: string;
      readonly encoderDataFilename?: string;
      readonly decoderDataFilename?: string;
      readonly jointDataFilename?: string;
    });

export type GigaAmRnntModelOptions = Omit<LasrCtcModelOptions, 'config' | 'source'> & {
  readonly source?: GigaAmRnntArtifactSource;
  readonly config?: Partial<GigaAmRnntModelConfig>;
};

export interface GigaAmRnntModelFamilyOptions {
  readonly dependencies?: {
    readonly executor?: import('./executor.js').OrtGigaAmRnntExecutor;
    readonly assetProvider?: AssetProvider;
    readonly runtimeHooks?: SpeechRuntimeHooks;
  };
}

export type GigaAmRnntNativeTranscript = LasrCtcNativeTranscript;
export type GigaAmRnntTranscriptionOptions = LasrCtcTranscriptionOptions;
