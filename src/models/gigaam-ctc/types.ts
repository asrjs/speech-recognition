import type {
  LasrCtcArtifactSource,
  LasrCtcModelConfig,
  LasrCtcModelOptions,
} from '../lasr-ctc/types.js';
import type {
  AssetProvider,
  AudioInputLike,
  SpeechBatchSession,
  SpeechRuntimeHooks,
  TranscriptResponse,
  TranscriptResponseFlavor,
} from '../../types/index.js';
import type { LasrCtcNativeTranscript, LasrCtcTranscriptionOptions } from '../lasr-ctc/types.js';

export interface GigaAmModelConfig extends Omit<
  LasrCtcModelConfig,
  'ecosystem' | 'architecture' | 'processorArchitecture' | 'encoderArchitecture'
> {
  readonly ecosystem: 'gigaam';
  readonly architecture: 'gigaam-ctc';
  readonly processorArchitecture: 'gigaam-fbank';
  readonly encoderArchitecture: 'gigaam-conformer';
  readonly nFft: number;
  readonly winLength: number;
  readonly hopLength: number;
  readonly center?: boolean;
  readonly featureLayout: 'mel-major';
}

export type GigaAmArtifactSource = LasrCtcArtifactSource;
export type GigaAmModelOptions = Omit<LasrCtcModelOptions, 'config' | 'source'> & {
  readonly source?: GigaAmArtifactSource;
  readonly config?: Partial<GigaAmModelConfig>;
};

export interface GigaAmModelFamilyOptions {
  readonly dependencies?: {
    readonly executor?: import('./executor.js').OrtGigaAmCtcExecutor;
    readonly assetProvider?: AssetProvider;
    readonly runtimeHooks?: SpeechRuntimeHooks;
    readonly signal?: import('../../types/index.js').AbortSignalLike | null;
  };
}

export interface GigaAmBatchSession extends SpeechBatchSession<
  LasrCtcTranscriptionOptions,
  LasrCtcNativeTranscript
> {
  transcribeBatch<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    audio: readonly AudioInputLike[],
    options?: LasrCtcTranscriptionOptions & { readonly responseFlavor?: TFlavor },
  ): Promise<readonly TranscriptResponse<LasrCtcNativeTranscript, TFlavor>[]>;
}
