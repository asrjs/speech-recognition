import type { AssetProvider, SpeechRuntimeHooks } from '../../types/index.js';
import type { LasrCtcArtifactSource, LasrCtcModelConfig, LasrCtcModelOptions, LasrCtcNativeTranscript, LasrCtcTranscriptionOptions } from '../lasr-ctc/types.js';

export type GigaAmRnntExecutionBackend = 'webgpu' | 'wasm';

export interface GigaAmRnntBackendSelection {
  readonly encoderBackend?: GigaAmRnntExecutionBackend;
  readonly decoderBackend?: GigaAmRnntExecutionBackend;
  readonly jointBackend?: GigaAmRnntExecutionBackend;
  /**
   * Experimental all-WebGPU startup probe. Mixed and WASM compositions stay
   * serial because ORT may initialize WASM fallback kernels from a WebGPU
   * session and rejects concurrent provider initialization.
   */
  readonly parallelSessionInitialization?: boolean;
}

export interface ResolvedGigaAmRnntBackends {
  readonly ortBackend: GigaAmRnntExecutionBackend;
  readonly encoderBackend: GigaAmRnntExecutionBackend;
  readonly decoderBackend: GigaAmRnntExecutionBackend;
  readonly jointBackend: GigaAmRnntExecutionBackend;
}

/**
 * Resolve per-component execution providers without changing the historical
 * all-on-one-backend default. Explicit component overrides enable measured
 * hybrid compositions such as WebGPU encoder + WASM decoder/joiner.
 */
export function resolveGigaAmRnntBackends(
  requested: GigaAmRnntBackendSelection | undefined,
  fallbackBackend: string,
): ResolvedGigaAmRnntBackends {
  const fallback: GigaAmRnntExecutionBackend = fallbackBackend.startsWith('webgpu') ? 'webgpu' : 'wasm';
  const encoderBackend = requested?.encoderBackend ?? fallback;
  const decoderBackend = requested?.decoderBackend ?? fallback;
  const jointBackend = requested?.jointBackend ?? fallback;
  return {
    ortBackend: encoderBackend === 'webgpu' || decoderBackend === 'webgpu' || jointBackend === 'webgpu' ? 'webgpu' : 'wasm',
    encoderBackend,
    decoderBackend,
    jointBackend,
  };
}

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
  readonly predictionRnnLayers?: number;
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
  | (Omit<Extract<LasrCtcArtifactSource, { kind: 'direct' }>, 'artifacts'> & GigaAmRnntBackendSelection & { readonly kind: 'direct'; readonly artifacts: GigaAmRnntDirectArtifacts })
  | (Omit<Extract<LasrCtcArtifactSource, { kind: 'huggingface' }>, 'modelFilename' | 'tokenizerFilename'> & {
      readonly kind: 'huggingface';
      readonly encoderBackend?: GigaAmRnntExecutionBackend;
      readonly decoderBackend?: GigaAmRnntExecutionBackend;
      readonly jointBackend?: GigaAmRnntExecutionBackend;
      readonly parallelSessionInitialization?: boolean;
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
    readonly signal?: import('../../types/index.js').AbortSignalLike | null;
  };
}

export type GigaAmRnntNativeTranscript = LasrCtcNativeTranscript;
export type GigaAmRnntTranscriptionOptions = LasrCtcTranscriptionOptions;
