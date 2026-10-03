import type { QuantizationMode } from '../../runtime/huggingface.js';
import type { ResolvedAssetHandle } from '../../types/index.js';

export interface ParakeetCtcModelConfig {
  readonly ecosystem: string;
  readonly architecture: string;
  readonly processorArchitecture: string;
  readonly encoderArchitecture: string;
  readonly decoderArchitecture: string;
  readonly sampleRate: number;
  readonly featureHopSeconds: number;
  readonly rawStride: number;
  readonly nMels: number;
  readonly nFft: number;
  readonly winLength: number;
  readonly hopLength: number;
  readonly preemphasis: number;
  readonly vocabularySize: number;
  readonly languages: readonly string[];
  readonly tokenizer: { readonly kind: 'sentencepiece'; readonly blankTokenId: number };
  readonly featureLayout: 'mel-major';
}

export interface ParakeetCtcDirectArtifacts {
  readonly encoderUrl: string;
  readonly tokenizerUrl: string;
  readonly encoderDataUrl?: string;
  readonly encoderFilename?: string;
}

export interface ParakeetCtcDirectArtifactSource {
  readonly kind: 'direct';
  readonly artifacts: ParakeetCtcDirectArtifacts;
  readonly encoderBackend?: 'wasm' | 'webgpu';
  readonly preprocessorBackend?: 'js' | 'onnx';
}

export type ParakeetCtcArtifactSource = ParakeetCtcDirectArtifactSource;

export interface ParakeetCtcModelOptions {
  readonly config?: Partial<ParakeetCtcModelConfig>;
  readonly source?: ParakeetCtcArtifactSource;
  readonly encoderBackend?: 'wasm' | 'webgpu';
  readonly encoderQuant?: QuantizationMode;
  readonly preprocessorBackend?: 'js' | 'onnx';
  readonly cpuThreads?: number;
  readonly wasmPaths?: string;
  readonly signal?: AbortSignal | null;
}

export interface ParakeetCtcResolvedOptions {
  readonly config: ParakeetCtcModelConfig;
  readonly source?: ParakeetCtcArtifactSource;
  readonly encoderBackend?: 'wasm' | 'webgpu';
  readonly preprocessorBackend: 'js' | 'onnx';
  readonly cpuThreads?: number;
  readonly wasmPaths?: string;
  readonly signal?: AbortSignal | null;
}

export interface ParakeetCtcSession {
  readonly config: ParakeetCtcModelConfig;
  readonly tokenizerUrl: string;
  readonly encoderUrl: string;
  readonly encoderDataUrl?: string;
  readonly encoderFilename?: string;
  readonly ort: unknown;
  readonly warnings: readonly string[];
}

export type { ResolvedAssetHandle };
