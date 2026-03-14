import type { AudioProcessorKind, ProcessorLogScale, ProcessorNormalization } from './base.js';

interface BaseProcessorSpec<TKind extends AudioProcessorKind> {
  readonly kind: TKind;
  readonly sampleRate: number;
}

export interface MelSpectrogramProcessorSpec
  extends BaseProcessorSpec<'nemo-mel' | 'kaldi-mel' | 'gigaam-mel'> {
  readonly featureSize: number;
  readonly fftSize: number;
  readonly windowSizeSamples: number;
  readonly hopSizeSamples: number;
  readonly logScale: ProcessorLogScale;
  readonly normalize: ProcessorNormalization;
  readonly melScale: 'slaney' | 'kaldi';
}

export interface WhisperMelProcessorSpec extends BaseProcessorSpec<'whisper-mel'> {
  readonly featureSize: number;
  readonly fftSize: number;
  readonly hopSizeSamples: number;
  readonly logScale: 'log10';
  readonly clampRangeDb: number;
  readonly scaleOffset: number;
  readonly scaleDivisor: number;
}

export interface Wav2Vec2ConvLayerSpec {
  readonly channels: number;
  readonly kernelSize: number;
  readonly stride: number;
}

export interface Wav2Vec2ConvProcessorSpec extends BaseProcessorSpec<'wav2vec2-conv'> {
  readonly featureSize: number;
  readonly outputStride: number;
  readonly convLayers: readonly Wav2Vec2ConvLayerSpec[];
}

export interface IdentityProcessorSpec extends BaseProcessorSpec<'identity'> {
  readonly chunkSizeSamples?: number;
}

export type ProcessorSpec =
  | MelSpectrogramProcessorSpec
  | WhisperMelProcessorSpec
  | Wav2Vec2ConvProcessorSpec
  | IdentityProcessorSpec;
