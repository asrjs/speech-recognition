import type { AudioBufferLike } from '../types/index.js';

/** Shared processor family identifiers used by architecture descriptors and model metadata. */
export type AudioProcessorKind =
  | 'nemo-mel'
  | 'kaldi-mel'
  | 'gigaam-mel'
  | 'whisper-mel'
  | 'wav2vec2-conv'
  | 'conv-subsampler'
  | 'projector'
  | 'identity';

export type ProcessorFeatureLayout = 'channels-first' | 'channels-last';
export type ProcessorLogScale = 'natural' | 'log10' | 'none';
export type ProcessorNormalization = 'none' | 'per_feature' | 'per_utterance';

/** Summary of a processor output without prescribing the concrete tensor container. */
export interface ProcessorFeatureDescriptor {
  readonly processor: AudioProcessorKind;
  readonly sampleRate: number;
  readonly channelCount: number;
  readonly durationSeconds: number;
  readonly frameCount: number;
  readonly featureSize: number;
  readonly layout: ProcessorFeatureLayout;
}

/**
 * Shared contract for audio and feature preprocessors.
 *
 * Concrete math still lives in model families or dedicated audio helpers; this
 * interface only defines the stable boundary from input audio to features.
 */
export interface AudioProcessor<
  TConfig = unknown,
  TOutput extends ProcessorFeatureDescriptor = ProcessorFeatureDescriptor,
> {
  readonly kind: AudioProcessorKind;
  readonly sharedModule: 'audio';
  compute(audio: AudioBufferLike, config: TConfig): Promise<TOutput> | TOutput;
}
