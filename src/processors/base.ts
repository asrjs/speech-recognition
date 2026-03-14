import type { AudioBufferLike } from '../types/index.js';

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

export interface ProcessorFeatureDescriptor {
  readonly processor: AudioProcessorKind;
  readonly sampleRate: number;
  readonly channelCount: number;
  readonly durationSeconds: number;
  readonly frameCount: number;
  readonly featureSize: number;
  readonly layout: ProcessorFeatureLayout;
}

export interface AudioProcessor<
  TConfig = unknown,
  TOutput extends ProcessorFeatureDescriptor = ProcessorFeatureDescriptor
> {
  readonly kind: AudioProcessorKind;
  readonly sharedModule: 'processors';
  compute(audio: AudioBufferLike, config: TConfig): Promise<TOutput> | TOutput;
}
