import type { AudioBufferLike } from '../types/index.js';
import type { ProcessorFeatureDescriptor } from './base.js';

export interface FrameBasedProcessorConfig {
  readonly processor: ProcessorFeatureDescriptor['processor'];
  readonly featureSize: number;
  readonly frameShiftSeconds: number;
  readonly subsamplingFactor?: number;
  readonly layout?: ProcessorFeatureDescriptor['layout'];
}

export function estimateFrameBasedProcessorDescriptor(
  audio: AudioBufferLike,
  config: FrameBasedProcessorConfig
): ProcessorFeatureDescriptor {
  const frameSeconds = Math.max(
    config.frameShiftSeconds * Math.max(1, config.subsamplingFactor ?? 1),
    1 / Math.max(audio.sampleRate, 1)
  );

  return {
    processor: config.processor,
    sampleRate: audio.sampleRate,
    channelCount: audio.numberOfChannels,
    durationSeconds: audio.durationSeconds,
    frameCount: Math.max(1, Math.ceil(audio.durationSeconds / frameSeconds)),
    featureSize: config.featureSize,
    layout: config.layout ?? 'channels-first'
  };
}
