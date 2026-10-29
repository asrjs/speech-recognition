import { durationMsToAlignedFrameCount } from './audio-timeline.js';
import type { StreamingDetectorConfig } from './streaming-config.js';

export interface ActiveSegmentState {
  readonly startFrame: number;
  readonly lastFrame: number;
  readonly source: 'ten-vad' | 'rough-fallback';
}

export interface FinalizedSegmentBounds {
  readonly logicalStartFrame: number;
  readonly logicalEndFrame: number;
  readonly extractedStartFrame: number;
  readonly extractedEndFrame: number;
}

export interface FinalizedSegmentBoundsContext {
  readonly baseFrame: number;
  readonly currentFrame: number;
  readonly config: StreamingDetectorConfig;
  readonly sampleRate: number;
  readonly continuationAfterMaxDuration: boolean;
  readonly lastAcceptedLogicalEndFrame: number | null;
}

export function computeFinalizedSegmentBounds(
  context: FinalizedSegmentBoundsContext,
  logicalStartFrame: number,
  logicalEndFrame: number,
): FinalizedSegmentBounds {
  const { baseFrame, currentFrame, config, sampleRate } = context;
  const prerollFrames = durationMsToAlignedFrameCount(
    config.prerollMs,
    sampleRate,
    'ceil',
    config.chunkDurationMs,
  );
  const overlapFrames = durationMsToAlignedFrameCount(
    config.overlapDurationMs,
    sampleRate,
    'ceil',
    config.chunkDurationMs,
  );
  const continuationOverlapMs = Math.min(160, Math.max(config.overlapDurationMs, 80));
  const effectiveOverlapFrames = context.continuationAfterMaxDuration
    ? durationMsToAlignedFrameCount(
        continuationOverlapMs,
        sampleRate,
        'ceil',
        config.chunkDurationMs,
      )
    : overlapFrames;
  const hangoverFrames = durationMsToAlignedFrameCount(
    config.speechHangoverMs,
    sampleRate,
    'ceil',
    config.chunkDurationMs,
  );

  let extractedStartFrame = Math.max(baseFrame, logicalStartFrame - prerollFrames);
  if (context.lastAcceptedLogicalEndFrame !== null) {
    const overlappedStartFrame = context.lastAcceptedLogicalEndFrame - effectiveOverlapFrames;
    extractedStartFrame = Math.max(baseFrame, Math.min(extractedStartFrame, overlappedStartFrame));
  }

  const clampedLogicalStartFrame = Math.max(baseFrame, logicalStartFrame);
  const clampedLogicalEndFrame = Math.max(
    clampedLogicalStartFrame + 1,
    Math.min(currentFrame, logicalEndFrame),
  );
  const extractedEndFrame = Math.max(
    extractedStartFrame + 1,
    Math.min(currentFrame, clampedLogicalEndFrame + hangoverFrames),
  );

  return {
    logicalStartFrame: clampedLogicalStartFrame,
    logicalEndFrame: clampedLogicalEndFrame,
    extractedStartFrame,
    extractedEndFrame,
  };
}

export function isQuietChunk(chunk: Float32Array, threshold: number): boolean {
  let squareSum = 0;
  for (const sample of chunk) squareSum += sample * sample;
  const rms = Math.sqrt(squareSum / chunk.length);
  return rms <= threshold;
}
