export const STREAMING_DEVICE_SAMPLE_RATE_FALLBACK = 48_000;
export const STREAMING_PROCESSING_SAMPLE_RATE = 16_000;
export const STREAMING_TIMELINE_CHUNK_MS = 16;
export const STREAMING_TIMELINE_CHUNK_FRAMES = Math.round(
  (STREAMING_TIMELINE_CHUNK_MS / 1000) * STREAMING_PROCESSING_SAMPLE_RATE,
);
export const STREAMING_ROUGH_GATE_ANALYSIS_CHUNKS = 5;
export const STREAMING_ROUGH_GATE_ANALYSIS_WINDOW_MS =
  STREAMING_TIMELINE_CHUNK_MS * STREAMING_ROUGH_GATE_ANALYSIS_CHUNKS;

export type TimelineAlignmentMode = 'floor' | 'round' | 'ceil';

export function resolveStreamingTimelineChunkFrames(sampleRate: number): number {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    return STREAMING_TIMELINE_CHUNK_FRAMES;
  }
  return Math.max(1, Math.round((STREAMING_TIMELINE_CHUNK_MS / 1000) * sampleRate));
}

export function alignFrameCountToTimeline(
  frameCount: number,
  sampleRate: number,
  mode: TimelineAlignmentMode = 'round',
): number {
  if (!Number.isFinite(frameCount) || frameCount <= 0) {
    return resolveStreamingTimelineChunkFrames(sampleRate);
  }

  const chunkFrames = resolveStreamingTimelineChunkFrames(sampleRate);
  const chunkCount = frameCount / chunkFrames;
  const aligner =
    mode === 'floor' ? Math.floor : mode === 'ceil' ? Math.ceil : Math.round;
  return Math.max(chunkFrames, aligner(chunkCount) * chunkFrames);
}

export function durationMsToAlignedFrameCount(
  durationMs: number,
  sampleRate: number,
  mode: TimelineAlignmentMode = 'round',
): number {
  if (!Number.isFinite(durationMs) || durationMs <= 0) {
    return resolveStreamingTimelineChunkFrames(sampleRate);
  }
  return alignFrameCountToTimeline((durationMs / 1000) * sampleRate, sampleRate, mode);
}

export function framesToMilliseconds(frameCount: number, sampleRate: number): number {
  if (!Number.isFinite(frameCount) || !Number.isFinite(sampleRate) || sampleRate <= 0) {
    return 0;
  }
  return (frameCount / sampleRate) * 1000;
}
