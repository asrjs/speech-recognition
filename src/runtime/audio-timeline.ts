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

export function resolveStreamingTimelineChunkDurationMs(chunkDurationMs: number): number {
  if (!Number.isFinite(chunkDurationMs) || chunkDurationMs <= 0) {
    return STREAMING_TIMELINE_CHUNK_MS;
  }
  return Math.max(1, Math.round(chunkDurationMs));
}

export function resolveStreamingTimelineChunkFrames(
  sampleRate: number,
  chunkDurationMs = STREAMING_TIMELINE_CHUNK_MS,
): number {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    return STREAMING_TIMELINE_CHUNK_FRAMES;
  }
  const safeChunkDurationMs = resolveStreamingTimelineChunkDurationMs(chunkDurationMs);
  return Math.max(1, Math.round((safeChunkDurationMs / 1000) * sampleRate));
}

export function alignDurationMsToTimeline(
  durationMs: number,
  chunkDurationMs = STREAMING_TIMELINE_CHUNK_MS,
  mode: TimelineAlignmentMode = 'round',
): number {
  const safeChunkDurationMs = resolveStreamingTimelineChunkDurationMs(chunkDurationMs);
  if (!Number.isFinite(durationMs) || durationMs <= 0) {
    return safeChunkDurationMs;
  }
  const chunkCount = durationMs / safeChunkDurationMs;
  const aligner = mode === 'floor' ? Math.floor : mode === 'ceil' ? Math.ceil : Math.round;
  return Math.max(safeChunkDurationMs, aligner(chunkCount) * safeChunkDurationMs);
}

export function alignFrameCountToTimeline(
  frameCount: number,
  sampleRate: number,
  mode: TimelineAlignmentMode = 'round',
  chunkDurationMs = STREAMING_TIMELINE_CHUNK_MS,
): number {
  if (!Number.isFinite(frameCount) || frameCount <= 0) {
    return resolveStreamingTimelineChunkFrames(sampleRate, chunkDurationMs);
  }

  const chunkFrames = resolveStreamingTimelineChunkFrames(sampleRate, chunkDurationMs);
  const chunkCount = frameCount / chunkFrames;
  const aligner = mode === 'floor' ? Math.floor : mode === 'ceil' ? Math.ceil : Math.round;
  return Math.max(chunkFrames, aligner(chunkCount) * chunkFrames);
}

export function durationMsToAlignedFrameCount(
  durationMs: number,
  sampleRate: number,
  mode: TimelineAlignmentMode = 'round',
  chunkDurationMs = STREAMING_TIMELINE_CHUNK_MS,
): number {
  if (!Number.isFinite(durationMs) || durationMs <= 0) {
    return resolveStreamingTimelineChunkFrames(sampleRate, chunkDurationMs);
  }
  return alignFrameCountToTimeline(
    (durationMs / 1000) * sampleRate,
    sampleRate,
    mode,
    chunkDurationMs,
  );
}

export function framesToMilliseconds(frameCount: number, sampleRate: number): number {
  if (!Number.isFinite(frameCount) || !Number.isFinite(sampleRate) || sampleRate <= 0) {
    return 0;
  }
  return (frameCount / sampleRate) * 1000;
}
