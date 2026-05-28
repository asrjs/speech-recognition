const DEFAULT_MIN_SPEECH_SECONDS = 0.25;
const DEFAULT_MIN_SILENCE_SECONDS = 0.1;
const DEFAULT_PAD_SECONDS = 0.03;
const DEFAULT_MAX_WINDOW_SECONDS = 30;
const TIME_PRECISION = 1_000_000;

export interface SpeechSegment {
  readonly index: number;
  readonly startTime: number;
  readonly endTime: number;
  readonly confidence?: number;
}

export interface SpeechWindow extends SpeechSegment {
  readonly sourceSegmentIndices?: readonly number[];
}

export interface PadSpeechSegmentsOptions {
  readonly padSeconds?: number;
  readonly minTime?: number;
  readonly maxTime?: number;
}

export interface MergeNearbySpeechSegmentsOptions {
  readonly minSilenceSeconds?: number;
}

export interface SplitLongSpeechSegmentsOptions {
  readonly maxDurationSeconds?: number;
  readonly minDurationSeconds?: number;
}

export interface SpeechSegmentsToWindowsOptions {
  readonly audioDurationSeconds?: number;
  readonly minSpeechSeconds?: number;
  readonly minSilenceSeconds?: number;
  readonly padSeconds?: number;
  readonly maxWindowSeconds?: number;
}

interface InternalSpeechSegment extends SpeechSegment {
  readonly sourceSegmentIndices?: readonly number[];
}

export function padSpeechSegments(
  segments: readonly SpeechSegment[],
  options: PadSpeechSegmentsOptions = {},
): SpeechSegment[] {
  const padSeconds = Math.max(0, options.padSeconds ?? DEFAULT_PAD_SECONDS);
  const minTime = options.minTime ?? 0;
  const maxTime = options.maxTime ?? Number.POSITIVE_INFINITY;

  return segments
    .map((segment) => ({
      ...segment,
      startTime: clampTime(segment.startTime - padSeconds, minTime, maxTime),
      endTime: clampTime(segment.endTime + padSeconds, minTime, maxTime),
    }))
    .filter((segment) => segment.endTime > segment.startTime)
    .map(reindexSegment);
}

export function mergeNearbySpeechSegments(
  segments: readonly SpeechSegment[],
  options: MergeNearbySpeechSegmentsOptions = {},
): SpeechSegment[] {
  return mergeInternalSegments(segments, options).map(reindexSegment);
}

export function splitLongSpeechSegments(
  segments: readonly SpeechSegment[],
  options: SplitLongSpeechSegmentsOptions = {},
): SpeechSegment[] {
  return splitInternalSegments(segments, options).map(reindexSegment);
}

export function speechSegmentsToWindows(
  segments: readonly SpeechSegment[],
  options: SpeechSegmentsToWindowsOptions = {},
): SpeechWindow[] {
  const minSpeechSeconds = Math.max(0, options.minSpeechSeconds ?? DEFAULT_MIN_SPEECH_SECONDS);
  const seeded = segments
    .filter((segment) => segment.endTime > segment.startTime)
    .filter((segment) => segment.endTime - segment.startTime >= minSpeechSeconds)
    .map((segment): InternalSpeechSegment => ({
      ...segment,
      sourceSegmentIndices: [segment.index],
    }));

  const padded = padInternalSegments(seeded, {
    padSeconds: options.padSeconds ?? DEFAULT_PAD_SECONDS,
    minTime: 0,
    maxTime: options.audioDurationSeconds ?? Number.POSITIVE_INFINITY,
  });
  const merged = mergeInternalSegments(padded, {
    minSilenceSeconds: options.minSilenceSeconds ?? DEFAULT_MIN_SILENCE_SECONDS,
  });
  const split = splitInternalSegments(merged, {
    maxDurationSeconds: options.maxWindowSeconds ?? DEFAULT_MAX_WINDOW_SECONDS,
    minDurationSeconds: minSpeechSeconds,
  });

  return split.map((segment, index) => ({
    index,
    startTime: segment.startTime,
    endTime: segment.endTime,
    ...(segment.confidence === undefined ? {} : { confidence: segment.confidence }),
    ...(segment.sourceSegmentIndices ? { sourceSegmentIndices: segment.sourceSegmentIndices } : {}),
  }));
}

function padInternalSegments(
  segments: readonly InternalSpeechSegment[],
  options: PadSpeechSegmentsOptions = {},
): InternalSpeechSegment[] {
  const padSeconds = Math.max(0, options.padSeconds ?? DEFAULT_PAD_SECONDS);
  const minTime = options.minTime ?? 0;
  const maxTime = options.maxTime ?? Number.POSITIVE_INFINITY;
  return segments
    .map((segment) => ({
      ...segment,
      startTime: clampTime(segment.startTime - padSeconds, minTime, maxTime),
      endTime: clampTime(segment.endTime + padSeconds, minTime, maxTime),
    }))
    .filter((segment) => segment.endTime > segment.startTime);
}

function mergeInternalSegments(
  segments: readonly InternalSpeechSegment[],
  options: MergeNearbySpeechSegmentsOptions = {},
): InternalSpeechSegment[] {
  const minSilenceSeconds = Math.max(0, options.minSilenceSeconds ?? DEFAULT_MIN_SILENCE_SECONDS);
  const sorted = [...segments]
    .filter((segment) => segment.endTime > segment.startTime)
    .sort((left, right) => left.startTime - right.startTime || left.endTime - right.endTime);
  const merged: InternalSpeechSegment[] = [];

  for (const segment of sorted) {
    const previous = merged.at(-1);
    if (!previous || segment.startTime - previous.endTime > minSilenceSeconds) {
      merged.push({ ...segment });
      continue;
    }

    merged[merged.length - 1] = {
      ...previous,
      endTime: Math.max(previous.endTime, segment.endTime),
      confidence: mergeConfidence(previous.confidence, segment.confidence),
      sourceSegmentIndices: mergeSourceIndices(previous, segment),
    };
  }

  return merged;
}

function splitInternalSegments(
  segments: readonly InternalSpeechSegment[],
  options: SplitLongSpeechSegmentsOptions = {},
): InternalSpeechSegment[] {
  const maxDurationSeconds = Math.max(0, options.maxDurationSeconds ?? DEFAULT_MAX_WINDOW_SECONDS);
  const minDurationSeconds = Math.max(0, options.minDurationSeconds ?? DEFAULT_MIN_SPEECH_SECONDS);
  const split: InternalSpeechSegment[] = [];

  for (const segment of segments) {
    if (maxDurationSeconds <= 0 || segment.endTime - segment.startTime <= maxDurationSeconds) {
      if (segment.endTime - segment.startTime >= minDurationSeconds) {
        split.push({ ...segment });
      }
      continue;
    }

    let startTime = segment.startTime;
    while (startTime < segment.endTime) {
      const endTime = Math.min(segment.endTime, startTime + maxDurationSeconds);
      if (endTime - startTime >= minDurationSeconds) {
        split.push({ ...segment, startTime: roundTime(startTime), endTime: roundTime(endTime) });
      }
      startTime = endTime;
    }
  }

  return split;
}

function reindexSegment(segment: SpeechSegment, index: number): SpeechSegment {
  return {
    index,
    startTime: segment.startTime,
    endTime: segment.endTime,
    ...(segment.confidence === undefined ? {} : { confidence: segment.confidence }),
  };
}

function clampTime(value: number, minTime: number, maxTime: number): number {
  return roundTime(Math.min(maxTime, Math.max(minTime, value)));
}

function roundTime(value: number): number {
  return Math.round(value * TIME_PRECISION) / TIME_PRECISION;
}

function mergeConfidence(left: number | undefined, right: number | undefined): number | undefined {
  if (left === undefined) {
    return right;
  }
  if (right === undefined) {
    return left;
  }
  return Math.max(left, right);
}

function mergeSourceIndices(
  left: InternalSpeechSegment,
  right: InternalSpeechSegment,
): readonly number[] | undefined {
  const sourceIndices = [...(left.sourceSegmentIndices ?? [left.index]), ...(right.sourceSegmentIndices ?? [right.index])];
  return [...new Set(sourceIndices)].sort((a, b) => a - b);
}
