import { amplitudeToDbfs } from './noise-floor.js';

export interface SegmentForegroundFilterConfig {
  readonly foregroundFilterEnabled: boolean;
  readonly foregroundMinDb: number;
  readonly foregroundOnsetMinDb: number;
  readonly foregroundOnsetWindowMs: number;
  readonly foregroundShortSpeechMs: number;
  readonly foregroundLongSpeechMs: number;
  readonly foregroundLongMinDb: number;
}

export interface SegmentForegroundFilterResult {
  readonly accepted: boolean;
  readonly reason: string;
  readonly durationMs: number;
  readonly noiseFloorDbfs: number;
  readonly segmentP90Dbfs: number;
  readonly onsetP90Dbfs: number;
  readonly foregroundDb: number;
  readonly onsetDb: number;
  readonly shortSpeech: boolean;
  readonly longSpeech: boolean;
}

const DEFAULT_FRAME_WINDOW_MS = 32;
const MIN_DBFS = -100;

function clampDb(value: number): number {
  if (!Number.isFinite(value)) {
    return MIN_DBFS;
  }
  return Math.max(MIN_DBFS, value);
}

function resolvePercentile(values: readonly number[], percentile: number): number {
  if (values.length === 0) {
    return MIN_DBFS;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(percentile * sorted.length) - 1),
  );
  return sorted[index] ?? MIN_DBFS;
}

function collectWindowDbfs(
  pcm: Float32Array,
  sampleRate: number,
  windowFrames: number,
  limitFrames = pcm.length,
): number[] {
  if (!pcm.length || sampleRate <= 0) {
    return [];
  }

  const resolvedWindowFrames = Math.max(1, windowFrames);
  const resolvedLimitFrames = Math.max(0, Math.min(limitFrames, pcm.length));
  const values: number[] = [];

  for (let start = 0; start < resolvedLimitFrames; start += resolvedWindowFrames) {
    const end = Math.min(resolvedLimitFrames, start + resolvedWindowFrames);
    let sumSquares = 0;
    for (let index = start; index < end; index += 1) {
      const sample = pcm[index] ?? 0;
      sumSquares += sample * sample;
    }
    const rms = end > start ? Math.sqrt(sumSquares / (end - start)) : 0;
    values.push(amplitudeToDbfs(rms));
  }

  return values;
}

export function scoreSegmentForeground(
  pcm: Float32Array,
  sampleRate: number,
  noiseFloorDbfs: number,
  config: SegmentForegroundFilterConfig,
): SegmentForegroundFilterResult {
  const durationMs =
    sampleRate > 0 ? Math.max(0, (pcm.length / sampleRate) * 1000) : 0;
  const resolvedNoiseFloorDbfs = clampDb(noiseFloorDbfs);

  if (!config.foregroundFilterEnabled) {
    return {
      accepted: true,
      reason: 'foreground-filter-disabled',
      durationMs,
      noiseFloorDbfs: resolvedNoiseFloorDbfs,
      segmentP90Dbfs: MIN_DBFS,
      onsetP90Dbfs: MIN_DBFS,
      foregroundDb: 0,
      onsetDb: 0,
      shortSpeech: durationMs < config.foregroundShortSpeechMs,
      longSpeech: durationMs >= config.foregroundLongSpeechMs,
    };
  }

  const analysisWindowFrames = Math.max(
    1,
    Math.round((DEFAULT_FRAME_WINDOW_MS / 1000) * sampleRate),
  );
  const onsetFrames = Math.max(
    analysisWindowFrames,
    Math.round((config.foregroundOnsetWindowMs / 1000) * sampleRate),
  );
  const segmentP90Dbfs = clampDb(
    resolvePercentile(
      collectWindowDbfs(pcm, sampleRate, analysisWindowFrames),
      0.9,
    ),
  );
  const onsetP90Dbfs = clampDb(
    resolvePercentile(
      collectWindowDbfs(pcm, sampleRate, analysisWindowFrames, onsetFrames),
      0.9,
    ),
  );
  const foregroundDb = segmentP90Dbfs - resolvedNoiseFloorDbfs;
  const onsetDb = onsetP90Dbfs - resolvedNoiseFloorDbfs;
  const shortSpeech = durationMs < config.foregroundShortSpeechMs;
  const longSpeech = durationMs >= config.foregroundLongSpeechMs;

  let accepted = true;
  let reason = 'accepted';
  if (foregroundDb < config.foregroundMinDb) {
    accepted = false;
    reason = 'foreground-too-quiet';
  } else if (shortSpeech && onsetDb < config.foregroundOnsetMinDb) {
    accepted = false;
    reason = 'short-onset-too-quiet';
  } else if (longSpeech && foregroundDb < config.foregroundLongMinDb) {
    accepted = false;
    reason = 'long-quiet-background';
  }

  return {
    accepted,
    reason,
    durationMs,
    noiseFloorDbfs: resolvedNoiseFloorDbfs,
    segmentP90Dbfs,
    onsetP90Dbfs,
    foregroundDb,
    onsetDb,
    shortSpeech,
    longSpeech,
  };
}
