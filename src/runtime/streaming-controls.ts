import { STREAMING_TIMELINE_CHUNK_MS } from './audio-timeline.js';
import type { StreamingDetectorConfig } from './streaming-config.js';

export interface StreamingControlDefinition {
  readonly field: keyof StreamingDetectorConfig;
  readonly label: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly description: string;
  readonly guide: string;
  readonly chunkAligned?: boolean;
  readonly stepChunkMultiplier?: number;
  readonly maxFromConfigField?: keyof StreamingDetectorConfig;
}

function isMillisecondsField(field: keyof StreamingDetectorConfig): boolean {
  return String(field).endsWith('Ms');
}

function isSecondsField(field: keyof StreamingDetectorConfig): boolean {
  return String(field).endsWith('Sec');
}

function isRateField(field: keyof StreamingDetectorConfig): boolean {
  return (
    field === 'energyThreshold'
    || field === 'initialNoiseFloor'
    || field === 'fastAdaptationRate'
    || field === 'slowAdaptationRate'
    || field === 'energyRiseThreshold'
  );
}

export const STREAMING_CONTROL_DEFINITIONS: readonly StreamingControlDefinition[] = [
  {
    field: 'energyThreshold',
    label: 'Energy threshold',
    min: 0.01,
    max: 0.3,
    step: 0.005,
    description: 'Smoothed peak-amplitude threshold that opens live speech.',
    guide: 'Raise to reject weak triggers. Lower to catch softer near-mic speech.',
  },
  {
    field: 'analysisWindowMs',
    label: 'Analysis window',
    min: 80,
    max: 320,
    step: STREAMING_TIMELINE_CHUNK_MS,
    description: 'Window size used for the peak-energy analysis chunks.',
    guide: 'Keep near 80 ms unless you have a clear reason to trade latency for stability.',
    chunkAligned: true,
  },
  {
    field: 'energySmoothingDurationMs',
    label: 'Energy smoothing',
    min: 160,
    max: 1280,
    step: STREAMING_TIMELINE_CHUNK_MS,
    description: 'Duration of the moving-average smoothing applied to peak energy.',
    guide: 'Raise for steadier decisions. Lower for faster reaction.',
    chunkAligned: true,
  },
  {
    field: 'prerollMs',
    label: 'Lookback',
    min: 80,
    max: 800,
    step: STREAMING_TIMELINE_CHUNK_MS,
    description: 'Audio prepended before the detected onset when a segment is extracted.',
    guide: 'Raise if initial consonants are clipped. Too high adds leading silence.',
    chunkAligned: true,
  },
  {
    field: 'overlapDurationMs',
    label: 'Overlap',
    min: 16,
    max: 320,
    step: STREAMING_TIMELINE_CHUNK_MS,
    description: 'Intentional overlap allowed between consecutive accepted segments.',
    guide: 'Raise to preserve continuity across splits. Lower to reduce duplicated audio.',
    chunkAligned: true,
  },
  {
    field: 'speechHangoverMs',
    label: 'Tail padding',
    min: 16,
    max: 640,
    step: STREAMING_TIMELINE_CHUNK_MS,
    description: 'Extra audio appended after the logical speech end during extraction.',
    guide: 'Raise if endings feel clipped. Lower to reduce trailing silence.',
    chunkAligned: true,
  },
  {
    field: 'minSpeechDurationMs',
    label: 'Min final duration',
    min: 80,
    max: 960,
    step: STREAMING_TIMELINE_CHUNK_MS,
    description: 'Final acceptance rejects segments shorter than this duration.',
    guide: 'Raise to reject short bursts. Lower to keep clipped short words.',
    chunkAligned: true,
  },
  {
    field: 'minSilenceDurationMs',
    label: 'Silence release',
    min: 0,
    max: 1600,
    step: STREAMING_TIMELINE_CHUNK_MS,
    description: 'Configured silence hold before the live detector ends speech.',
    guide: 'Lower for faster turn-taking. Raise to keep internal pauses inside one segment.',
    chunkAligned: true,
  },
  {
    field: 'maxSegmentDurationMs',
    label: 'Max segment',
    min: 400,
    max: 12000,
    step: 80,
    description: 'Hard cap for one segment before the detector forces a split.',
    guide: 'Raise for long dictation. Lower for faster transcript turnover.',
    maxFromConfigField: 'ringBufferDurationMs',
  },
  {
    field: 'initialNoiseFloor',
    label: 'Initial noise',
    min: 0.00001,
    max: 0.05,
    step: 0.0005,
    description: 'Starting baseline for the adaptive noise-floor tracker.',
    guide: 'Lower for quiet-room defaults. Raise if your input chain has a persistent floor.',
  },
  {
    field: 'fastAdaptationRate',
    label: 'Fast adapt',
    min: 0,
    max: 1,
    step: 0.01,
    description: 'Noise-floor adaptation used early in a silence run.',
    guide: 'Raise to react faster to room changes. Lower for a steadier baseline.',
  },
  {
    field: 'slowAdaptationRate',
    label: 'Slow adapt',
    min: 0,
    max: 1,
    step: 0.01,
    description: 'Noise-floor adaptation used after sustained silence.',
    guide: 'Lower for a more stable baseline. Raise if the environment changes often.',
  },
  {
    field: 'minBackgroundDurationSec',
    label: 'Blend duration',
    min: 0,
    max: 5,
    step: 0.1,
    description: 'Silence duration needed before the tracker fully shifts from fast to slow adaptation.',
    guide: 'Raise to keep the tracker in fast mode longer. Lower for quicker stabilization.',
  },
  {
    field: 'snrThreshold',
    label: 'SNR display gate',
    min: 0,
    max: 20,
    step: 0.5,
    description: 'Heuristic SNR level used for diagnostics and optional rough gating.',
    guide: 'Usually leave this alone unless you explicitly enable SNR gating.',
  },
  {
    field: 'minSnrThreshold',
    label: 'Onset SNR floor',
    min: 0,
    max: 10,
    step: 0.25,
    description: 'Minimum SNR cue used when backtracking the real speech start.',
    guide: 'Raise to stop onset search sooner in noisy rooms. Lower to search farther back.',
  },
  {
    field: 'energyRiseThreshold',
    label: 'Rise threshold',
    min: 0,
    max: 0.5,
    step: 0.01,
    description: 'Required relative energy rise when searching backward for onset.',
    guide: 'Raise to be stricter about what counts as a meaningful onset ramp.',
  },
  {
    field: 'minEnergyPerSecond',
    label: 'Min power',
    min: 0,
    max: 40,
    step: 0.5,
    description: 'Minimum 16 kHz-normalized average power required for final acceptance.',
    guide: 'Raise to reject quiet background speech. Lower to keep softer valid speech.',
  },
  {
    field: 'minEnergyIntegral',
    label: 'Min integral',
    min: 0,
    max: 200,
    step: 1,
    description: 'Minimum 16 kHz-normalized total energy required for final acceptance.',
    guide: 'Raise to reject weak long segments. Lower if quiet long dictation is being dropped.',
  },
  {
    field: 'adaptiveEnergyPerSecondFactor',
    label: 'Adaptive power factor',
    min: 0,
    max: 100,
    step: 1,
    description: 'Noise-scaled factor used to derive the adaptive per-second threshold.',
    guide: 'Raise if noisy rooms still accept weak speech. Lower if valid speech is over-rejected.',
  },
  {
    field: 'adaptiveEnergyIntegralFactor',
    label: 'Adaptive integral factor',
    min: 0,
    max: 100,
    step: 1,
    description: 'Noise-scaled factor used to derive the adaptive energy-integral threshold.',
    guide: 'Raise for stricter long-segment rejection in noise.',
  },
  {
    field: 'minAdaptiveEnergyPerSecond',
    label: 'Adaptive power floor',
    min: 0,
    max: 20,
    step: 0.5,
    description: 'Minimum floor for the adaptive per-second threshold.',
    guide: 'Raise to keep a stronger base rejection floor even in quiet rooms.',
  },
  {
    field: 'minAdaptiveEnergyIntegral',
    label: 'Adaptive integral floor',
    min: 0,
    max: 40,
    step: 0.5,
    description: 'Minimum floor for the adaptive energy-integral threshold.',
    guide: 'Raise to keep a stronger long-segment rejection floor even in quiet rooms.',
  },
] as const;

export interface StreamingControlConstraints {
  readonly min: number;
  readonly max: number;
  readonly step: number;
}

function alignChunkAlignedMinimum(min: number, step: number): number {
  if (!Number.isFinite(min) || !Number.isFinite(step) || step <= 0) {
    return min;
  }
  return Math.ceil(min / step) * step;
}

export function listStreamingControls(): readonly StreamingControlDefinition[] {
  return STREAMING_CONTROL_DEFINITIONS;
}

export function getStreamingControlDefinition(
  field: keyof StreamingDetectorConfig,
): StreamingControlDefinition | undefined {
  return STREAMING_CONTROL_DEFINITIONS.find((definition) => definition.field === field);
}

export function resolveStreamingControlConstraints(
  definition: StreamingControlDefinition,
  resolvedConfig?: Partial<StreamingDetectorConfig> | null,
): StreamingControlConstraints {
  const chunkDurationMs = Math.max(
    1,
    Math.round(resolvedConfig?.chunkDurationMs ?? STREAMING_TIMELINE_CHUNK_MS),
  );
  const step = definition.chunkAligned
    ? Math.max(1, chunkDurationMs * Math.max(1, definition.stepChunkMultiplier ?? 1))
    : definition.step;
  const maxFromConfig =
    definition.maxFromConfigField &&
    typeof resolvedConfig?.[definition.maxFromConfigField] === 'number' &&
    Number.isFinite(resolvedConfig[definition.maxFromConfigField])
      ? Number(resolvedConfig[definition.maxFromConfigField])
      : null;
  const min = definition.chunkAligned
    ? alignChunkAlignedMinimum(definition.min, step)
    : definition.min;
  const max = Math.max(min, maxFromConfig ?? definition.max);
  return {
    min,
    max,
    step,
  };
}

export function resolveStreamingControlStep(
  definition: StreamingControlDefinition,
  resolvedConfig?: Partial<StreamingDetectorConfig> | null,
): number {
  return resolveStreamingControlConstraints(definition, resolvedConfig).step;
}

export function clampStreamingControlValue(
  definition: StreamingControlDefinition,
  value: number,
  resolvedConfig?: Partial<StreamingDetectorConfig> | null,
): number {
  if (!Number.isFinite(value)) {
    return resolveStreamingControlConstraints(definition, resolvedConfig).min;
  }
  const { min, max } = resolveStreamingControlConstraints(definition, resolvedConfig);
  return Math.min(max, Math.max(min, value));
}

export function normalizeStreamingControlValue(
  definition: StreamingControlDefinition,
  value: number,
  resolvedConfig?: Partial<StreamingDetectorConfig> | null,
): number {
  const { min, step } = resolveStreamingControlConstraints(definition, resolvedConfig);
  const clamped = clampStreamingControlValue(definition, value, resolvedConfig);
  if (!Number.isFinite(step) || step <= 0) {
    return clamped;
  }
  const snapped = min + Math.round((clamped - min) / step) * step;
  return clampStreamingControlValue(definition, snapped, resolvedConfig);
}

export function formatStreamingControlValue(
  definition: StreamingControlDefinition,
  value: number | null | undefined,
): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '--';
  }
  if (isMillisecondsField(definition.field)) {
    return `${value.toFixed(0)} ms`;
  }
  if (isSecondsField(definition.field)) {
    return `${value.toFixed(1)} s`;
  }
  if (isRateField(definition.field)) {
    return definition.field === 'initialNoiseFloor'
      ? value.toFixed(5)
      : value.toFixed(2);
  }
  return value.toFixed(2);
}

export function formatStreamingControlHint(
  definition: StreamingControlDefinition,
  resolvedConfig?: Partial<StreamingDetectorConfig> | null,
): string {
  const { min, max, step } = resolveStreamingControlConstraints(definition, resolvedConfig);
  const chunkDurationMs = Math.round(
    resolvedConfig?.chunkDurationMs ?? STREAMING_TIMELINE_CHUNK_MS,
  );
  const chunkNote = definition.chunkAligned
    ? `chunk-aligned ${chunkDurationMs} ms`
    : null;

  const unit =
    isMillisecondsField(definition.field)
      ? 'ms'
      : isSecondsField(definition.field)
        ? 's'
        : null;
  const range = unit ? `${min}..${max} ${unit}` : `${min}..${max}`;
  const stepLabel = unit ? `step ${step} ${unit}` : `step ${step}`;

  return chunkNote ? `${range} · ${stepLabel} · ${chunkNote}` : `${range} · ${stepLabel}`;
}
