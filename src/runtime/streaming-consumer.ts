import type { StreamingDetectorConfig } from './streaming-config.js';
import type {
  StreamingDetectorSegment,
  StreamingSpeechDetectorSnapshot,
} from './streaming-detector.js';
import {
  type StreamingControlDefinition,
  listStreamingControls,
} from './streaming-controls.js';

export interface StreamingControlGroupDefinition {
  readonly id: string;
  readonly title: string;
  readonly description: string;
  readonly fields: readonly (keyof StreamingDetectorConfig)[];
}

export interface ResolvedStreamingControlGroup extends StreamingControlGroupDefinition {
  readonly controls: readonly StreamingControlDefinition[];
}

export const STREAMING_CONTROL_GROUPS: readonly StreamingControlGroupDefinition[] = [
  {
    id: 'segmenter',
    title: 'Live Segmenter',
    description: 'Peak-energy speech trigger, smoothing, and silence release.',
    fields: [
      'energyThreshold',
      'analysisWindowMs',
      'energySmoothingDurationMs',
      'minSilenceDurationMs',
      'maxSegmentDurationMs',
    ],
  },
  {
    id: 'timing',
    title: 'Extraction Timing',
    description: 'Lookback, overlap, and tail padding applied when a segment is extracted.',
    fields: [
      'prerollMs',
      'overlapDurationMs',
      'speechHangoverMs',
    ],
  },
  {
    id: 'acceptance',
    title: 'Final Acceptance',
    description: 'Duration and normalized 16 kHz energy thresholds used after live segmentation.',
    fields: [
      'minSpeechDurationMs',
      'minEnergyPerSecond',
      'minEnergyIntegral',
    ],
  },
  {
    id: 'adaptation',
    title: 'Noise Adaptation',
    description: 'How the baseline noise floor and onset backtracking adapt over time.',
    fields: [
      'initialNoiseFloor',
      'fastAdaptationRate',
      'slowAdaptationRate',
      'minBackgroundDurationSec',
      'snrThreshold',
      'minSnrThreshold',
      'energyRiseThreshold',
    ],
  },
  {
    id: 'acceptance-advanced',
    title: 'Adaptive Acceptance',
    description: 'Noise-scaled energy thresholds for the final segment acceptance gate.',
    fields: [
      'adaptiveEnergyPerSecondFactor',
      'adaptiveEnergyIntegralFactor',
      'minAdaptiveEnergyPerSecond',
      'minAdaptiveEnergyIntegral',
    ],
  },
] as const;

export function listStreamingControlGroups(): readonly StreamingControlGroupDefinition[] {
  return STREAMING_CONTROL_GROUPS;
}

export function resolveStreamingControlGroups(
  controls: readonly StreamingControlDefinition[] = listStreamingControls(),
): readonly ResolvedStreamingControlGroup[] {
  const controlByField = new Map(controls.map((definition) => [definition.field, definition]));
  return STREAMING_CONTROL_GROUPS.map((group) => ({
    ...group,
    controls: group.fields
      .map((field) => controlByField.get(field))
      .filter((definition): definition is StreamingControlDefinition => Boolean(definition)),
  }));
}

export function estimateStreamingReleaseMs(
  config:
    | Pick<
        Partial<StreamingDetectorConfig>,
        'minSilenceDurationMs' | 'speechHangoverMs'
      >
    | null
    | undefined,
): number | null {
  if (
    typeof config?.minSilenceDurationMs !== 'number'
    || !Number.isFinite(config.minSilenceDurationMs)
    || typeof config?.speechHangoverMs !== 'number'
    || !Number.isFinite(config.speechHangoverMs)
  ) {
    return null;
  }

  return Math.max(config.minSilenceDurationMs, config.speechHangoverMs);
}

export function resolveStreamingSnapshotNoiseFloorDbfs(
  snapshot:
    | Pick<StreamingSpeechDetectorSnapshot, 'foreground' | 'rough'>
    | null
    | undefined,
): number | null {
  const foregroundNoiseFloor = snapshot?.foreground?.noiseFloorDbfs;
  if (typeof foregroundNoiseFloor === 'number' && Number.isFinite(foregroundNoiseFloor)) {
    return foregroundNoiseFloor;
  }

  const roughNoiseFloor = snapshot?.rough?.noiseFloorDbfs;
  if (typeof roughNoiseFloor === 'number' && Number.isFinite(roughNoiseFloor)) {
    return roughNoiseFloor;
  }

  return null;
}

export function resolveStreamingForegroundThresholdDbfs(
  config:
    | Pick<
        Partial<StreamingDetectorConfig>,
        'minSpeechLevelDbfs'
      >
    | null
    | undefined,
  _noiseFloorDbfs: number | null | undefined,
): number | null {
  if (
    typeof config?.minSpeechLevelDbfs !== 'number'
    || !Number.isFinite(config.minSpeechLevelDbfs)
  ) {
    return null;
  }

  return config.minSpeechLevelDbfs;
}

export function resolveStreamingOnsetThresholdDbfs(
  config:
    | Pick<
        Partial<StreamingDetectorConfig>,
        'minSpeechLevelDbfs'
      >
    | null
    | undefined,
  _noiseFloorDbfs: number | null | undefined,
): number | null {
  if (
    typeof config?.minSpeechLevelDbfs !== 'number'
    || !Number.isFinite(config.minSpeechLevelDbfs)
  ) {
    return null;
  }

  return config.minSpeechLevelDbfs;
}

export function getStreamingSegmentDurationSeconds(
  segment: Pick<StreamingDetectorSegment, 'startFrame' | 'endFrame'> | null | undefined,
  sampleRate: number | null | undefined,
): number | null {
  if (!segment || typeof sampleRate !== 'number' || !Number.isFinite(sampleRate) || sampleRate <= 0) {
    return null;
  }
  return (segment.endFrame - segment.startFrame) / sampleRate;
}
