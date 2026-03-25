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
    title: 'TEN-VAD Segmenter',
    description: 'Speech proposal, confirmation, and release timing.',
    fields: [
      'prerollMs',
      'tenVadThreshold',
      'tenVadConfirmationWindowMs',
      'tenVadHangoverMs',
      'tenVadMinSpeechDurationMs',
      'tenVadMinSilenceDurationMs',
      'maxSegmentDurationMs',
    ],
  },
  {
    id: 'foreground',
    title: 'Foreground Filter',
    description: 'Rejects quiet background speech after TEN-VAD creates a segment.',
    fields: [
      'foregroundOnsetWindowMs',
      'foregroundShortSpeechMs',
      'foregroundLongSpeechMs',
      'foregroundMinDb',
      'foregroundOnsetMinDb',
      'foregroundLongMinDb',
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
        'tenVadConfirmationWindowMs' | 'tenVadMinSilenceDurationMs' | 'tenVadHangoverMs'
      >
    | null
    | undefined,
): number | null {
  if (
    typeof config?.tenVadConfirmationWindowMs !== 'number'
    || !Number.isFinite(config.tenVadConfirmationWindowMs)
    || typeof config?.tenVadMinSilenceDurationMs !== 'number'
    || !Number.isFinite(config.tenVadMinSilenceDurationMs)
    || typeof config?.tenVadHangoverMs !== 'number'
    || !Number.isFinite(config.tenVadHangoverMs)
  ) {
    return null;
  }

  return Math.max(
    config.tenVadConfirmationWindowMs,
    config.tenVadMinSilenceDurationMs,
    config.tenVadHangoverMs,
  );
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
    | Pick<Partial<StreamingDetectorConfig>, 'foregroundFilterEnabled' | 'foregroundMinDb'>
    | null
    | undefined,
  noiseFloorDbfs: number | null | undefined,
): number | null {
  if (
    !config?.foregroundFilterEnabled
    || typeof noiseFloorDbfs !== 'number'
    || !Number.isFinite(noiseFloorDbfs)
    || typeof config?.foregroundMinDb !== 'number'
    || !Number.isFinite(config.foregroundMinDb)
  ) {
    return null;
  }

  return noiseFloorDbfs + config.foregroundMinDb;
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
