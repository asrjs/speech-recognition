import {
  STREAMING_PROCESSING_SAMPLE_RATE,
  STREAMING_ROUGH_GATE_ANALYSIS_WINDOW_MS,
} from './audio-timeline.js';

export const STREAMING_PROFILE_IDS = {
  REALTIME_RNNT: 'realtime-rnnt',
  GENERIC_STREAMING: 'generic-streaming',
  AGGRESSIVE: 'aggressive',
  CONSERVATIVE: 'conservative',
  CUSTOM: 'custom',
} as const;

export type StreamingProfileId =
  (typeof STREAMING_PROFILE_IDS)[keyof typeof STREAMING_PROFILE_IDS];

export interface StreamingDetectorConfig {
  readonly sampleRate: number;
  readonly ringBufferDurationMs: number;
  readonly waveformPointCount: number;
  readonly analysisWindowMs: number;
  readonly energySmoothingWindows: number;
  readonly prerollMs: number;
  readonly minSpeechDurationMs: number;
  readonly minSilenceDurationMs: number;
  readonly maxSegmentDurationMs: number;
  readonly minSpeechLevelDbfs: number;
  readonly useSnrGate: boolean;
  readonly snrThreshold: number;
  readonly minSnrThreshold: number;
  readonly energyRiseThreshold: number;
  readonly maxOnsetLookbackChunks: number;
  readonly defaultOnsetLookbackChunks: number;
  readonly maxHistoryChunks: number;
  readonly initialNoiseFloor: number;
  readonly fastAdaptationRate: number;
  readonly slowAdaptationRate: number;
  readonly minBackgroundDurationSec: number;
  readonly levelWindowMs: number;
  readonly tenVadEnabled: boolean;
  readonly tenVadThreshold: number;
  readonly tenVadConfirmationWindowMs: number;
  readonly tenVadHangoverMs: number;
}

export interface StreamingDetectorPreset {
  readonly id: StreamingProfileId;
  readonly label: string;
  readonly mode: 'manual' | 'speech-detect';
  readonly config: StreamingDetectorConfig;
}

export const DEFAULT_STREAMING_DETECTOR_CONFIG: StreamingDetectorConfig = {
  sampleRate: STREAMING_PROCESSING_SAMPLE_RATE,
  ringBufferDurationMs: 12000,
  waveformPointCount: 180,
  analysisWindowMs: STREAMING_ROUGH_GATE_ANALYSIS_WINDOW_MS,
  energySmoothingWindows: 6,
  prerollMs: 320,
  minSpeechDurationMs: 320,
  minSilenceDurationMs: 500,
  maxSegmentDurationMs: 10000,
  minSpeechLevelDbfs: -38,
  useSnrGate: false,
  snrThreshold: 3.0,
  minSnrThreshold: 1.0,
  energyRiseThreshold: 0.08,
  maxOnsetLookbackChunks: 3,
  defaultOnsetLookbackChunks: 3,
  maxHistoryChunks: 20,
  initialNoiseFloor: 0.005,
  fastAdaptationRate: 0.15,
  slowAdaptationRate: 0.05,
  minBackgroundDurationSec: 1,
  levelWindowMs: 1000,
  tenVadEnabled: true,
  tenVadThreshold: 0.5,
  tenVadConfirmationWindowMs: 192,
  tenVadHangoverMs: 320,
};

export const STREAMING_PRESETS: Record<StreamingProfileId, StreamingDetectorPreset> = {
  [STREAMING_PROFILE_IDS.REALTIME_RNNT]: {
    id: STREAMING_PROFILE_IDS.REALTIME_RNNT,
    label: 'Realtime RNNT',
    mode: 'speech-detect',
    config: {
      ...DEFAULT_STREAMING_DETECTOR_CONFIG,
      prerollMs: 680,
      minSilenceDurationMs: 820,
      maxSegmentDurationMs: 3600,
      snrThreshold: 3.2,
      tenVadThreshold: 0.55,
    },
  },
  [STREAMING_PROFILE_IDS.GENERIC_STREAMING]: {
    id: STREAMING_PROFILE_IDS.GENERIC_STREAMING,
    label: 'Generic Streaming',
    mode: 'speech-detect',
    config: {
      ...DEFAULT_STREAMING_DETECTOR_CONFIG,
      minSilenceDurationMs: 400,
    },
  },
  [STREAMING_PROFILE_IDS.AGGRESSIVE]: {
    id: STREAMING_PROFILE_IDS.AGGRESSIVE,
    label: 'Aggressive',
    mode: 'speech-detect',
    config: {
      ...DEFAULT_STREAMING_DETECTOR_CONFIG,
      prerollMs: 520,
      minSilenceDurationMs: 560,
      snrThreshold: 1.75,
      tenVadThreshold: 0.42,
      tenVadConfirmationWindowMs: 128,
    },
  },
  [STREAMING_PROFILE_IDS.CONSERVATIVE]: {
    id: STREAMING_PROFILE_IDS.CONSERVATIVE,
    label: 'Conservative',
    mode: 'speech-detect',
    config: {
      ...DEFAULT_STREAMING_DETECTOR_CONFIG,
      prerollMs: 800,
      minSilenceDurationMs: 1100,
      snrThreshold: 3.2,
      minSnrThreshold: 1.75,
      tenVadThreshold: 0.6,
      tenVadConfirmationWindowMs: 256,
      tenVadHangoverMs: 480,
    },
  },
  [STREAMING_PROFILE_IDS.CUSTOM]: {
    id: STREAMING_PROFILE_IDS.CUSTOM,
    label: 'Custom',
    mode: 'speech-detect',
    config: DEFAULT_STREAMING_DETECTOR_CONFIG,
  },
};

export function resolveDefaultMicMode(isRealtimeEouModel: boolean): 'manual' | 'speech-detect' {
  return isRealtimeEouModel ? 'speech-detect' : 'manual';
}

export function resolveStreamingProfileId(isRealtimeEouModel: boolean): StreamingProfileId {
  return isRealtimeEouModel
    ? STREAMING_PROFILE_IDS.REALTIME_RNNT
    : STREAMING_PROFILE_IDS.GENERIC_STREAMING;
}

export function getStreamingPreset(profileId: string): StreamingDetectorPreset {
  return (
    STREAMING_PRESETS[profileId as StreamingProfileId] ??
    STREAMING_PRESETS[STREAMING_PROFILE_IDS.GENERIC_STREAMING]
  );
}

export function listStreamingPresets(): readonly StreamingDetectorPreset[] {
  return [
    STREAMING_PRESETS[STREAMING_PROFILE_IDS.REALTIME_RNNT],
    STREAMING_PRESETS[STREAMING_PROFILE_IDS.GENERIC_STREAMING],
    STREAMING_PRESETS[STREAMING_PROFILE_IDS.AGGRESSIVE],
    STREAMING_PRESETS[STREAMING_PROFILE_IDS.CONSERVATIVE],
  ];
}

function normalizeStreamingConfig(
  config: Partial<StreamingDetectorConfig> & { readonly energyThreshold?: number } = {},
): Partial<StreamingDetectorConfig> {
  const { energyThreshold, ...rest } = config;
  if (
    typeof energyThreshold === 'number' &&
    typeof rest.minSpeechLevelDbfs !== 'number'
  ) {
    return {
      ...rest,
      minSpeechLevelDbfs: 20 * Math.log10(Math.max(energyThreshold, 0.000001)),
    };
  }
  return rest;
}

export function mergeStreamingConfig(
  profileId: string,
  overrides: Partial<StreamingDetectorConfig> & { readonly energyThreshold?: number } = {},
): StreamingDetectorConfig {
  return {
    ...getStreamingPreset(profileId).config,
    ...normalizeStreamingConfig(overrides),
  };
}

export function isStreamingConfigEqual(
  left: Partial<StreamingDetectorConfig> | null | undefined,
  right: Partial<StreamingDetectorConfig> | null | undefined,
): boolean {
  const leftEntries = Object.entries(left ?? {});
  const rightEntries = Object.entries(right ?? {});
  if (leftEntries.length !== rightEntries.length) return false;
  return leftEntries.every(([key, value]) => right?.[key as keyof StreamingDetectorConfig] === value);
}
