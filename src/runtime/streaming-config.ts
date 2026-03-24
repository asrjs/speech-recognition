import {
  STREAMING_PROCESSING_SAMPLE_RATE,
  STREAMING_ROUGH_GATE_ANALYSIS_WINDOW_MS,
  STREAMING_TIMELINE_CHUNK_MS,
  alignDurationMsToTimeline,
  resolveStreamingTimelineChunkDurationMs,
} from './audio-timeline.js';

const DEFAULT_ENERGY_SMOOTHING_DURATION_MS = 480;
const DEFAULT_MAX_ONSET_LOOKBACK_MS = 240;
const DEFAULT_ONSET_LOOKBACK_MS = 240;
const DEFAULT_TEN_VAD_MIN_SPEECH_DURATION_MS = 240;
const DEFAULT_TEN_VAD_MIN_SILENCE_DURATION_MS = 80;
const DEFAULT_TEN_VAD_SPEECH_PADDING_CHUNKS = 3;

export const STREAMING_PROFILE_IDS = {
  REALTIME_RNNT: 'realtime-rnnt',
  GENERIC_STREAMING: 'generic-streaming',
  AGGRESSIVE: 'aggressive',
  CONSERVATIVE: 'conservative',
  CUSTOM: 'custom',
} as const;

export type StreamingProfileId = (typeof STREAMING_PROFILE_IDS)[keyof typeof STREAMING_PROFILE_IDS];

export const STREAMING_GATE_MODES = {
  ROUGH_ONLY: 'rough-only',
  TEN_VAD_ONLY: 'ten-vad-only',
  ROUGH_AND_TEN_VAD: 'rough-and-ten-vad',
} as const;

export type StreamingGateMode = (typeof STREAMING_GATE_MODES)[keyof typeof STREAMING_GATE_MODES];

export interface StreamingDetectorConfig {
  readonly sampleRate: number;
  readonly chunkDurationMs: number;
  readonly gateMode: StreamingGateMode;
  readonly ringBufferDurationMs: number;
  readonly analysisWindowMs: number;
  readonly energySmoothingDurationMs: number;
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
  readonly tenVadMinSpeechDurationMs: number;
  readonly tenVadMinSilenceDurationMs: number;
  readonly tenVadSpeechPaddingMs: number;
}

export interface StreamingDetectorPreset {
  readonly id: StreamingProfileId;
  readonly label: string;
  readonly mode: 'manual' | 'speech-detect';
  readonly config: StreamingDetectorConfig;
}

function resolveAnalysisWindowCount(durationMs: number, analysisWindowMs: number): number {
  const safeAnalysisWindowMs = Math.max(1, analysisWindowMs);
  return Math.max(1, Math.round(durationMs / safeAnalysisWindowMs));
}

function hasOwnConfigValue<Key extends keyof StreamingDetectorConfig>(
  config: Partial<StreamingDetectorConfig>,
  key: Key,
): boolean {
  return Object.prototype.hasOwnProperty.call(config, key);
}

function deriveStreamingConfig(
  config: Partial<StreamingDetectorConfig> = {},
): StreamingDetectorConfig {
  const sampleRate = config.sampleRate ?? STREAMING_PROCESSING_SAMPLE_RATE;
  const chunkDurationMs = resolveStreamingTimelineChunkDurationMs(
    config.chunkDurationMs ?? STREAMING_TIMELINE_CHUNK_MS,
  );
  const alignDuration = (durationMs: number): number =>
    alignDurationMsToTimeline(durationMs, chunkDurationMs, 'ceil');
  const analysisWindowMs = alignDuration(
    config.analysisWindowMs ?? STREAMING_ROUGH_GATE_ANALYSIS_WINDOW_MS,
  );
  const resolveOptionalZeroAlignedDuration = (
    durationMs: number | undefined,
    fallbackMs: number,
  ): number => {
    if (durationMs === 0) {
      return 0;
    }
    return alignDuration(durationMs ?? fallbackMs);
  };
  const resolvedEnergySmoothingWindows = hasOwnConfigValue(config, 'energySmoothingWindows')
    ? Math.max(1, Math.round(config.energySmoothingWindows ?? 1))
    : resolveAnalysisWindowCount(
        alignDuration(config.energySmoothingDurationMs ?? DEFAULT_ENERGY_SMOOTHING_DURATION_MS),
        analysisWindowMs,
      );
  const resolvedEnergySmoothingDurationMs = resolvedEnergySmoothingWindows * analysisWindowMs;

  return {
    sampleRate,
    chunkDurationMs,
    gateMode: config.gateMode ?? STREAMING_GATE_MODES.ROUGH_AND_TEN_VAD,
    ringBufferDurationMs: alignDuration(config.ringBufferDurationMs ?? 12000),
    analysisWindowMs,
    energySmoothingDurationMs: resolvedEnergySmoothingDurationMs,
    energySmoothingWindows: resolvedEnergySmoothingWindows,
    prerollMs: alignDuration(config.prerollMs ?? 320),
    minSpeechDurationMs: alignDuration(config.minSpeechDurationMs ?? 320),
    minSilenceDurationMs: resolveOptionalZeroAlignedDuration(config.minSilenceDurationMs, 500),
    maxSegmentDurationMs: alignDuration(config.maxSegmentDurationMs ?? 10000),
    minSpeechLevelDbfs: config.minSpeechLevelDbfs ?? -38,
    useSnrGate: config.useSnrGate ?? false,
    snrThreshold: config.snrThreshold ?? 3.0,
    minSnrThreshold: config.minSnrThreshold ?? 1.0,
    energyRiseThreshold: config.energyRiseThreshold ?? 0.08,
    maxOnsetLookbackChunks: hasOwnConfigValue(config, 'maxOnsetLookbackChunks')
      ? Math.max(1, Math.round(config.maxOnsetLookbackChunks ?? 1))
      : resolveAnalysisWindowCount(DEFAULT_MAX_ONSET_LOOKBACK_MS, analysisWindowMs),
    defaultOnsetLookbackChunks: hasOwnConfigValue(config, 'defaultOnsetLookbackChunks')
      ? Math.max(1, Math.round(config.defaultOnsetLookbackChunks ?? 1))
      : resolveAnalysisWindowCount(DEFAULT_ONSET_LOOKBACK_MS, analysisWindowMs),
    maxHistoryChunks: Math.max(1, Math.round(config.maxHistoryChunks ?? 20)),
    initialNoiseFloor: config.initialNoiseFloor ?? 0.005,
    fastAdaptationRate: config.fastAdaptationRate ?? 0.15,
    slowAdaptationRate: config.slowAdaptationRate ?? 0.05,
    minBackgroundDurationSec: config.minBackgroundDurationSec ?? 1,
    levelWindowMs: alignDuration(config.levelWindowMs ?? 1000),
    tenVadEnabled: config.tenVadEnabled ?? true,
    tenVadThreshold: config.tenVadThreshold ?? 0.5,
    tenVadConfirmationWindowMs: alignDuration(config.tenVadConfirmationWindowMs ?? 192),
    tenVadHangoverMs: alignDuration(config.tenVadHangoverMs ?? 320),
    tenVadMinSpeechDurationMs: alignDuration(
      config.tenVadMinSpeechDurationMs ?? DEFAULT_TEN_VAD_MIN_SPEECH_DURATION_MS,
    ),
    tenVadMinSilenceDurationMs: alignDuration(
      config.tenVadMinSilenceDurationMs ?? DEFAULT_TEN_VAD_MIN_SILENCE_DURATION_MS,
    ),
    tenVadSpeechPaddingMs: alignDuration(
      config.tenVadSpeechPaddingMs ?? chunkDurationMs * DEFAULT_TEN_VAD_SPEECH_PADDING_CHUNKS,
    ),
  };
}

export const DEFAULT_STREAMING_DETECTOR_CONFIG: StreamingDetectorConfig = deriveStreamingConfig();

export const STREAMING_PRESETS: Record<StreamingProfileId, StreamingDetectorPreset> = {
  [STREAMING_PROFILE_IDS.REALTIME_RNNT]: {
    id: STREAMING_PROFILE_IDS.REALTIME_RNNT,
    label: 'Realtime RNNT',
    mode: 'speech-detect',
    config: deriveStreamingConfig({
      prerollMs: 680,
      minSilenceDurationMs: 820,
      maxSegmentDurationMs: 3600,
      snrThreshold: 3.2,
      tenVadThreshold: 0.55,
    }),
  },
  [STREAMING_PROFILE_IDS.GENERIC_STREAMING]: {
    id: STREAMING_PROFILE_IDS.GENERIC_STREAMING,
    label: 'Generic Streaming',
    mode: 'speech-detect',
    config: deriveStreamingConfig({
      minSilenceDurationMs: 400,
    }),
  },
  [STREAMING_PROFILE_IDS.AGGRESSIVE]: {
    id: STREAMING_PROFILE_IDS.AGGRESSIVE,
    label: 'Aggressive',
    mode: 'speech-detect',
    config: deriveStreamingConfig({
      prerollMs: 520,
      minSilenceDurationMs: 560,
      snrThreshold: 1.75,
      tenVadThreshold: 0.42,
      tenVadConfirmationWindowMs: 128,
    }),
  },
  [STREAMING_PROFILE_IDS.CONSERVATIVE]: {
    id: STREAMING_PROFILE_IDS.CONSERVATIVE,
    label: 'Conservative',
    mode: 'speech-detect',
    config: deriveStreamingConfig({
      prerollMs: 800,
      minSilenceDurationMs: 1100,
      snrThreshold: 3.2,
      minSnrThreshold: 1.75,
      tenVadThreshold: 0.6,
      tenVadConfirmationWindowMs: 256,
      tenVadHangoverMs: 480,
    }),
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

export interface StreamingDetectorConfigOverrides extends Partial<StreamingDetectorConfig> {
  readonly energyThreshold?: number;
  // Deprecated and ignored. Waveform resolution is derived from ringBufferDurationMs and timeline chunk size.
  readonly waveformPointCount?: number;
}

function normalizeStreamingConfig(
  config: StreamingDetectorConfigOverrides = {},
): Partial<StreamingDetectorConfig> {
  const { energyThreshold, waveformPointCount: _waveformPointCount, ...rest } = config;
  if (typeof energyThreshold === 'number' && typeof rest.minSpeechLevelDbfs !== 'number') {
    return {
      ...rest,
      minSpeechLevelDbfs: 20 * Math.log10(Math.max(energyThreshold, 0.000001)),
    };
  }
  return rest;
}

export function mergeStreamingConfig(
  profileId: string,
  overrides: StreamingDetectorConfigOverrides = {},
): StreamingDetectorConfig {
  const presetConfig = {
    ...getStreamingPreset(profileId).config,
  } as Record<string, unknown>;
  const normalizedOverrides = normalizeStreamingConfig(overrides);

  if (
    hasOwnConfigValue(normalizedOverrides, 'chunkDurationMs') ||
    hasOwnConfigValue(normalizedOverrides, 'analysisWindowMs') ||
    hasOwnConfigValue(normalizedOverrides, 'energySmoothingDurationMs')
  ) {
    if (!hasOwnConfigValue(normalizedOverrides, 'energySmoothingWindows')) {
      delete presetConfig.energySmoothingWindows;
    }
    if (!hasOwnConfigValue(normalizedOverrides, 'maxOnsetLookbackChunks')) {
      delete presetConfig.maxOnsetLookbackChunks;
    }
    if (!hasOwnConfigValue(normalizedOverrides, 'defaultOnsetLookbackChunks')) {
      delete presetConfig.defaultOnsetLookbackChunks;
    }
  }

  if (
    hasOwnConfigValue(normalizedOverrides, 'chunkDurationMs') &&
    !hasOwnConfigValue(normalizedOverrides, 'tenVadSpeechPaddingMs')
  ) {
    delete presetConfig.tenVadSpeechPaddingMs;
  }

  return deriveStreamingConfig({
    ...(presetConfig as Partial<StreamingDetectorConfig>),
    ...normalizedOverrides,
  });
}

export function isStreamingConfigEqual(
  left: Partial<StreamingDetectorConfig> | null | undefined,
  right: Partial<StreamingDetectorConfig> | null | undefined,
): boolean {
  const leftEntries = Object.entries(left ?? {});
  const rightEntries = Object.entries(right ?? {});
  if (leftEntries.length !== rightEntries.length) return false;
  return leftEntries.every(
    ([key, value]) => right?.[key as keyof StreamingDetectorConfig] === value,
  );
}
