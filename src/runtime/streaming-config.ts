import {
  STREAMING_PROCESSING_SAMPLE_RATE,
  STREAMING_ROUGH_GATE_ANALYSIS_WINDOW_MS,
  STREAMING_TIMELINE_CHUNK_MS,
  alignDurationMsToTimeline,
  resolveStreamingTimelineChunkDurationMs,
} from './audio-timeline.js';

const DEFAULT_ENERGY_SMOOTHING_DURATION_MS = 480;
const DEFAULT_MAX_ONSET_LOOKBACK_MS = 480;
const DEFAULT_ONSET_LOOKBACK_MS = 320;
const DEFAULT_TEN_VAD_MIN_SPEECH_DURATION_MS = 240;
const DEFAULT_TEN_VAD_MIN_SILENCE_DURATION_MS = 80;
const DEFAULT_TEN_VAD_SPEECH_PADDING_CHUNKS = 3;
const DEFAULT_ENERGY_THRESHOLD = 0.08;

export const PARAKEET_SEGMENTATION_PRESETS = {
  FAST: {
    id: 'fast',
    label: 'Fast',
    energyThreshold: 0.12,
    minSilenceDurationMs: 100,
    speechHangoverMs: 80,
  },
  MEDIUM: {
    id: 'medium',
    label: 'Medium',
    energyThreshold: 0.08,
    minSilenceDurationMs: 400,
    speechHangoverMs: 160,
  },
  SLOW: {
    id: 'slow',
    label: 'Slow',
    energyThreshold: 0.06,
    minSilenceDurationMs: 1000,
    speechHangoverMs: 240,
  },
} as const;

function amplitudeToDbfs(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return -120;
  }
  return 20 * Math.log10(value);
}

function dbfsToAmplitude(value: number): number {
  if (!Number.isFinite(value)) {
    return DEFAULT_ENERGY_THRESHOLD;
  }
  return 10 ** (value / 20);
}

export const STREAMING_PROFILE_IDS = {
  REALTIME_RNNT: 'realtime-rnnt',
  GENERIC_STREAMING: 'generic-streaming',
  AGGRESSIVE: 'aggressive',
  CONSERVATIVE: 'conservative',
  CUSTOM: 'custom',
} as const;

export type StreamingProfileId =
  (typeof STREAMING_PROFILE_IDS)[keyof typeof STREAMING_PROFILE_IDS];

export const STREAMING_GATE_MODES = {
  ROUGH_ONLY: 'rough-only',
  TEN_VAD_ONLY: 'ten-vad-only',
  ROUGH_AND_TEN_VAD: 'rough-and-ten-vad',
} as const;

export type StreamingGateMode =
  (typeof STREAMING_GATE_MODES)[keyof typeof STREAMING_GATE_MODES];

export interface StreamingDetectorConfig {
  readonly sampleRate: number;
  readonly chunkDurationMs: number;
  readonly gateMode: StreamingGateMode;
  readonly ringBufferDurationMs: number;
  readonly analysisWindowMs: number;
  readonly energySmoothingDurationMs: number;
  readonly energySmoothingWindows: number;
  readonly prerollMs: number;
  readonly overlapDurationMs: number;
  readonly speechHangoverMs: number;
  readonly minSpeechDurationMs: number;
  readonly minSilenceDurationMs: number;
  readonly maxSegmentDurationMs: number;
  readonly energyThreshold: number;
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
  readonly minEnergyIntegral: number;
  readonly minEnergyPerSecond: number;
  readonly useAdaptiveEnergyThresholds: boolean;
  readonly adaptiveEnergyIntegralFactor: number;
  readonly adaptiveEnergyPerSecondFactor: number;
  readonly minAdaptiveEnergyIntegral: number;
  readonly minAdaptiveEnergyPerSecond: number;
  readonly tenVadEnabled: boolean;
  readonly tenVadThreshold: number;
  readonly tenVadConfirmationWindowMs: number;
  readonly tenVadHangoverMs: number;
  readonly tenVadMinSpeechDurationMs: number;
  readonly tenVadMinSilenceDurationMs: number;
  readonly tenVadSpeechPaddingMs: number;
  readonly foregroundFilterEnabled: boolean;
  readonly foregroundMinDb: number;
  readonly foregroundOnsetMinDb: number;
  readonly foregroundOnsetWindowMs: number;
  readonly foregroundShortSpeechMs: number;
  readonly foregroundLongSpeechMs: number;
  readonly foregroundLongExcessDbMs: number;
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
        alignDuration(
          config.energySmoothingDurationMs ?? DEFAULT_ENERGY_SMOOTHING_DURATION_MS,
        ),
        analysisWindowMs,
      );
  const resolvedEnergySmoothingDurationMs =
    resolvedEnergySmoothingWindows * analysisWindowMs;
  const resolvedEnergyThreshold = Math.max(
    0.000001,
    config.energyThreshold
      ?? (typeof config.minSpeechLevelDbfs === 'number'
        ? dbfsToAmplitude(config.minSpeechLevelDbfs)
        : DEFAULT_ENERGY_THRESHOLD),
  );
  const resolvedMinSpeechLevelDbfs =
    typeof config.minSpeechLevelDbfs === 'number' && Number.isFinite(config.minSpeechLevelDbfs)
      ? config.minSpeechLevelDbfs
      : amplitudeToDbfs(resolvedEnergyThreshold);

  return {
    sampleRate,
    chunkDurationMs,
    gateMode: config.gateMode ?? STREAMING_GATE_MODES.ROUGH_ONLY,
    ringBufferDurationMs: alignDuration(config.ringBufferDurationMs ?? 12000),
    analysisWindowMs,
    energySmoothingDurationMs: resolvedEnergySmoothingDurationMs,
    energySmoothingWindows: resolvedEnergySmoothingWindows,
    prerollMs: alignDuration(config.prerollMs ?? 120),
    overlapDurationMs: alignDuration(config.overlapDurationMs ?? 80),
    speechHangoverMs: alignDuration(config.speechHangoverMs ?? 160),
    minSpeechDurationMs: alignDuration(config.minSpeechDurationMs ?? 240),
    minSilenceDurationMs: resolveOptionalZeroAlignedDuration(
      config.minSilenceDurationMs,
      400,
    ),
    maxSegmentDurationMs: alignDuration(config.maxSegmentDurationMs ?? 4800),
    energyThreshold: resolvedEnergyThreshold,
    minSpeechLevelDbfs: resolvedMinSpeechLevelDbfs,
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
    levelWindowMs: alignDuration(config.levelWindowMs ?? 480),
    minEnergyIntegral: config.minEnergyIntegral ?? 22,
    minEnergyPerSecond: config.minEnergyPerSecond ?? 5,
    useAdaptiveEnergyThresholds: config.useAdaptiveEnergyThresholds ?? true,
    adaptiveEnergyIntegralFactor: config.adaptiveEnergyIntegralFactor ?? 25,
    adaptiveEnergyPerSecondFactor: config.adaptiveEnergyPerSecondFactor ?? 10,
    minAdaptiveEnergyIntegral: config.minAdaptiveEnergyIntegral ?? 3,
    minAdaptiveEnergyPerSecond: config.minAdaptiveEnergyPerSecond ?? 1,
    tenVadEnabled: config.tenVadEnabled ?? false,
    tenVadThreshold: config.tenVadThreshold ?? 0.5,
    tenVadConfirmationWindowMs: alignDuration(
      config.tenVadConfirmationWindowMs ?? 192,
    ),
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
    foregroundFilterEnabled: config.foregroundFilterEnabled ?? true,
    foregroundMinDb: config.foregroundMinDb ?? 8,
    foregroundOnsetMinDb: config.foregroundOnsetMinDb ?? 10,
    foregroundOnsetWindowMs: alignDuration(config.foregroundOnsetWindowMs ?? 192),
    foregroundShortSpeechMs: alignDuration(config.foregroundShortSpeechMs ?? 240),
    foregroundLongSpeechMs: alignDuration(config.foregroundLongSpeechMs ?? 1200),
    foregroundLongExcessDbMs: config.foregroundLongExcessDbMs ?? 1800,
  };
}

export const DEFAULT_STREAMING_DETECTOR_CONFIG: StreamingDetectorConfig =
  deriveStreamingConfig();

export const STREAMING_PRESETS: Record<StreamingProfileId, StreamingDetectorPreset> = {
  [STREAMING_PROFILE_IDS.REALTIME_RNNT]: {
    id: STREAMING_PROFILE_IDS.REALTIME_RNNT,
    label: PARAKEET_SEGMENTATION_PRESETS.MEDIUM.label,
    mode: 'speech-detect',
    config: deriveStreamingConfig({
      gateMode: STREAMING_GATE_MODES.ROUGH_ONLY,
      prerollMs: 120,
      overlapDurationMs: 80,
      speechHangoverMs: PARAKEET_SEGMENTATION_PRESETS.MEDIUM.speechHangoverMs,
      minSilenceDurationMs: PARAKEET_SEGMENTATION_PRESETS.MEDIUM.minSilenceDurationMs,
      maxSegmentDurationMs: 4800,
      energyThreshold: PARAKEET_SEGMENTATION_PRESETS.MEDIUM.energyThreshold,
    }),
  },
  [STREAMING_PROFILE_IDS.GENERIC_STREAMING]: {
    id: STREAMING_PROFILE_IDS.GENERIC_STREAMING,
    label: PARAKEET_SEGMENTATION_PRESETS.MEDIUM.label,
    mode: 'speech-detect',
    config: deriveStreamingConfig({
      gateMode: STREAMING_GATE_MODES.ROUGH_ONLY,
      prerollMs: 120,
      overlapDurationMs: 80,
      speechHangoverMs: PARAKEET_SEGMENTATION_PRESETS.MEDIUM.speechHangoverMs,
      minSilenceDurationMs: PARAKEET_SEGMENTATION_PRESETS.MEDIUM.minSilenceDurationMs,
      maxSegmentDurationMs: 4800,
      energyThreshold: PARAKEET_SEGMENTATION_PRESETS.MEDIUM.energyThreshold,
    }),
  },
  [STREAMING_PROFILE_IDS.AGGRESSIVE]: {
    id: STREAMING_PROFILE_IDS.AGGRESSIVE,
    label: PARAKEET_SEGMENTATION_PRESETS.FAST.label,
    mode: 'speech-detect',
    config: deriveStreamingConfig({
      gateMode: STREAMING_GATE_MODES.ROUGH_ONLY,
      prerollMs: 120,
      overlapDurationMs: 80,
      speechHangoverMs: PARAKEET_SEGMENTATION_PRESETS.FAST.speechHangoverMs,
      minSilenceDurationMs: PARAKEET_SEGMENTATION_PRESETS.FAST.minSilenceDurationMs,
      energyThreshold: PARAKEET_SEGMENTATION_PRESETS.FAST.energyThreshold,
    }),
  },
  [STREAMING_PROFILE_IDS.CONSERVATIVE]: {
    id: STREAMING_PROFILE_IDS.CONSERVATIVE,
    label: PARAKEET_SEGMENTATION_PRESETS.SLOW.label,
    mode: 'speech-detect',
    config: deriveStreamingConfig({
      gateMode: STREAMING_GATE_MODES.ROUGH_ONLY,
      prerollMs: 120,
      overlapDurationMs: 80,
      speechHangoverMs: PARAKEET_SEGMENTATION_PRESETS.SLOW.speechHangoverMs,
      minSilenceDurationMs: PARAKEET_SEGMENTATION_PRESETS.SLOW.minSilenceDurationMs,
      energyThreshold: PARAKEET_SEGMENTATION_PRESETS.SLOW.energyThreshold,
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
  const {
    energyThreshold,
    waveformPointCount: _waveformPointCount,
    foregroundLongMinDb: _foregroundLongMinDb,
    ...rest
  } = config as StreamingDetectorConfigOverrides & { readonly foregroundLongMinDb?: number };
  const normalized: Record<string, unknown> = {
    ...rest,
  };
  if (typeof energyThreshold === 'number' && Number.isFinite(energyThreshold)) {
    const resolvedEnergyThreshold = Math.max(energyThreshold, 0.000001);
    normalized.energyThreshold = resolvedEnergyThreshold;
    if (typeof normalized.minSpeechLevelDbfs !== 'number') {
      normalized.minSpeechLevelDbfs = amplitudeToDbfs(resolvedEnergyThreshold);
    }
    return normalized as Partial<StreamingDetectorConfig>;
  }
  if (
    typeof normalized.minSpeechLevelDbfs === 'number'
    && Number.isFinite(normalized.minSpeechLevelDbfs)
    && typeof normalized.energyThreshold !== 'number'
  ) {
    normalized.energyThreshold = dbfsToAmplitude(normalized.minSpeechLevelDbfs);
  }
  return normalized as Partial<StreamingDetectorConfig>;
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
  return leftEntries.every(([key, value]) => right?.[key as keyof StreamingDetectorConfig] === value);
}
