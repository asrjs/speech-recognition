import { AssetLoadAbortedError, isAssetLoadAbortedError, isDomAbortError } from '../io/abort.js';
import type { RoughSpeechGateConfig } from './rough-gate-config.js';
import type { SegmentForegroundFilterConfig } from './segment-foreground-filter.js';
import { resolveStreamingTimelineChunkFrames } from './audio-timeline.js';
import {
  STREAMING_GATE_MODES,
  getStreamingPreset,
  type StreamingDetectorConfig,
  type StreamingDetectorConfigOverrides,
  type StreamingGateMode,
} from './streaming-config.js';
import type { StreamingTenVadStatus } from './streaming-detector-types.js';

export function buildRoughGateConfig(
  config: StreamingDetectorConfig,
  sampleRate: number,
): RoughSpeechGateConfig {
  return {
    sampleRate: sampleRate,
    analysisWindowMs: config.analysisWindowMs,
    energySmoothingWindows: config.energySmoothingWindows,
    energyThreshold: config.energyThreshold,
    minSpeechLevelDbfs: config.minSpeechLevelDbfs,
    useSnrGate: config.useSnrGate,
    snrThreshold: config.snrThreshold,
    minSnrThreshold: config.minSnrThreshold,
    energyRiseThreshold: config.energyRiseThreshold,
    maxOnsetLookbackChunks: config.maxOnsetLookbackChunks,
    defaultOnsetLookbackChunks: config.defaultOnsetLookbackChunks,
    maxHistoryChunks: Math.max(
      config.maxHistoryChunks,
      Math.ceil(config.ringBufferDurationMs / Math.max(1, config.analysisWindowMs)) + 2,
    ),
    minSpeechDurationMs: config.minSpeechDurationMs,
    minSilenceDurationMs: config.minSilenceDurationMs,
    initialNoiseFloor: config.initialNoiseFloor,
    fastAdaptationRate: config.fastAdaptationRate,
    slowAdaptationRate: config.slowAdaptationRate,
    minBackgroundDurationSec: config.minBackgroundDurationSec,
    levelWindowMs: config.levelWindowMs,
  };
}

export function buildTenVadConfig(
  config: StreamingDetectorConfig,
  sampleRate: number,
): Record<string, unknown> {
  return {
    vadBackend: config.vadBackend,
    sampleRate: sampleRate,
    hopSize: resolveStreamingTimelineChunkFrames(sampleRate, config.vadHopDurationMs),
    threshold: config.tenVadThreshold,
    confirmationWindowMs: config.tenVadConfirmationWindowMs,
    hangoverMs: config.tenVadHangoverMs,
    minSpeechDurationMs: config.tenVadMinSpeechDurationMs,
    minSilenceDurationMs: config.tenVadMinSilenceDurationMs,
    speechPaddingMs: config.tenVadSpeechPaddingMs,
  };
}

export function buildForegroundFilterConfig(
  config: StreamingDetectorConfig,
  continuationAfterMaxDuration: boolean,
): SegmentForegroundFilterConfig {
  const minSpeechDurationMs = continuationAfterMaxDuration
    ? Math.min(64, config.minSpeechDurationMs)
    : config.minSpeechDurationMs;
  const factor = continuationAfterMaxDuration
    ? minSpeechDurationMs / Math.max(1, config.minSpeechDurationMs)
    : 1;

  return {
    foregroundFilterEnabled: config.foregroundFilterEnabled,
    minSpeechDurationMs,
    minEnergyPerSecond: config.minEnergyPerSecond,
    minEnergyIntegral: config.minEnergyIntegral * factor,
    useAdaptiveEnergyThresholds: config.useAdaptiveEnergyThresholds,
    adaptiveEnergyIntegralFactor: config.adaptiveEnergyIntegralFactor * factor,
    adaptiveEnergyPerSecondFactor: config.adaptiveEnergyPerSecondFactor,
    minAdaptiveEnergyIntegral: config.minAdaptiveEnergyIntegral * factor,
    minAdaptiveEnergyPerSecond: config.minAdaptiveEnergyPerSecond,
  };
}

export function resolveEffectiveGateMode(
  config: StreamingDetectorConfig,
  hasTenVad: boolean,
  tenVadState: StreamingTenVadStatus['state'],
): StreamingGateMode {
  const isVadReady = tenVadState === 'ready';
  const isVadEnabled =
    hasTenVad &&
    config.tenVadEnabled &&
    (config as { readonly vadMode?: string }).vadMode !== 'disabled';

  if (isVadReady && isVadEnabled) {
    if (config.gateMode === STREAMING_GATE_MODES.TEN_VAD_ONLY) return STREAMING_GATE_MODES.VAD_ONLY;
    if (config.gateMode === STREAMING_GATE_MODES.ROUGH_AND_TEN_VAD)
      return STREAMING_GATE_MODES.HYBRID;
    if (
      config.gateMode === STREAMING_GATE_MODES.HYBRID ||
      config.gateMode === STREAMING_GATE_MODES.VAD_ONLY ||
      config.gateMode === STREAMING_GATE_MODES.ROUGH_ONLY
    ) {
      return config.gateMode;
    }
  }

  return STREAMING_GATE_MODES.ROUGH_ONLY;
}

export function getConfigOverrides(
  profileId: string,
  config: StreamingDetectorConfig,
): Partial<StreamingDetectorConfig> {
  const preset = getStreamingPreset(profileId).config;
  const overrides: Partial<StreamingDetectorConfig> = {};
  const mutableOverrides = overrides as Record<string, unknown>;
  for (const [key, value] of Object.entries(config) as Array<
    [keyof StreamingDetectorConfig, StreamingDetectorConfig[keyof StreamingDetectorConfig]]
  >) {
    if (preset[key] !== value) {
      mutableOverrides[key] = value;
    }
  }
  return overrides;
}

export function mergeConfigOverrides(
  currentOverrides: Partial<StreamingDetectorConfig>,
  partial: StreamingDetectorConfigOverrides & { readonly profileId?: string },
): StreamingDetectorConfigOverrides & { profileId?: string } {
  const nextOverrides = {
    ...currentOverrides,
    ...partial,
  };
  delete nextOverrides.profileId;
  if (
    Object.prototype.hasOwnProperty.call(partial, 'vadBackend') &&
    !Object.prototype.hasOwnProperty.call(partial, 'vadHopDurationMs')
  ) {
    delete nextOverrides.vadHopDurationMs;
  }
  if (
    Object.prototype.hasOwnProperty.call(partial, 'vadBackend') &&
    !Object.prototype.hasOwnProperty.call(partial, 'vadVisualBucketDurationMs')
  ) {
    delete nextOverrides.vadVisualBucketDurationMs;
  }
  return nextOverrides;
}

export function createStreamingVadInitAbortedError(): AssetLoadAbortedError {
  return new AssetLoadAbortedError('streaming-vad-init');
}

export function isStreamingVadInitAborted(
  error: unknown,
  signal?: { readonly aborted: boolean } | null,
): boolean {
  return isAssetLoadAbortedError(error) || isDomAbortError(error) || Boolean(signal?.aborted);
}
