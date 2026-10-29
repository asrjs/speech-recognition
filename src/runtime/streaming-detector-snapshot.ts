import type { RoughSpeechGate, RoughSpeechGateWindowResult } from './rough-speech-gate.js';
import type { SegmentForegroundFilterResult } from './segment-foreground-filter.js';
import type { ActiveSegmentState, FinalizedSegmentBounds } from './streaming-segment-bounds.js';
import {
  STREAMING_GATE_MODES,
  STREAMING_VAD_BACKENDS,
  type StreamingDetectorConfig,
  type StreamingGateMode,
} from './streaming-config.js';
import { amplitudeToDbfs } from './noise-floor.js';
import type {
  StreamingSpeechDetectorSnapshot,
  StreamingTenVadStatus,
} from './streaming-detector-types.js';

export function cloneEntries<T extends Record<string, unknown>>(entries: readonly T[]): T[] {
  return entries.map((entry) => ({ ...entry }));
}

export function createDisabledTenVadStatus(
  config: StreamingDetectorConfig,
  error: Error | null,
): StreamingTenVadStatus {
  return {
    state: config.tenVadEnabled ? 'idle' : 'disabled',
    error: error?.message ?? null,
    probability: 0,
    speaking: false,
    threshold: config.tenVadThreshold,
  };
}

export function getStreamingVadBackendName(config: StreamingDetectorConfig): string {
  return config.vadBackend === STREAMING_VAD_BACKENDS.TEN_VAD ? 'TEN-VAD' : 'FireRed VAD';
}

export function buildRoughSnapshot(
  lastMetrics: RoughSpeechGateWindowResult | null,
  config: StreamingDetectorConfig,
  roughGate: RoughSpeechGate,
  waveform: { readonly startFrame: number; readonly endFrame: number },
  waveformPointCount: number,
): StreamingSpeechDetectorSnapshot['rough'] {
  const defaultNoiseFloorDbfs = amplitudeToDbfs(config.initialNoiseFloor);
  return lastMetrics
    ? {
        energy: lastMetrics.energy,
        rawEnergy: lastMetrics.rawEnergy,
        snr: lastMetrics.snr,
        noiseFloor: lastMetrics.noiseFloor,
        noiseFloorDbfs: lastMetrics.noiseFloorDbfs,
        backgroundAverage: lastMetrics.backgroundAverage,
        backgroundAverageDbfs: lastMetrics.backgroundAverageDbfs,
        confirmedSilenceAverage: lastMetrics.confirmedSilenceAverage,
        confirmedSilenceAverageDbfs: lastMetrics.confirmedSilenceAverageDbfs,
        rejectedCandidateAverage: lastMetrics.rejectedCandidateAverage,
        rejectedCandidateAverageDbfs: lastMetrics.rejectedCandidateAverageDbfs,
        threshold: lastMetrics.threshold,
        thresholdDbfs: lastMetrics.thresholdDbfs,
        levelDbfs: lastMetrics.levelDbfs,
        levelWindowRms: lastMetrics.levelWindowRms,
        levelWindowDbfs: lastMetrics.levelWindowDbfs,
        levelWindowMs: lastMetrics.levelWindowMs,
        energyPass: lastMetrics.energyPass,
        candidateReason: lastMetrics.candidateReason,
        snrThreshold: lastMetrics.snrThreshold,
        minSnrThreshold: lastMetrics.minSnrThreshold,
        useSnrGate: config.useSnrGate,
        snrPass: lastMetrics.snrPass,
        isSpeech: lastMetrics.isSpeech,
        recent: roughGate.getRecentChunks(),
        timeline: roughGate.getTimeline(waveform.startFrame, waveform.endFrame, waveformPointCount),
      }
    : {
        energy: 0,
        rawEnergy: 0,
        snr: 0,
        noiseFloor: config.initialNoiseFloor,
        noiseFloorDbfs: defaultNoiseFloorDbfs,
        backgroundAverage: config.initialNoiseFloor,
        backgroundAverageDbfs: defaultNoiseFloorDbfs,
        confirmedSilenceAverage: config.initialNoiseFloor,
        confirmedSilenceAverageDbfs: defaultNoiseFloorDbfs,
        rejectedCandidateAverage: config.initialNoiseFloor,
        rejectedCandidateAverageDbfs: defaultNoiseFloorDbfs,
        threshold: config.energyThreshold,
        thresholdDbfs: config.minSpeechLevelDbfs,
        levelDbfs: -100,
        levelWindowRms: 0,
        levelWindowDbfs: -100,
        levelWindowMs: config.levelWindowMs,
        energyPass: false,
        candidateReason: 'none',
        snrThreshold: config.snrThreshold,
        minSnrThreshold: config.minSnrThreshold,
        useSnrGate: config.useSnrGate,
        snrPass: false,
        isSpeech: false,
        recent: [],
        timeline: [],
      };
}

export function buildDetectorWarnings(
  config: StreamingDetectorConfig,
  tenVadStatus: StreamingTenVadStatus | undefined,
  effectiveGateMode: StreamingGateMode,
): string[] {
  const warnings: string[] = [];
  const backendName = getStreamingVadBackendName(config);

  if (tenVadStatus?.state === 'degraded') {
    const detail =
      typeof tenVadStatus.error === 'string' && tenVadStatus.error.length > 0
        ? ` (${tenVadStatus.error})`
        : '';
    warnings.push(`${backendName} is degraded. Runtime plots may be incomplete.${detail}`);
  }

  if (config.tenVadEnabled && effectiveGateMode === STREAMING_GATE_MODES.ROUGH_ONLY) {
    warnings.push(
      `Speech segmentation is running on the rough gate. ${backendName} is diagnostics-only.`,
    );
  }

  if (
    config.gateMode !== STREAMING_GATE_MODES.ROUGH_ONLY &&
    effectiveGateMode === STREAMING_GATE_MODES.ROUGH_ONLY
  ) {
    const backendLabel =
      config.vadBackend === STREAMING_VAD_BACKENDS.FIRERED_VAD ? 'FireRed-first' : 'TEN-VAD-first';
    warnings.push(
      `Requested ${backendLabel} gating is ignored by this Parakeet-style detector port.`,
    );
  }

  if (!config.foregroundFilterEnabled) {
    warnings.push(
      'Final segment energy rejection is disabled. Quiet background audio may be accepted.',
    );
  }

  return warnings;
}

export function buildForegroundSnapshot(
  config: StreamingDetectorConfig,
  lastMetrics: RoughSpeechGateWindowResult | null,
  lastForegroundResult: SegmentForegroundFilterResult | null,
): StreamingSpeechDetectorSnapshot['foreground'] {
  const defaultNoiseFloorDbfs = amplitudeToDbfs(config.initialNoiseFloor);
  const lastNoiseFloor = lastMetrics?.noiseFloor ?? config.initialNoiseFloor;
  return {
    enabled: config.foregroundFilterEnabled,
    noiseFloor: lastNoiseFloor,
    noiseFloorDbfs: lastMetrics?.noiseFloorDbfs ?? defaultNoiseFloorDbfs,
    energyThreshold: config.energyThreshold,
    energyThresholdDbfs: config.minSpeechLevelDbfs,
    minSpeechDurationMs: config.minSpeechDurationMs,
    minEnergyPerSecond: config.minEnergyPerSecond,
    minEnergyIntegral: config.minEnergyIntegral,
    adaptiveEnergyThresholdsEnabled: config.useAdaptiveEnergyThresholds,
    activeResult: null,
    lastResult: lastForegroundResult as unknown as Record<string, unknown> | null,
  };
}

export function buildAcceptanceInfo(
  filterResult: SegmentForegroundFilterResult,
  reason: string,
  source: ActiveSegmentState['source'],
  bounds: FinalizedSegmentBounds,
  gateMode: StreamingGateMode,
  tenVadState: StreamingTenVadStatus['state'],
): Record<string, unknown> {
  return {
    accepted: filterResult.accepted,
    reason: filterResult.reason,
    segmentReason: reason,
    source: source,
    logicalStartFrame: bounds.logicalStartFrame,
    logicalEndFrame: bounds.logicalEndFrame,
    extractedStartFrame: bounds.extractedStartFrame,
    extractedEndFrame: bounds.extractedEndFrame,
    extractedDurationMs: Number(filterResult.durationMs.toFixed(1)),
    noiseFloor: filterResult.noiseFloor,
    noiseFloorDbfs: filterResult.noiseFloorDbfs,
    normalizedPowerAt16k: filterResult.normalizedPowerAt16k,
    normalizedEnergyIntegralAt16k: filterResult.normalizedEnergyIntegralAt16k,
    minEnergyPerSecondThreshold: filterResult.minEnergyPerSecondThreshold,
    minEnergyIntegralThreshold: filterResult.minEnergyIntegralThreshold,
    adaptiveThresholds: filterResult.usedAdaptiveThresholds,
    gateMode: gateMode,
    tenVadState: tenVadState,
  };
}
