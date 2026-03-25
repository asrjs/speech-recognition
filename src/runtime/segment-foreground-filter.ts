import { amplitudeToDbfs } from './noise-floor.js';

export interface SegmentForegroundFilterConfig {
  readonly foregroundFilterEnabled: boolean;
  readonly minSpeechDurationMs: number;
  readonly minEnergyPerSecond: number;
  readonly minEnergyIntegral: number;
  readonly useAdaptiveEnergyThresholds: boolean;
  readonly adaptiveEnergyIntegralFactor: number;
  readonly adaptiveEnergyPerSecondFactor: number;
  readonly minAdaptiveEnergyIntegral: number;
  readonly minAdaptiveEnergyPerSecond: number;
}

export interface SegmentForegroundSpeechHop {
  readonly startFrame: number;
  readonly endFrame: number;
  readonly probability: number;
  readonly speaking: boolean;
}

export interface SegmentForegroundFilterResult {
  readonly accepted: boolean;
  readonly reason: string;
  readonly durationMs: number;
  readonly noiseFloor: number;
  readonly noiseFloorDbfs: number;
  readonly averagePower: number;
  readonly normalizedPowerAt16k: number;
  readonly normalizedEnergyIntegralAt16k: number;
  readonly minEnergyPerSecondThreshold: number;
  readonly minEnergyIntegralThreshold: number;
  readonly usedAdaptiveThresholds: boolean;
  readonly segmentP90Dbfs: number;
  readonly onsetP90Dbfs: number;
  readonly speechDbfs: number;
  readonly foregroundDb: number;
  readonly onsetDb: number;
  readonly speechNoiseRatio: number;
  readonly effectiveForegroundThresholdDbfs: number;
  readonly effectiveOnsetThresholdDbfs: number;
  readonly excessEnergyDbMs: number;
  readonly effectiveLongExcessDbMs: number;
  readonly shortSpeech: boolean;
  readonly longSpeech: boolean;
  readonly usedSpeechWindowCount: number;
}

function resolveAveragePower(pcm: Float32Array): number {
  if (!pcm.length) {
    return 0;
  }
  let sumSquares = 0;
  for (let index = 0; index < pcm.length; index += 1) {
    const sample = pcm[index] ?? 0;
    sumSquares += sample * sample;
  }
  return sumSquares / pcm.length;
}

function resolveRmsDbfs(averagePower: number): number {
  return amplitudeToDbfs(Math.sqrt(Math.max(0, averagePower)));
}

export function scoreSegmentForeground(
  pcm: Float32Array,
  sampleRate: number,
  noiseFloor: number,
  config: SegmentForegroundFilterConfig,
  options: {
    readonly noiseWindowFrames?: number | null;
    readonly speechHops?: readonly SegmentForegroundSpeechHop[] | null;
    readonly segmentStartFrame?: number | null;
  } = {},
): SegmentForegroundFilterResult {
  const durationMs =
    sampleRate > 0 ? Math.max(0, (pcm.length / sampleRate) * 1000) : 0;
  const durationSeconds = durationMs / 1000;
  const resolvedNoiseFloor = Math.max(0.00001, noiseFloor);
  const noiseFloorDbfs = amplitudeToDbfs(resolvedNoiseFloor);
  const averagePower = resolveAveragePower(pcm);
  const normalizedPowerAt16k = averagePower * 16000;
  const normalizedEnergyIntegralAt16k = normalizedPowerAt16k * durationSeconds;
  const shortSpeech = durationMs < config.minSpeechDurationMs;
  const longSpeech = durationMs >= config.minSpeechDurationMs;
  const noiseWindowFrames = Math.max(1, Math.round(options.noiseWindowFrames ?? 1));

  let minEnergyIntegralThreshold = config.minEnergyIntegral;
  let minEnergyPerSecondThreshold = config.minEnergyPerSecond;
  let usedAdaptiveThresholds = false;

  if (config.useAdaptiveEnergyThresholds) {
    const normalizedNoiseFloor = resolvedNoiseFloor / noiseWindowFrames;
    const noiseFloorAt16k = normalizedNoiseFloor * 16000;
    minEnergyIntegralThreshold = Math.max(
      config.minAdaptiveEnergyIntegral,
      noiseFloorAt16k * config.adaptiveEnergyIntegralFactor,
    );
    minEnergyPerSecondThreshold = Math.max(
      config.minAdaptiveEnergyPerSecond,
      noiseFloorAt16k * config.adaptiveEnergyPerSecondFactor,
    );
    usedAdaptiveThresholds = true;
  }

  const segmentDbfs = resolveRmsDbfs(averagePower);
  const foregroundDb = segmentDbfs - noiseFloorDbfs;
  const speechNoiseRatio = Number.isFinite(foregroundDb)
    ? 10 ** (foregroundDb / 20)
    : 1;
  const thresholdRms = Math.sqrt(Math.max(minEnergyPerSecondThreshold / 16000, 0));
  const thresholdDbfs = amplitudeToDbfs(thresholdRms);

  let accepted = config.foregroundFilterEnabled;
  let reason = config.foregroundFilterEnabled ? 'accepted' : 'foreground-filter-disabled';

  if (config.foregroundFilterEnabled) {
    if (durationMs < config.minSpeechDurationMs) {
      accepted = false;
      reason = 'too-short';
    } else if (normalizedEnergyIntegralAt16k < minEnergyIntegralThreshold) {
      accepted = false;
      reason = 'low-energy-integral';
    } else if (normalizedPowerAt16k < minEnergyPerSecondThreshold) {
      accepted = false;
      reason = 'low-energy-per-second';
    }
  }

  return {
    accepted,
    reason,
    durationMs,
    noiseFloor: resolvedNoiseFloor,
    noiseFloorDbfs,
    averagePower,
    normalizedPowerAt16k,
    normalizedEnergyIntegralAt16k,
    minEnergyPerSecondThreshold,
    minEnergyIntegralThreshold,
    usedAdaptiveThresholds,
    segmentP90Dbfs: segmentDbfs,
    onsetP90Dbfs: segmentDbfs,
    speechDbfs: segmentDbfs,
    foregroundDb,
    onsetDb: foregroundDb,
    speechNoiseRatio,
    effectiveForegroundThresholdDbfs: thresholdDbfs,
    effectiveOnsetThresholdDbfs: thresholdDbfs,
    excessEnergyDbMs: normalizedEnergyIntegralAt16k,
    effectiveLongExcessDbMs: minEnergyIntegralThreshold,
    shortSpeech,
    longSpeech,
    usedSpeechWindowCount: Array.isArray(options.speechHops) ? options.speechHops.length : 0,
  };
}
