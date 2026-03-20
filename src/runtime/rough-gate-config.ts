export interface RoughSpeechGateConfig {
  readonly sampleRate: number;
  readonly analysisWindowMs: number;
  readonly energySmoothingWindows: number;
  readonly minSpeechLevelDbfs: number;
  readonly useSnrGate: boolean;
  readonly snrThreshold: number;
  readonly minSnrThreshold: number;
  readonly energyRiseThreshold: number;
  readonly maxOnsetLookbackChunks: number;
  readonly defaultOnsetLookbackChunks: number;
  readonly maxHistoryChunks: number;
  readonly minSpeechDurationMs: number;
  readonly minSilenceDurationMs: number;
  readonly initialNoiseFloor: number;
  readonly fastAdaptationRate: number;
  readonly slowAdaptationRate: number;
  readonly minBackgroundDurationSec: number;
  readonly levelWindowMs: number;
}

export const DEFAULT_ROUGH_GATE_CONFIG: RoughSpeechGateConfig = {
  sampleRate: 16000,
  analysisWindowMs: 80,
  energySmoothingWindows: 6,
  minSpeechLevelDbfs: -38,
  useSnrGate: false,
  snrThreshold: 2.5,
  minSnrThreshold: 1.25,
  energyRiseThreshold: 0.08,
  maxOnsetLookbackChunks: 6,
  defaultOnsetLookbackChunks: 4,
  maxHistoryChunks: 24,
  minSpeechDurationMs: 240,
  minSilenceDurationMs: 800,
  initialNoiseFloor: 0.004,
  fastAdaptationRate: 0.15,
  slowAdaptationRate: 0.05,
  minBackgroundDurationSec: 1,
  levelWindowMs: 1000,
};
