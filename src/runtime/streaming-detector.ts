import { AudioRingBuffer } from './realtime.js';

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

const BASE_CONFIG: StreamingDetectorConfig = {
  sampleRate: 16000,
  ringBufferDurationMs: 8000,
  waveformPointCount: 180,
  analysisWindowMs: 80,
  energySmoothingWindows: 6,
  prerollMs: 600,
  minSpeechDurationMs: 240,
  minSilenceDurationMs: 800,
  maxSegmentDurationMs: 4000,
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
      ...BASE_CONFIG,
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
      ...BASE_CONFIG,
      minSilenceDurationMs: 400,
    },
  },
  [STREAMING_PROFILE_IDS.AGGRESSIVE]: {
    id: STREAMING_PROFILE_IDS.AGGRESSIVE,
    label: 'Aggressive',
    mode: 'speech-detect',
    config: {
      ...BASE_CONFIG,
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
      ...BASE_CONFIG,
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
    config: BASE_CONFIG,
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
  if (
    typeof config.energyThreshold === 'number' &&
    typeof config.minSpeechLevelDbfs !== 'number'
  ) {
    return {
      ...config,
      minSpeechLevelDbfs: 20 * Math.log10(Math.max(config.energyThreshold, 0.000001)),
    };
  }
  return config;
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

function amplitudeToDbfs(value: number, floorDbfs = -100): number {
  if (!Number.isFinite(value) || value <= 0) {
    return floorDbfs;
  }
  return Math.max(floorDbfs, 20 * Math.log10(value));
}

function dbfsToAmplitude(dbfs: number): number {
  return 10 ** (dbfs / 20);
}

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

interface RoughSpeechChunkHistory {
  readonly startFrame: number;
  readonly endFrame: number;
  readonly energy: number;
  readonly snr: number;
  readonly isSpeech: boolean;
}

interface LevelHistoryEntry {
  readonly sumSquares: number;
  readonly frameCount: number;
}

export interface RoughSpeechGateWindowResult {
  readonly isSpeech: boolean;
  readonly speechStart: boolean;
  readonly speechEnd: boolean;
  readonly onsetFrame: number | null;
  readonly energy: number;
  readonly snr: number;
  readonly noiseFloor: number;
  readonly threshold: number;
  readonly thresholdDbfs: number;
  readonly levelDbfs: number;
  readonly levelWindowRms: number;
  readonly levelWindowDbfs: number;
  readonly levelWindowMs: number;
  readonly noiseFloorDbfs: number;
  readonly energyPass: boolean;
  readonly snrPass: boolean;
  readonly candidateReason: 'energy-threshold' | 'snr-threshold' | 'none';
  readonly snrThreshold: number;
  readonly minSnrThreshold: number;
  readonly chunkStartFrame: number;
  readonly chunkEndFrame: number;
  readonly analysisWindowFrames: number;
  readonly inputChunkStartFrame?: number;
  readonly inputChunkEndFrame?: number;
}

export class RoughSpeechGate {
  private config: RoughSpeechGateConfig;
  private isSpeechActive = false;
  private speechConfirmationFrames = 0;
  private silenceConfirmationFrames = 0;
  private noiseFloor: number;
  private snr = 0;
  private lastEnergy = 0;
  private lastLevelDbfs = -100;
  private lastLevelWindowRms = 0;
  private lastLevelWindowDbfs = -100;
  private lastNoiseFloorDbfs: number;
  private silenceDurationSec = 0;
  private processedFrames = 0;
  private windowBaseFrame = 0;
  private recentChunks: RoughSpeechChunkHistory[] = [];
  private recentWindowEnergies: number[] = [];
  private analysisWindowFrames: number;
  private analysisBuffer: Float32Array;
  private analysisBufferIndex = 0;
  private levelWindowFrames: number;
  private levelHistory: LevelHistoryEntry[] = [];
  private levelHistorySumSquares = 0;
  private levelHistoryFrameCount = 0;
  private energyThresholdAmplitude: number;
  private minSpeechFrames: number;
  private minSilenceFrames: number;

  constructor(config: Partial<RoughSpeechGateConfig> = {}) {
    this.config = {
      ...DEFAULT_ROUGH_GATE_CONFIG,
      ...config,
    };
    this.noiseFloor = this.config.initialNoiseFloor;
    this.lastNoiseFloorDbfs = amplitudeToDbfs(this.noiseFloor);
    this.analysisWindowFrames = Math.max(
      1,
      Math.round((this.config.analysisWindowMs / 1000) * this.config.sampleRate),
    );
    this.analysisBuffer = new Float32Array(this.analysisWindowFrames);
    this.levelWindowFrames = Math.max(
      this.analysisWindowFrames,
      Math.round((this.config.levelWindowMs / 1000) * this.config.sampleRate),
    );
    this.energyThresholdAmplitude = dbfsToAmplitude(this.config.minSpeechLevelDbfs);
    this.minSpeechFrames = Math.ceil(
      (this.config.minSpeechDurationMs / 1000) * this.config.sampleRate,
    );
    this.minSilenceFrames = Math.ceil(
      (this.config.minSilenceDurationMs / 1000) * this.config.sampleRate,
    );
  }

  updateConfig(config: Partial<RoughSpeechGateConfig> = {}): void {
    const previousWindowFrames = this.analysisWindowFrames;
    this.config = {
      ...this.config,
      ...config,
    };
    this.analysisWindowFrames = Math.max(
      1,
      Math.round((this.config.analysisWindowMs / 1000) * this.config.sampleRate),
    );
    this.minSpeechFrames = Math.ceil(
      (this.config.minSpeechDurationMs / 1000) * this.config.sampleRate,
    );
    this.minSilenceFrames = Math.ceil(
      (this.config.minSilenceDurationMs / 1000) * this.config.sampleRate,
    );
    if (previousWindowFrames !== this.analysisWindowFrames) {
      this.analysisBuffer = new Float32Array(this.analysisWindowFrames);
      this.analysisBufferIndex = 0;
      this.recentWindowEnergies = [];
      this.levelHistory = [];
      this.levelHistorySumSquares = 0;
      this.levelHistoryFrameCount = 0;
    }
    this.levelWindowFrames = Math.max(
      this.analysisWindowFrames,
      Math.round((this.config.levelWindowMs / 1000) * this.config.sampleRate),
    );
    this.energyThresholdAmplitude = dbfsToAmplitude(this.config.minSpeechLevelDbfs);
  }

  process(chunk: Float32Array): RoughSpeechGateWindowResult {
    const inputStartFrame = this.processedFrames;
    this.processedFrames += chunk.length;

    let speechStart = false;
    let speechEnd = false;
    let onsetFrame: number | null = null;
    let lastResult: RoughSpeechGateWindowResult | null = null;
    let readOffset = 0;

    while (readOffset < chunk.length) {
      const remainingWindowFrames = this.analysisWindowFrames - this.analysisBufferIndex;
      const copyLength = Math.min(remainingWindowFrames, chunk.length - readOffset);
      this.analysisBuffer.set(
        chunk.subarray(readOffset, readOffset + copyLength),
        this.analysisBufferIndex,
      );
      this.analysisBufferIndex += copyLength;
      readOffset += copyLength;

      if (this.analysisBufferIndex >= this.analysisWindowFrames) {
        const windowEndFrame = this.windowBaseFrame + this.analysisWindowFrames;
        const result = this.processAnalysisWindow(
          this.analysisBuffer,
          this.windowBaseFrame,
          windowEndFrame,
        );
        lastResult = result;
        speechStart ||= result.speechStart;
        speechEnd ||= result.speechEnd;
        onsetFrame ??= result.onsetFrame;
        this.windowBaseFrame = windowEndFrame;
        this.analysisBufferIndex = 0;
      }
    }

    const fallbackEndFrame = Math.max(this.windowBaseFrame, this.processedFrames);
    const fallbackStartFrame = Math.max(0, fallbackEndFrame - this.analysisWindowFrames);

    return (
      lastResult ?? {
        isSpeech: this.isSpeechActive,
        speechStart,
        speechEnd,
        onsetFrame,
        energy: this.lastEnergy,
        snr: this.snr,
        noiseFloor: this.noiseFloor,
        threshold: this.energyThresholdAmplitude,
        thresholdDbfs: this.config.minSpeechLevelDbfs,
        levelDbfs: this.lastLevelDbfs,
        levelWindowRms: this.lastLevelWindowRms,
        levelWindowDbfs: this.lastLevelWindowDbfs,
        levelWindowMs: this.config.levelWindowMs,
        noiseFloorDbfs: this.lastNoiseFloorDbfs,
        energyPass: false,
        snrPass: false,
        candidateReason: 'none',
        snrThreshold: this.config.snrThreshold,
        minSnrThreshold: this.config.minSnrThreshold,
        chunkStartFrame: fallbackStartFrame,
        chunkEndFrame: fallbackEndFrame,
        analysisWindowFrames: this.analysisWindowFrames,
        inputChunkStartFrame: inputStartFrame,
        inputChunkEndFrame: this.processedFrames,
      }
    );
  }

  private processAnalysisWindow(
    window: Float32Array,
    chunkStartFrame: number,
    chunkEndFrame: number,
  ): RoughSpeechGateWindowResult {
    let sumSquares = 0;
    for (let index = 0; index < window.length; index += 1) {
      sumSquares += window[index]! * window[index]!;
    }

    const rawEnergy = window.length > 0 ? Math.sqrt(sumSquares / window.length) : 0;
    this.recentWindowEnergies.push(rawEnergy);
    if (this.recentWindowEnergies.length > this.config.energySmoothingWindows) {
      this.recentWindowEnergies.shift();
    }
    const smoothedEnergy =
      this.recentWindowEnergies.reduce((sum, value) => sum + value, 0) /
      Math.max(1, this.recentWindowEnergies.length);

    this.lastEnergy = smoothedEnergy;
    this.lastLevelDbfs = amplitudeToDbfs(smoothedEnergy);
    const chunkDurationSec = window.length / this.config.sampleRate;
    this.levelHistory.push({
      sumSquares,
      frameCount: window.length,
    });
    this.levelHistorySumSquares += sumSquares;
    this.levelHistoryFrameCount += window.length;
    while (
      this.levelHistory.length > 0 &&
      this.levelHistoryFrameCount - this.levelHistory[0]!.frameCount >= this.levelWindowFrames
    ) {
      const removed = this.levelHistory.shift()!;
      this.levelHistorySumSquares -= removed.sumSquares;
      this.levelHistoryFrameCount -= removed.frameCount;
    }
    const levelWindowMeanSquare =
      this.levelHistoryFrameCount > 0
        ? this.levelHistorySumSquares / this.levelHistoryFrameCount
        : 0;
    this.lastLevelWindowRms = Math.sqrt(levelWindowMeanSquare);
    this.lastLevelWindowDbfs = amplitudeToDbfs(this.lastLevelWindowRms);
    const safeNoiseFloor = Math.max(0.0001, this.noiseFloor);
    this.lastNoiseFloorDbfs = amplitudeToDbfs(safeNoiseFloor);
    this.snr = this.lastLevelDbfs - this.lastNoiseFloorDbfs;
    const energyPass = smoothedEnergy > this.energyThresholdAmplitude;
    const snrPass = this.snr > this.config.snrThreshold;
    const isCandidateSpeech = energyPass || (this.config.useSnrGate && snrPass);

    if (!isCandidateSpeech) {
      this.silenceDurationSec += chunkDurationSec;
      const adaptationRate =
        this.silenceDurationSec < this.config.minBackgroundDurationSec
          ? this.config.fastAdaptationRate
          : this.config.slowAdaptationRate;
      this.noiseFloor = this.noiseFloor * (1 - adaptationRate) + smoothedEnergy * adaptationRate;
      this.noiseFloor = Math.max(0.00001, this.noiseFloor);
    } else {
      this.silenceDurationSec = 0;
    }

    this.recentChunks.push({
      startFrame: chunkStartFrame,
      endFrame: chunkEndFrame,
      energy: smoothedEnergy,
      snr: this.snr,
      isSpeech: isCandidateSpeech,
    });
    if (this.recentChunks.length > this.config.maxHistoryChunks) {
      this.recentChunks.shift();
    }

    let speechStart = false;
    let speechEnd = false;
    let onsetFrame: number | null = null;

    if (isCandidateSpeech) {
      this.silenceConfirmationFrames = 0;
      if (!this.isSpeechActive) {
        this.speechConfirmationFrames += window.length;
        if (this.speechConfirmationFrames >= this.minSpeechFrames) {
          this.isSpeechActive = true;
          speechStart = true;
          onsetFrame = this.findSpeechStartFrame();
        }
      }
    } else {
      this.speechConfirmationFrames = 0;
      if (this.isSpeechActive) {
        this.silenceConfirmationFrames += window.length;
        if (this.silenceConfirmationFrames >= this.minSilenceFrames) {
          this.isSpeechActive = false;
          this.silenceConfirmationFrames = 0;
          speechEnd = true;
        }
      }
    }

    return {
      isSpeech: this.isSpeechActive,
      speechStart,
      speechEnd,
      onsetFrame,
      energy: smoothedEnergy,
      snr: this.snr,
      noiseFloor: this.noiseFloor,
      threshold: this.energyThresholdAmplitude,
      thresholdDbfs: this.config.minSpeechLevelDbfs,
      levelDbfs: this.lastLevelDbfs,
      levelWindowRms: this.lastLevelWindowRms,
      levelWindowDbfs: this.lastLevelWindowDbfs,
      levelWindowMs: this.config.levelWindowMs,
      noiseFloorDbfs: this.lastNoiseFloorDbfs,
      energyPass,
      snrPass,
      candidateReason: energyPass
        ? 'energy-threshold'
        : this.config.useSnrGate && snrPass
          ? 'snr-threshold'
          : 'none',
      snrThreshold: this.config.snrThreshold,
      minSnrThreshold: this.config.minSnrThreshold,
      chunkStartFrame,
      chunkEndFrame,
      analysisWindowFrames: this.analysisWindowFrames,
    };
  }

  private findSpeechStartFrame(): number | null {
    const chunks = this.recentChunks;
    if (!chunks.length) {
      return null;
    }

    let firstSpeechIndex = chunks.length - 1;
    for (let index = chunks.length - 1; index >= 0; index -= 1) {
      if (chunks[index]!.isSpeech) {
        firstSpeechIndex = index;
        break;
      }
    }

    let earliestRisingIndex = firstSpeechIndex;
    let foundRisingTrend = false;

    for (let index = firstSpeechIndex - 1; index >= 0; index -= 1) {
      if (chunks[index + 1]!.energy > chunks[index]!.energy * (1 + this.config.energyRiseThreshold)) {
        earliestRisingIndex = index;
        foundRisingTrend = true;
      }

      if (this.config.useSnrGate && chunks[index]!.snr < this.config.minSnrThreshold / 2) {
        break;
      }

      if (firstSpeechIndex - index > this.config.maxOnsetLookbackChunks) {
        break;
      }
    }

    if (foundRisingTrend) {
      return chunks[earliestRisingIndex]!.startFrame;
    }

    if (this.config.useSnrGate) {
      for (let index = firstSpeechIndex; index >= 0; index -= 1) {
        if (chunks[index]!.snr < this.config.minSnrThreshold) {
          const onsetIndex = Math.min(chunks.length - 1, index + 1);
          return chunks[onsetIndex]!.startFrame;
        }
      }
    }

    const fallbackIndex = Math.max(0, firstSpeechIndex - this.config.defaultOnsetLookbackChunks);
    return chunks[fallbackIndex]!.startFrame;
  }

  reset({ processedFrames = 0 }: { readonly processedFrames?: number } = {}): void {
    this.isSpeechActive = false;
    this.speechConfirmationFrames = 0;
    this.silenceConfirmationFrames = 0;
    this.noiseFloor = this.config.initialNoiseFloor;
    this.snr = 0;
    this.lastEnergy = 0;
    this.lastLevelDbfs = -100;
    this.lastLevelWindowRms = 0;
    this.lastLevelWindowDbfs = -100;
    this.lastNoiseFloorDbfs = amplitudeToDbfs(this.noiseFloor);
    this.silenceDurationSec = 0;
    this.processedFrames = processedFrames;
    this.windowBaseFrame = processedFrames;
    this.recentChunks = [];
    this.recentWindowEnergies = [];
    this.analysisWindowFrames = Math.max(
      1,
      Math.round((this.config.analysisWindowMs / 1000) * this.config.sampleRate),
    );
    this.analysisBuffer = new Float32Array(this.analysisWindowFrames);
    this.analysisBufferIndex = 0;
    this.levelWindowFrames = Math.max(
      this.analysisWindowFrames,
      Math.round((this.config.levelWindowMs / 1000) * this.config.sampleRate),
    );
    this.levelHistory = [];
    this.levelHistorySumSquares = 0;
    this.levelHistoryFrameCount = 0;
    this.energyThresholdAmplitude = dbfsToAmplitude(this.config.minSpeechLevelDbfs);
    this.minSpeechFrames = Math.ceil(
      (this.config.minSpeechDurationMs / 1000) * this.config.sampleRate,
    );
    this.minSilenceFrames = Math.ceil(
      (this.config.minSilenceDurationMs / 1000) * this.config.sampleRate,
    );
  }
}

export interface StreamingTenVadStatus {
  readonly state: 'idle' | 'initializing' | 'ready' | 'degraded' | 'disabled';
  readonly error: string | null;
  readonly probability: number;
  readonly speaking: boolean;
  readonly threshold: number;
}

export interface StreamingTenVadWindowSummary {
  readonly totalHops: number;
  readonly speechHopCount: number;
  readonly nonSpeechHopCount: number;
  readonly maxConsecutiveSpeech: number;
  readonly maxProbability: number;
  readonly recent: readonly unknown[];
}

export interface StreamingTenVadResultEvent {
  readonly type: 'result';
  readonly payload: unknown;
}

export interface StreamingTenVadLike {
  subscribe(listener: (event: StreamingTenVadResultEvent) => void): () => void;
  init(): Promise<void>;
  reset(): Promise<void>;
  dispose(): Promise<void>;
  updateConfig(config?: Record<string, unknown>): void;
  process(samples: Float32Array, globalSampleOffset: number): boolean;
  getStatus(): StreamingTenVadStatus;
  findFirstSpeechFrame(startFrame: number, endFrame: number): number | null;
  hasRecentSpeech(endFrame: number, windowMs: number, sampleRate: number): boolean;
  hasRecentSilence(endFrame: number, windowMs: number, sampleRate: number): boolean;
  getWindowSummary(
    endFrame: number,
    windowMs: number,
    sampleRate: number,
  ): StreamingTenVadWindowSummary;
}

export interface StreamingDetectorSegment {
  readonly startFrame: number;
  readonly endFrame: number;
  readonly sampleRate: number;
  readonly reason: string;
  readonly metadata: {
    readonly profileId: string;
    readonly rough: RoughSpeechGateWindowResult | null;
    readonly tenVad: StreamingTenVadStatus;
  };
  readPcm(): Float32Array;
}

export interface StreamingSpeechDetectorSnapshot {
  readonly state: 'idle' | 'listening' | 'candidate' | 'speaking';
  readonly sampleRate: number;
  readonly profileId: string;
  readonly config: StreamingDetectorConfig;
  readonly waveform: ReturnType<AudioRingBuffer['getMinMaxPairs']>;
  readonly activeSegment: { readonly startFrame: number; readonly endFrame: number } | null;
  readonly pendingSegmentStartFrame: number | null;
  readonly recentSegments: readonly StreamingDetectorSegment[];
  readonly recentDecisions: ReadonlyArray<{
    readonly id: string;
    readonly at: string;
    readonly message: string;
    readonly meta: Record<string, unknown>;
  }>;
  readonly acceptance: Record<string, unknown> | null;
  readonly rough: Record<string, unknown>;
  readonly tenVad: StreamingTenVadStatus;
  readonly warnings: readonly string[];
  readonly error: string | null;
}

export type StreamingSpeechDetectorEvent =
  | { readonly type: 'metrics'; readonly payload: StreamingSpeechDetectorSnapshot }
  | {
      readonly type: 'speech-start';
      readonly payload: { readonly startFrame: number; readonly sampleRate: number };
    }
  | {
      readonly type: 'speech-update';
      readonly payload: {
        readonly startFrame: number;
        readonly endFrame: number;
        readonly sampleRate: number;
      };
    }
  | {
      readonly type: 'speech-end';
      readonly payload: {
        readonly startFrame: number;
        readonly endFrame: number;
        readonly sampleRate: number;
      };
    }
  | { readonly type: 'segment-ready'; readonly payload: StreamingDetectorSegment }
  | { readonly type: 'error'; readonly payload: Error };

interface ActiveSegmentState {
  readonly startFrame: number;
  readonly lastFrame: number;
  readonly reason: string;
}

export interface StreamingSpeechDetectorOptions {
  readonly profileId?: string;
  readonly config?: Partial<StreamingDetectorConfig> & { readonly energyThreshold?: number };
  readonly isRealtimeEouModel?: boolean;
  readonly tenVadFactory?: (
    config: Record<string, unknown>,
    options?: unknown,
  ) => StreamingTenVadLike;
  readonly tenVadOptions?: unknown;
}

function cloneSegments<T extends Record<string, unknown>>(segments: readonly T[]): T[] {
  return segments.map((segment) => ({ ...segment }));
}

export class StreamingSpeechDetector {
  private readonly createTenVad;
  private profileId: string;
  private config: StreamingDetectorConfig;
  private listeners = new Set<(event: StreamingSpeechDetectorEvent) => void>();
  private sampleRate: number;
  private ringBuffer: AudioRingBuffer;
  private roughGate: RoughSpeechGate;
  private tenVad: StreamingTenVadLike | null;
  private state: 'idle' | 'listening' | 'candidate' | 'speaking' = 'idle';
  private activeSegment: ActiveSegmentState | null = null;
  private pendingSegmentStartFrame: number | null = null;
  private recentSegments: StreamingDetectorSegment[] = [];
  private recentDecisions: Array<{
    readonly id: string;
    readonly at: string;
    readonly message: string;
    readonly meta: Record<string, unknown>;
  }> = [];
  private lastMetrics: RoughSpeechGateWindowResult | null = null;
  private lastAcceptanceInfo: Record<string, unknown> | null = null;
  private lastError: Error | null = null;
  private disposed = false;
  private readonly tenVadUnsubscribe: (() => void) | undefined;

  constructor(options: StreamingSpeechDetectorOptions = {}) {
    this.createTenVad =
      options.tenVadFactory ??
      (() => {
        throw new Error('StreamingSpeechDetector requires a tenVadFactory when TEN-VAD is enabled.');
      });
    this.profileId =
      options.profileId ?? resolveStreamingProfileId(options.isRealtimeEouModel === true);
    this.config = mergeStreamingConfig(this.profileId, options.config);
    this.sampleRate = this.config.sampleRate;
    this.ringBuffer = new AudioRingBuffer({
      sampleRate: this.sampleRate,
      durationSeconds: this.config.ringBufferDurationMs / 1000,
    });
    this.roughGate = new RoughSpeechGate(this.buildRoughGateConfig());
    this.tenVad = this.config.tenVadEnabled
      ? this.createTenVad(this.buildTenVadConfig(), options.tenVadOptions)
      : null;

    this.tenVadUnsubscribe = this.tenVad?.subscribe((event) => {
      if (event.type === 'result') {
        this.emit({
          type: 'metrics',
          payload: this.getSnapshot(),
        });
      }
    });
  }

  private buildRoughGateConfig(): RoughSpeechGateConfig {
    return {
      sampleRate: this.sampleRate,
      analysisWindowMs: this.config.analysisWindowMs,
      energySmoothingWindows: this.config.energySmoothingWindows,
      minSpeechLevelDbfs: this.config.minSpeechLevelDbfs,
      useSnrGate: this.config.useSnrGate,
      snrThreshold: this.config.snrThreshold,
      minSnrThreshold: this.config.minSnrThreshold,
      energyRiseThreshold: this.config.energyRiseThreshold,
      maxOnsetLookbackChunks: this.config.maxOnsetLookbackChunks,
      defaultOnsetLookbackChunks: this.config.defaultOnsetLookbackChunks,
      maxHistoryChunks: this.config.maxHistoryChunks,
      minSpeechDurationMs: this.config.minSpeechDurationMs,
      minSilenceDurationMs: this.config.minSilenceDurationMs,
      initialNoiseFloor: this.config.initialNoiseFloor,
      fastAdaptationRate: this.config.fastAdaptationRate,
      slowAdaptationRate: this.config.slowAdaptationRate,
      minBackgroundDurationSec: this.config.minBackgroundDurationSec,
      levelWindowMs: this.config.levelWindowMs,
    };
  }

  private buildTenVadConfig(): Record<string, unknown> {
    return {
      threshold: this.config.tenVadThreshold,
      confirmationWindowMs: this.config.tenVadConfirmationWindowMs,
      hangoverMs: this.config.tenVadHangoverMs,
    };
  }

  subscribe(listener: (event: StreamingSpeechDetectorEvent) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private emit(event: StreamingSpeechDetectorEvent): void {
    for (const listener of this.listeners) {
      listener(event);
    }
  }

  private recordDecision(message: string, meta: Record<string, unknown> = {}): void {
    const entry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      at: new Date().toLocaleTimeString(),
      message,
      meta,
    };
    this.recentDecisions = [...this.recentDecisions.slice(-19), entry];
    console.debug(`[StreamingSpeechDetector] ${message}`, meta);
  }

  async start({ sampleRate }: { readonly sampleRate?: number } = {}): Promise<void> {
    if (this.disposed) {
      throw new Error('StreamingSpeechDetector has been disposed.');
    }

    if (sampleRate && sampleRate !== this.sampleRate) {
      this.sampleRate = sampleRate;
      this.ringBuffer = new AudioRingBuffer({
        sampleRate: this.sampleRate,
        durationSeconds: this.config.ringBufferDurationMs / 1000,
      });
      this.roughGate = new RoughSpeechGate(this.buildRoughGateConfig());
    }

    this.state = 'listening';
    this.activeSegment = null;
    this.pendingSegmentStartFrame = null;
    this.lastError = null;
    this.ringBuffer.reset();
    this.roughGate.reset();

    if (this.tenVad) {
      try {
        await this.tenVad.init();
        await this.tenVad.reset();
        this.recordDecision(
          'TEN-VAD ready',
          this.tenVad.getStatus() as unknown as Record<string, unknown>,
        );
      } catch (error) {
        this.lastError = error instanceof Error ? error : new Error(String(error));
        this.recordDecision('TEN-VAD degraded, falling back to rough gate', {
          error: this.lastError.message,
        });
        this.emit({
          type: 'error',
          payload: this.lastError,
        });
      }
    }

    this.emit({
      type: 'metrics',
      payload: this.getSnapshot(),
    });
  }

  processChunk(
    chunk: Float32Array,
    meta: { readonly startFrame?: number; readonly endFrame?: number } = {},
  ): void {
    if (!chunk.length) return;

    const chunkStartFrame = meta.startFrame ?? this.ringBuffer.getCurrentFrame();
    this.ringBuffer.write(chunk);
    this.tenVad?.process(chunk, chunkStartFrame);

    const rough = this.roughGate.process(chunk);
    this.lastMetrics = rough;

    const nowFrame = this.ringBuffer.getCurrentFrame();
    const isNearSpeechThreshold =
      rough.levelDbfs >= this.config.minSpeechLevelDbfs - 6 ||
      rough.snr >= this.config.minSnrThreshold - 1;

    if (rough.speechStart) {
      const fallbackStart = Math.max(
        this.ringBuffer.getBaseFrameOffset(),
        rough.onsetFrame ?? nowFrame - chunk.length,
      );
      const roughStart = Math.max(
        this.ringBuffer.getBaseFrameOffset(),
        fallbackStart - Math.ceil((this.config.prerollMs / 1000) * this.sampleRate),
      );
      const tenVadSummary = this.tenVad?.getWindowSummary(
        nowFrame,
        this.config.tenVadConfirmationWindowMs,
        this.sampleRate,
      );
      const tenVadStart = this.tenVad?.findFirstSpeechFrame(roughStart, nowFrame) ?? null;
      const startFrame = Math.max(this.ringBuffer.getBaseFrameOffset(), tenVadStart ?? roughStart);
      if (
        this.tenVad &&
        this.tenVad.getStatus().state === 'ready' &&
        tenVadStart === null &&
        !this.tenVad.hasRecentSpeech(
          nowFrame,
          this.config.tenVadConfirmationWindowMs,
          this.sampleRate,
        )
      ) {
        this.pendingSegmentStartFrame = startFrame;
        this.state = 'candidate';
        this.recordDecision('Rough onset waiting for TEN-VAD confirmation', {
          startFrame,
          reason: rough.candidateReason,
          energy: Number(rough.energy.toFixed(4)),
          levelDbfs: Number(rough.levelDbfs.toFixed(1)),
          snr: Number(rough.snr.toFixed(2)),
          tenVadState: this.tenVad.getStatus() as unknown as Record<string, unknown>,
          tenVadSummary: tenVadSummary as unknown as Record<string, unknown>,
        });
      } else {
        this.activateSegment(startFrame, nowFrame, rough, tenVadStart, tenVadSummary ?? null);
      }
    }

    if (
      !this.activeSegment &&
      this.pendingSegmentStartFrame !== null &&
      rough.isSpeech &&
      (!this.tenVad ||
        this.tenVad.getStatus().state !== 'ready' ||
        this.tenVad.hasRecentSpeech(
          nowFrame,
          this.config.tenVadConfirmationWindowMs,
          this.sampleRate,
        ))
    ) {
      this.activateSegment(
        this.pendingSegmentStartFrame,
        nowFrame,
        rough,
        this.pendingSegmentStartFrame,
        this.tenVad?.getWindowSummary(
          nowFrame,
          this.config.tenVadConfirmationWindowMs,
          this.sampleRate,
        ) ?? null,
      );
    }

    if (
      this.pendingSegmentStartFrame !== null &&
      !rough.isSpeech &&
      !rough.speechStart &&
      (!this.tenVad ||
        this.tenVad.getStatus().state !== 'ready' ||
        !this.tenVad.hasRecentSpeech(
          nowFrame,
          this.config.tenVadConfirmationWindowMs,
          this.sampleRate,
        ))
    ) {
      this.recordDecision('Candidate rejected by TEN-VAD / duration guard', {
        startFrame: this.pendingSegmentStartFrame,
        reason: rough.candidateReason,
        energy: Number(rough.energy.toFixed(4)),
        levelDbfs: Number(rough.levelDbfs.toFixed(1)),
        snr: Number(rough.snr.toFixed(2)),
        tenVadSummary:
          this.tenVad?.getWindowSummary(
            nowFrame,
            this.config.tenVadConfirmationWindowMs,
            this.sampleRate,
          ) ?? null,
      });
      this.pendingSegmentStartFrame = null;
    }

    if (this.activeSegment) {
      this.activeSegment = {
        ...this.activeSegment,
        lastFrame: nowFrame,
      };

      const segmentFrames = nowFrame - this.activeSegment.startFrame;
      const segmentDurationMs = (segmentFrames / this.sampleRate) * 1000;
      if (segmentDurationMs >= this.config.maxSegmentDurationMs) {
        this.finalizeSegment('max-duration', nowFrame);
      } else if (rough.isSpeech) {
        this.state = 'speaking';
        this.emit({
          type: 'speech-update',
          payload: {
            startFrame: this.activeSegment.startFrame,
            endFrame: nowFrame,
            sampleRate: this.sampleRate,
          },
        });
      } else if (rough.speechEnd) {
        const tenVadAllowsEnd =
          !this.tenVad ||
          this.tenVad.getStatus().state !== 'ready' ||
          this.tenVad.hasRecentSilence(
            nowFrame,
            this.config.tenVadConfirmationWindowMs,
            this.sampleRate,
          );

        if (tenVadAllowsEnd) {
          this.recordDecision('Speech finalized after silence', {
            endFrame: nowFrame,
            energy: Number(rough.energy.toFixed(4)),
            levelDbfs: Number(rough.levelDbfs.toFixed(1)),
            snr: Number(rough.snr.toFixed(2)),
          });
          this.finalizeSegment('silence', nowFrame);
        } else {
          this.state = 'candidate';
          this.recordDecision('Silence rejected by TEN-VAD hangover', {
            endFrame: nowFrame,
            probability: this.tenVad?.getStatus().probability ?? 0,
          });
        }
      } else if (
        this.tenVad &&
        this.tenVad.getStatus().state === 'ready' &&
        this.tenVad.hasRecentSpeech(nowFrame, this.config.tenVadHangoverMs, this.sampleRate)
      ) {
        this.state = 'candidate';
        this.recordDecision('Speech tail held by TEN-VAD hangover', {
          endFrame: nowFrame,
          probability: this.tenVad.getStatus().probability,
        });
      }
    } else if (isNearSpeechThreshold && !rough.speechStart && !rough.isSpeech) {
      this.recordDecision('Candidate rejected before speech start', {
        reason: rough.candidateReason,
        energy: Number(rough.energy.toFixed(4)),
        levelDbfs: Number(rough.levelDbfs.toFixed(1)),
        snr: Number(rough.snr.toFixed(2)),
        energyThresholdDbfs: this.config.minSpeechLevelDbfs,
        useSnrGate: this.config.useSnrGate,
        snrThreshold: this.config.snrThreshold,
        minSpeechDurationMs: this.config.minSpeechDurationMs,
      });
    }

    this.emit({
      type: 'metrics',
      payload: this.getSnapshot(),
    });
  }

  private activateSegment(
    startFrame: number,
    nowFrame: number,
    rough: RoughSpeechGateWindowResult,
    tenVadStart: number | null,
    tenVadSummary: StreamingTenVadWindowSummary | null,
  ): void {
    this.pendingSegmentStartFrame = null;
    this.activeSegment = {
      startFrame,
      lastFrame: nowFrame,
      reason: 'speech-start',
    };
    this.state = 'speaking';
    this.lastAcceptanceInfo = {
      reason:
        rough.candidateReason === 'energy-threshold'
          ? tenVadStart !== null
            ? 'Accepted because energy crossed threshold and TEN-VAD confirmed speech'
            : 'Accepted because energy crossed threshold'
          : 'Accepted by rough speech gate',
      energy: rough.energy,
      threshold: rough.threshold,
      thresholdDbfs: rough.thresholdDbfs,
      levelDbfs: rough.levelDbfs,
      levelWindowDbfs: rough.levelWindowDbfs,
      energyPass: rough.energyPass,
      snr: rough.snr,
      snrPass: rough.snrPass,
      useSnrGate: this.config.useSnrGate,
      tenVadConfirmed: tenVadStart !== null,
      tenVadState: this.tenVad?.getStatus().state ?? 'disabled',
    };
    this.recordDecision('Speech accepted', {
      startFrame,
      reason: rough.candidateReason,
      energy: Number(rough.energy.toFixed(4)),
      levelDbfs: Number(rough.levelDbfs.toFixed(1)),
      snr: Number(rough.snr.toFixed(2)),
      tenVadStart,
      tenVadSummary: tenVadSummary as unknown as Record<string, unknown> | null,
    });
    this.emit({
      type: 'speech-start',
      payload: {
        startFrame,
        sampleRate: this.sampleRate,
      },
    });
  }

  finalizeSegment(
    reason: string,
    endFrame = this.ringBuffer.getCurrentFrame(),
  ): StreamingDetectorSegment | null {
    if (!this.activeSegment) {
      return null;
    }

    const startFrame = this.activeSegment.startFrame;
    const segment: StreamingDetectorSegment = {
      startFrame,
      endFrame,
      sampleRate: this.sampleRate,
      reason,
      metadata: {
        profileId: this.profileId,
        rough: this.lastMetrics,
        tenVad:
          this.tenVad?.getStatus() ?? {
            state: 'disabled',
            error: null,
            probability: 0,
            speaking: false,
            threshold: this.config.tenVadThreshold,
          },
      },
      readPcm: () => this.ringBuffer.read(startFrame, endFrame),
    };

    this.recentSegments = [...this.recentSegments.slice(-11), segment];
    this.recordDecision(`Segment ready: ${reason}`, {
      startFrame: segment.startFrame,
      endFrame: segment.endFrame,
      durationSec: Number(
        ((segment.endFrame - segment.startFrame) / segment.sampleRate).toFixed(2),
      ),
    });
    this.emit({
      type: 'speech-end',
      payload: {
        startFrame: segment.startFrame,
        endFrame: segment.endFrame,
        sampleRate: segment.sampleRate,
      },
    });
    this.emit({
      type: 'segment-ready',
      payload: segment,
    });

    this.activeSegment = null;
    this.pendingSegmentStartFrame = null;
    this.state = 'listening';
    return segment;
  }

  flush(reason = 'manual'): StreamingDetectorSegment | null {
    return this.finalizeSegment(reason);
  }

  async stop({
    flush = true,
  }: {
    readonly flush?: boolean;
  } = {}): Promise<StreamingDetectorSegment | null> {
    let segment: StreamingDetectorSegment | null = null;
    if (flush) {
      segment = this.finalizeSegment('stop');
    } else {
      this.activeSegment = null;
      this.pendingSegmentStartFrame = null;
    }
    this.state = 'idle';
    this.emit({
      type: 'metrics',
      payload: this.getSnapshot(),
    });
    return segment;
  }

  updateConfig(
    partial: Partial<StreamingDetectorConfig> & { readonly profileId?: string } = {},
  ): void {
    this.config = {
      ...this.config,
      ...partial,
    };
    this.roughGate.updateConfig(this.buildRoughGateConfig());
    this.tenVad?.updateConfig(this.buildTenVadConfig());
    this.profileId = partial.profileId ?? this.profileId;
    this.emit({
      type: 'metrics',
      payload: this.getSnapshot(),
    });
  }

  getSnapshot(): StreamingSpeechDetectorSnapshot {
    const waveform = this.ringBuffer.getMinMaxPairs(
      this.config.waveformPointCount,
      Math.ceil((this.config.ringBufferDurationMs / 1000) * this.sampleRate),
    );

    return {
      state: this.state,
      sampleRate: this.sampleRate,
      profileId: this.profileId,
      config: { ...this.config },
      waveform,
      activeSegment: this.activeSegment
        ? {
            startFrame: this.activeSegment.startFrame,
            endFrame: this.ringBuffer.getCurrentFrame(),
          }
        : null,
      pendingSegmentStartFrame: this.pendingSegmentStartFrame,
      recentSegments: this.recentSegments.map((segment) => ({ ...segment })),
      recentDecisions: cloneSegments(this.recentDecisions),
      acceptance: this.lastAcceptanceInfo,
      rough: this.lastMetrics
        ? {
            energy: this.lastMetrics.energy,
            snr: this.lastMetrics.snr,
            noiseFloor: this.lastMetrics.noiseFloor,
            noiseFloorDbfs: this.lastMetrics.noiseFloorDbfs,
            threshold: this.lastMetrics.threshold,
            thresholdDbfs: this.lastMetrics.thresholdDbfs,
            levelDbfs: this.lastMetrics.levelDbfs,
            levelWindowRms: this.lastMetrics.levelWindowRms,
            levelWindowDbfs: this.lastMetrics.levelWindowDbfs,
            levelWindowMs: this.lastMetrics.levelWindowMs,
            energyPass: this.lastMetrics.energyPass,
            candidateReason: this.lastMetrics.candidateReason,
            snrThreshold: this.lastMetrics.snrThreshold,
            minSnrThreshold: this.lastMetrics.minSnrThreshold,
            useSnrGate: this.config.useSnrGate,
            snrPass: this.lastMetrics.snrPass,
            isSpeech: this.lastMetrics.isSpeech,
          }
        : {
            energy: 0,
            snr: 0,
            noiseFloor: this.config.initialNoiseFloor,
            noiseFloorDbfs: 20 * Math.log10(Math.max(this.config.initialNoiseFloor, 0.000001)),
            threshold: 10 ** (this.config.minSpeechLevelDbfs / 20),
            thresholdDbfs: this.config.minSpeechLevelDbfs,
            levelDbfs: -100,
            levelWindowRms: 0,
            levelWindowDbfs: -100,
            levelWindowMs: this.config.levelWindowMs,
            energyPass: false,
            candidateReason: 'none',
            snrThreshold: this.config.snrThreshold,
            minSnrThreshold: this.config.minSnrThreshold,
            useSnrGate: this.config.useSnrGate,
            snrPass: false,
            isSpeech: false,
          },
      tenVad:
        this.tenVad?.getStatus() ?? {
          state: this.config.tenVadEnabled ? 'idle' : 'disabled',
          error: this.lastError?.message ?? null,
          probability: 0,
          speaking: false,
          threshold: this.config.tenVadThreshold,
        },
      warnings: this.buildWarnings(),
      error: this.lastError?.message ?? null,
    };
  }

  private buildWarnings(): string[] {
    const warnings: string[] = [];
    const tenVadStatus = this.tenVad?.getStatus();

    if (tenVadStatus?.state === 'degraded') {
      warnings.push('TEN-VAD is degraded. Speech detection is running on rough energy gating only.');
    }

    if (this.state === 'candidate' && tenVadStatus?.state === 'ready' && !tenVadStatus.speaking) {
      warnings.push('TEN-VAD has not confirmed speech yet. Candidate onset is still being held.');
    }

    if (
      this.lastAcceptanceInfo &&
      this.lastAcceptanceInfo.tenVadConfirmed === false &&
      tenVadStatus?.state !== 'degraded'
    ) {
      warnings.push(
        'Latest accepted speech came from the rough gate without TEN-VAD confirmation.',
      );
    }

    if (this.lastMetrics && !this.config.useSnrGate) {
      warnings.push(
        'SNR is shown for diagnostics only. Speech triggering currently uses the dBFS speech-level threshold.',
      );
    }

    return warnings;
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    this.tenVadUnsubscribe?.();
    await this.tenVad?.dispose();
    this.listeners.clear();
  }
}
