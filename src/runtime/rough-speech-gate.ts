import {
  DEFAULT_ROUGH_GATE_CONFIG,
  type RoughSpeechGateConfig,
} from './rough-gate-config.js';

function amplitudeToDbfs(value: number, floorDbfs = -100): number {
  if (!Number.isFinite(value) || value <= 0) {
    return floorDbfs;
  }
  return Math.max(floorDbfs, 20 * Math.log10(value));
}

function dbfsToAmplitude(dbfs: number): number {
  return 10 ** (dbfs / 20);
}

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

  updateConfig(config: Partial<RoughSpeechGateConfig>): void {
    this.config = {
      ...this.config,
      ...config,
    };
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
