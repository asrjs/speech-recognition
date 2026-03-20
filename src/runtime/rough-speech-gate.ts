import {
  DEFAULT_ROUGH_GATE_CONFIG,
  type RoughSpeechGateConfig,
} from './rough-gate-config.js';
import {
  durationMsToAlignedFrameCount,
  framesToMilliseconds,
} from './audio-timeline.js';

function amplitudeToDbfs(value: number, floorDbfs = -100): number {
  if (!Number.isFinite(value) || value <= 0) {
    return floorDbfs;
  }
  return Math.max(floorDbfs, 20 * Math.log10(value));
}

function dbfsToAmplitude(dbfs: number): number {
  return 10 ** (dbfs / 20);
}

export interface RoughSpeechChunkSummary {
  readonly startFrame: number;
  readonly endFrame: number;
  readonly energy: number;
  readonly snr: number;
  readonly isSpeech: boolean;
}

export interface RoughSpeechTimelinePoint {
  readonly startFrame: number;
  readonly endFrame: number;
  readonly energy: number;
  readonly speechRatio: number;
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
  private lastEnergyPass = false;
  private lastSnrPass = false;
  private lastCandidateReason: 'energy-threshold' | 'snr-threshold' | 'none' = 'none';
  private silenceDurationSec = 0;
  private processedFrames = 0;
  private windowBaseFrame = 0;
  private recentChunks: RoughSpeechChunkSummary[] = [];
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
    this.analysisWindowFrames = 0;
    this.analysisBuffer = new Float32Array(0);
    this.levelWindowFrames = 0;
    this.energyThresholdAmplitude = 0;
    this.minSpeechFrames = 0;
    this.minSilenceFrames = 0;
    this.rebuildDerivedFrames();
  }

  updateConfig(config: Partial<RoughSpeechGateConfig>): void {
    this.config = {
      ...this.config,
      ...config,
    };
    this.rebuildDerivedFrames();
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
        levelWindowMs: framesToMilliseconds(this.levelWindowFrames, this.config.sampleRate),
        noiseFloorDbfs: this.lastNoiseFloorDbfs,
        energyPass: this.lastEnergyPass,
        snrPass: this.lastSnrPass,
        candidateReason: this.lastCandidateReason,
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
    this.lastEnergyPass = energyPass;
    this.lastSnrPass = snrPass;
    this.lastCandidateReason = energyPass
      ? 'energy-threshold'
      : this.config.useSnrGate && snrPass
        ? 'snr-threshold'
        : 'none';

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
      levelWindowMs: framesToMilliseconds(this.levelWindowFrames, this.config.sampleRate),
      noiseFloorDbfs: this.lastNoiseFloorDbfs,
      energyPass,
      snrPass,
      candidateReason: this.lastCandidateReason,
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
    this.lastEnergyPass = false;
    this.lastSnrPass = false;
    this.lastCandidateReason = 'none';
    this.silenceDurationSec = 0;
    this.processedFrames = processedFrames;
    this.windowBaseFrame = processedFrames;
    this.recentChunks = [];
    this.recentWindowEnergies = [];
    this.analysisBufferIndex = 0;
    this.rebuildDerivedFrames();
    this.levelHistory = [];
    this.levelHistorySumSquares = 0;
    this.levelHistoryFrameCount = 0;
  }

  getRecentChunks(): readonly RoughSpeechChunkSummary[] {
    return this.recentChunks.map((chunk) => ({ ...chunk }));
  }

  getTimeline(
    startFrame: number,
    endFrame: number,
    pointCount: number,
  ): readonly RoughSpeechTimelinePoint[] {
    const safeStart = Math.max(0, Math.floor(startFrame));
    const safeEnd = Math.max(safeStart + 1, Math.floor(endFrame));
    const safePoints = Math.max(1, Math.floor(pointCount));
    const visibleChunks = this.recentChunks.filter(
      (chunk) => chunk.endFrame > safeStart && chunk.startFrame < safeEnd,
    );

    const points: RoughSpeechTimelinePoint[] = [];
    for (let index = 0; index < safePoints; index += 1) {
      const bucketStart =
        safeStart + Math.floor((index * (safeEnd - safeStart)) / safePoints);
      const bucketEnd =
        index === safePoints - 1
          ? safeEnd
          : safeStart + Math.floor(((index + 1) * (safeEnd - safeStart)) / safePoints);

      let totalWeight = 0;
      let weightedEnergy = 0;
      let weightedSpeech = 0;

      for (const chunk of visibleChunks) {
        const overlapStart = Math.max(bucketStart, chunk.startFrame);
        const overlapEnd = Math.min(bucketEnd, chunk.endFrame);
        const overlapFrames = overlapEnd - overlapStart;
        if (overlapFrames <= 0) {
          continue;
        }

        totalWeight += overlapFrames;
        weightedEnergy += chunk.energy * overlapFrames;
        weightedSpeech += (chunk.isSpeech ? 1 : 0) * overlapFrames;
      }

      const energy = totalWeight > 0 ? weightedEnergy / totalWeight : 0;
      const speechRatio = totalWeight > 0 ? weightedSpeech / totalWeight : 0;
      points.push({
        startFrame: bucketStart,
        endFrame: Math.max(bucketStart + 1, bucketEnd),
        energy,
        speechRatio,
        isSpeech: speechRatio >= 0.5,
      });
    }

    return points;
  }

  private rebuildDerivedFrames(): void {
    this.analysisWindowFrames = durationMsToAlignedFrameCount(
      this.config.analysisWindowMs,
      this.config.sampleRate,
      'ceil',
    );
    this.analysisBuffer = new Float32Array(this.analysisWindowFrames);
    this.levelWindowFrames = Math.max(
      this.analysisWindowFrames,
      durationMsToAlignedFrameCount(this.config.levelWindowMs, this.config.sampleRate, 'round'),
    );
    this.energyThresholdAmplitude = dbfsToAmplitude(this.config.minSpeechLevelDbfs);
    this.minSpeechFrames = durationMsToAlignedFrameCount(
      this.config.minSpeechDurationMs,
      this.config.sampleRate,
      'ceil',
    );
    this.minSilenceFrames = durationMsToAlignedFrameCount(
      this.config.minSilenceDurationMs,
      this.config.sampleRate,
      'ceil',
    );
  }
}
