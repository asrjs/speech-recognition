import type { VoiceActivityEvent } from '../types/index.js';
import type { StreamingActivityBuffer } from './realtime.js';

export interface VoiceActivityObservation {
  readonly startFrame: number;
  readonly endFrame: number;
  readonly speechProbability: number;
  readonly isSpeech?: boolean;
}

export interface VoiceActivitySegment extends VoiceActivityObservation {
  readonly isSpeech: boolean;
  readonly startTimeSeconds: number;
  readonly endTimeSeconds: number;
}

export interface VoiceActivityTimelineOptions {
  readonly sampleRate: number;
  readonly maxDurationSeconds: number;
  readonly speechThreshold?: number;
}

export interface VoiceActivityTimelineSnapshot {
  readonly sampleRate: number;
  readonly latestFrame: number;
  readonly baseFrame: number;
  readonly trailingSilenceSeconds: number;
  readonly segments: readonly VoiceActivitySegment[];
}

export interface VoiceActivityProbabilityBufferOptions {
  readonly sampleRate: number;
  readonly maxDurationSeconds: number;
  readonly hopFrames?: number;
  readonly speechThreshold?: number;
}

export interface VoiceActivityProbabilityBufferWindowSummary {
  readonly totalHops: number;
  readonly speechHopCount: number;
  readonly nonSpeechHopCount: number;
  readonly maxConsecutiveSpeech: number;
  readonly maxProbability: number;
  readonly recent: readonly {
    readonly startFrame: number;
    readonly endFrame: number;
    readonly probability: number;
    readonly speaking: boolean;
  }[];
}

export interface VoiceActivityProbabilityTimelinePoint {
  readonly startFrame: number;
  readonly endFrame: number;
  readonly probability: number;
  readonly speechRatio: number;
  readonly speaking: boolean;
}

export class VoiceActivityProbabilityBuffer implements StreamingActivityBuffer {
  readonly sampleRate: number;
  readonly hopFrames: number;
  readonly speechThreshold: number;
  readonly maxEntries: number;
  private readonly buffer: Float32Array;
  private nextEntryIndex = 0;

  constructor(options: VoiceActivityProbabilityBufferOptions) {
    if (!Number.isFinite(options.sampleRate) || options.sampleRate <= 0) {
      throw new TypeError('VoiceActivityProbabilityBuffer requires a positive sampleRate.');
    }
    if (!Number.isFinite(options.maxDurationSeconds) || options.maxDurationSeconds <= 0) {
      throw new TypeError(
        'VoiceActivityProbabilityBuffer requires a positive maxDurationSeconds.',
      );
    }

    this.sampleRate = options.sampleRate;
    this.hopFrames = Math.max(1, Math.floor(options.hopFrames ?? 512));
    this.speechThreshold = options.speechThreshold ?? 0.5;
    this.maxEntries = Math.max(
      1,
      Math.ceil((options.sampleRate * options.maxDurationSeconds) / this.hopFrames),
    );
    this.buffer = new Float32Array(this.maxEntries);
  }

  appendProbability(probability: number): number {
    const writePosition = this.nextEntryIndex % this.maxEntries;
    this.buffer[writePosition] = probability;
    this.nextEntryIndex += 1;
    return this.getLatestFrame();
  }

  appendProbabilities(probabilities: Float32Array | readonly number[]): number {
    for (let index = 0; index < probabilities.length; index += 1) {
      this.appendProbability(probabilities[index] ?? 0);
    }
    return this.getLatestFrame();
  }

  getLatestFrame(): number {
    return this.nextEntryIndex * this.hopFrames;
  }

  getBaseEntry(): number {
    return Math.max(0, this.nextEntryIndex - this.maxEntries);
  }

  getBaseFrame(): number {
    return this.getBaseEntry() * this.hopFrames;
  }

  findSilenceBoundary(searchEndFrame: number, minimumFrame: number, threshold: number): number {
    const cappedEndFrame = Math.min(searchEndFrame, this.getLatestFrame());
    const fromEntry =
      cappedEndFrame <= 0 ? 0 : Math.max(0, Math.ceil(cappedEndFrame / this.hopFrames) - 1);
    const minimumEntry = Math.floor(minimumFrame / this.hopFrames);
    const baseEntry = this.getBaseEntry();
    const clampedMinimum = Math.max(minimumEntry, baseEntry);

    for (let entryIndex = fromEntry; entryIndex >= clampedMinimum; entryIndex -= 1) {
      const probability = this.buffer[entryIndex % this.maxEntries] ?? 0;
      if (probability < threshold) {
        return entryIndex * this.hopFrames;
      }
    }

    return minimumFrame;
  }

  getSilenceTailDuration(threshold: number): number {
    if (this.nextEntryIndex === 0) {
      return 0;
    }

    let silentEntries = 0;
    const baseEntry = this.getBaseEntry();
    for (let entryIndex = this.nextEntryIndex - 1; entryIndex >= baseEntry; entryIndex -= 1) {
      const probability = this.buffer[entryIndex % this.maxEntries] ?? 0;
      if (probability >= threshold) {
        break;
      }
      silentEntries += 1;
    }

    return silentEntries * this.hopFrames;
  }

  hasSpeechInRange(startFrame: number, endFrame: number, threshold: number): boolean {
    if (endFrame <= startFrame) {
      return false;
    }

    const startEntry = Math.floor(startFrame / this.hopFrames);
    const endEntry = Math.ceil(endFrame / this.hopFrames);
    const baseEntry = this.getBaseEntry();
    const clampedStart = Math.max(startEntry, baseEntry);
    const clampedEnd = Math.min(endEntry, this.nextEntryIndex);

    for (let entryIndex = clampedStart; entryIndex < clampedEnd; entryIndex += 1) {
      const probability = this.buffer[entryIndex % this.maxEntries] ?? 0;
      if (probability >= threshold) {
        return true;
      }
    }

    return false;
  }

  findFirstSpeechFrame(
    startFrame: number,
    endFrame: number,
    threshold: number = this.speechThreshold,
    minSpeechHops = 4,
  ): number | null {
    if (endFrame <= startFrame || minSpeechHops <= 0) {
      return null;
    }

    const startEntry = Math.floor(startFrame / this.hopFrames);
    const endEntry = Math.ceil(endFrame / this.hopFrames);
    const baseEntry = this.getBaseEntry();
    const clampedStart = Math.max(startEntry, baseEntry);
    const clampedEnd = Math.min(endEntry, this.nextEntryIndex);

    let runStart: number | null = null;
    let runLength = 0;
    for (let entryIndex = clampedStart; entryIndex < clampedEnd; entryIndex += 1) {
      const probability = this.buffer[entryIndex % this.maxEntries] ?? 0;
      if (probability >= threshold) {
        if (runStart === null) {
          runStart = entryIndex * this.hopFrames;
          runLength = 1;
        } else {
          runLength += 1;
        }

        if (runLength >= minSpeechHops) {
          return runStart;
        }
      } else {
        runStart = null;
        runLength = 0;
      }
    }

    return null;
  }

  getWindowSummary(
    endFrame: number,
    windowMs: number,
    sampleRate = this.sampleRate,
  ): VoiceActivityProbabilityBufferWindowSummary {
    const windowFrames = Math.ceil((windowMs / 1000) * sampleRate);
    const startFrame = Math.max(0, endFrame - windowFrames);
    const startEntry = Math.floor(startFrame / this.hopFrames);
    const endEntry = Math.ceil(endFrame / this.hopFrames);
    const baseEntry = this.getBaseEntry();
    const clampedStart = Math.max(startEntry, baseEntry);
    const clampedEnd = Math.min(endEntry, this.nextEntryIndex);

    const recent: {
      readonly startFrame: number;
      readonly endFrame: number;
      readonly probability: number;
      readonly speaking: boolean;
    }[] = [];
    let speechHopCount = 0;
    let nonSpeechHopCount = 0;
    let maxConsecutiveSpeech = 0;
    let consecutiveSpeech = 0;
    let maxProbability = 0;

    for (let entryIndex = clampedStart; entryIndex < clampedEnd; entryIndex += 1) {
      const probability = this.buffer[entryIndex % this.maxEntries] ?? 0;
      const speaking = probability >= this.speechThreshold;
      if (speaking) {
        speechHopCount += 1;
        consecutiveSpeech += 1;
        maxConsecutiveSpeech = Math.max(maxConsecutiveSpeech, consecutiveSpeech);
      } else {
        nonSpeechHopCount += 1;
        consecutiveSpeech = 0;
      }
      maxProbability = Math.max(maxProbability, probability);
      recent.push({
        startFrame: entryIndex * this.hopFrames,
        endFrame: (entryIndex + 1) * this.hopFrames,
        probability,
        speaking,
      });
    }

    return {
      totalHops: recent.length,
      speechHopCount,
      nonSpeechHopCount,
      maxConsecutiveSpeech,
      maxProbability,
      recent,
    };
  }

  getTimeline(
    startFrame: number,
    endFrame: number,
    pointCount: number,
  ): readonly VoiceActivityProbabilityTimelinePoint[] {
    const safeStart = Math.max(0, Math.floor(startFrame));
    const safeEnd = Math.max(safeStart + 1, Math.floor(endFrame));
    const safePoints = Math.max(1, Math.floor(pointCount));
    const baseEntry = this.getBaseEntry();
    const endEntry = this.nextEntryIndex;
    const points: VoiceActivityProbabilityTimelinePoint[] = [];

    for (let index = 0; index < safePoints; index += 1) {
      const bucketStart =
        safeStart + Math.floor((index * (safeEnd - safeStart)) / safePoints);
      const bucketEnd =
        index === safePoints - 1
          ? safeEnd
          : safeStart + Math.floor(((index + 1) * (safeEnd - safeStart)) / safePoints);

      const startEntry = Math.max(baseEntry, Math.floor(bucketStart / this.hopFrames));
      const bucketEndEntry = Math.ceil(bucketEnd / this.hopFrames);
      const clampedEndEntry = Math.min(endEntry, bucketEndEntry);

      let totalWeight = 0;
      let weightedProbability = 0;
      let weightedSpeech = 0;

      for (let entryIndex = startEntry; entryIndex < clampedEndEntry; entryIndex += 1) {
        const entryStart = entryIndex * this.hopFrames;
        const entryEnd = entryStart + this.hopFrames;
        const overlapStart = Math.max(bucketStart, entryStart);
        const overlapEnd = Math.min(bucketEnd, entryEnd);
        const overlapFrames = overlapEnd - overlapStart;
        if (overlapFrames <= 0) {
          continue;
        }

        const probability = this.buffer[entryIndex % this.maxEntries] ?? 0;
        totalWeight += overlapFrames;
        weightedProbability += probability * overlapFrames;
        weightedSpeech += (probability >= this.speechThreshold ? 1 : 0) * overlapFrames;
      }

      const probability = totalWeight > 0 ? weightedProbability / totalWeight : 0;
      const speechRatio = totalWeight > 0 ? weightedSpeech / totalWeight : 0;
      points.push({
        startFrame: bucketStart,
        endFrame: Math.max(bucketStart + 1, bucketEnd),
        probability,
        speechRatio,
        speaking: speechRatio >= 0.5,
      });
    }

    return points;
  }

  hasRecentSpeech(
    endFrame: number,
    windowMs: number,
    sampleRate = this.sampleRate,
    minSpeechHops = 4,
    minSpeechRatio = 0.5,
  ): boolean {
    const summary = this.getWindowSummary(endFrame, windowMs, sampleRate);
    const speechRatio = summary.totalHops > 0 ? summary.speechHopCount / summary.totalHops : 0;
    return (
      summary.speechHopCount >= minSpeechHops &&
      summary.maxConsecutiveSpeech >= minSpeechHops &&
      summary.maxProbability >= this.speechThreshold &&
      speechRatio >= minSpeechRatio
    );
  }

  hasRecentSilence(
    endFrame: number,
    windowMs: number,
    sampleRate = this.sampleRate,
    minSilenceHops = 5,
  ): boolean {
    const summary = this.getWindowSummary(endFrame, windowMs, sampleRate);
    return summary.totalHops >= minSilenceHops && summary.nonSpeechHopCount >= minSilenceHops;
  }

  reset(): void {
    this.nextEntryIndex = 0;
    this.buffer.fill(0);
  }
}

function toSpeechFlag(
  observation: Pick<VoiceActivityObservation, 'speechProbability' | 'isSpeech'>,
  threshold: number,
): boolean {
  if (typeof observation.isSpeech === 'boolean') {
    return observation.isSpeech;
  }
  return observation.speechProbability >= threshold;
}

export class VoiceActivityTimeline implements StreamingActivityBuffer {
  readonly sampleRate: number;
  readonly maxFrames: number;
  readonly speechThreshold: number;
  private latestFrame = 0;
  private segments: VoiceActivitySegment[] = [];

  constructor(options: VoiceActivityTimelineOptions) {
    if (!Number.isFinite(options.sampleRate) || options.sampleRate <= 0) {
      throw new TypeError('VoiceActivityTimeline requires a positive sampleRate.');
    }
    if (!Number.isFinite(options.maxDurationSeconds) || options.maxDurationSeconds <= 0) {
      throw new TypeError('VoiceActivityTimeline requires a positive maxDurationSeconds.');
    }

    this.sampleRate = options.sampleRate;
    this.maxFrames = Math.max(1, Math.floor(options.sampleRate * options.maxDurationSeconds));
    this.speechThreshold = options.speechThreshold ?? 0.5;
  }

  appendObservation(observation: VoiceActivityObservation): VoiceActivitySegment {
    if (!Number.isFinite(observation.startFrame) || observation.startFrame < 0) {
      throw new RangeError(
        'VoiceActivityTimeline observation startFrame must be a non-negative finite number.',
      );
    }
    if (!Number.isFinite(observation.endFrame) || observation.endFrame <= observation.startFrame) {
      throw new RangeError(
        'VoiceActivityTimeline observation endFrame must be greater than startFrame.',
      );
    }

    const segment: VoiceActivitySegment = {
      startFrame: observation.startFrame,
      endFrame: observation.endFrame,
      speechProbability: observation.speechProbability,
      isSpeech: toSpeechFlag(observation, this.speechThreshold),
      startTimeSeconds: observation.startFrame / this.sampleRate,
      endTimeSeconds: observation.endFrame / this.sampleRate,
    };

    this.latestFrame = Math.max(this.latestFrame, segment.endFrame);
    this.segments.push(segment);
    this.prune();
    return segment;
  }

  appendChunk(
    startFrame: number,
    frameCount: number,
    speechProbability: number,
    isSpeech?: boolean,
  ): VoiceActivitySegment {
    if (!Number.isFinite(frameCount) || frameCount <= 0) {
      throw new RangeError('VoiceActivityTimeline frameCount must be a positive finite number.');
    }
    return this.appendObservation({
      startFrame,
      endFrame: startFrame + frameCount,
      speechProbability,
      isSpeech,
    });
  }

  appendVoiceActivityEvent(
    event: VoiceActivityEvent,
    defaultStartFrame: number,
    defaultEndFrame: number,
  ): VoiceActivitySegment {
    const startFrame = Number.isFinite(event.startTimeSeconds)
      ? Math.max(0, Math.round((event.startTimeSeconds ?? 0) * this.sampleRate))
      : defaultStartFrame;
    const endFrame = Number.isFinite(event.endTimeSeconds)
      ? Math.max(startFrame + 1, Math.round((event.endTimeSeconds ?? 0) * this.sampleRate))
      : defaultEndFrame;

    return this.appendObservation({
      startFrame,
      endFrame,
      speechProbability: event.speechProbability,
      isSpeech: event.isSpeech,
    });
  }

  getLatestFrame(): number {
    return this.latestFrame;
  }

  getBaseFrame(): number {
    return Math.max(0, this.latestFrame - this.maxFrames);
  }

  getSegments(): readonly VoiceActivitySegment[] {
    return this.segments;
  }

  findSilenceBoundary(searchEndFrame: number, minimumFrame: number, threshold: number): number {
    const cappedEnd = Math.min(searchEndFrame, this.latestFrame);
    let boundary = minimumFrame;

    for (const segment of this.segments) {
      if (segment.endFrame <= minimumFrame || segment.startFrame >= cappedEnd) {
        continue;
      }
      if (segment.speechProbability < threshold || !segment.isSpeech) {
        boundary = Math.max(boundary, Math.min(cappedEnd, segment.endFrame));
      }
    }

    return boundary;
  }

  getSilenceTailDuration(threshold: number): number {
    if (this.segments.length === 0) {
      return 0;
    }

    let tailStart = this.latestFrame;
    for (let index = this.segments.length - 1; index >= 0; index -= 1) {
      const segment = this.segments[index]!;
      if (segment.endFrame < tailStart) {
        break;
      }
      if (segment.speechProbability >= threshold && segment.isSpeech) {
        break;
      }
      tailStart = segment.startFrame;
    }

    return Math.max(0, this.latestFrame - tailStart);
  }

  hasSpeechInRange(startFrame: number, endFrame: number, threshold: number): boolean {
    return this.segments.some(
      (segment) =>
        segment.endFrame > startFrame &&
        segment.startFrame < endFrame &&
        segment.speechProbability >= threshold &&
        segment.isSpeech,
    );
  }

  createSnapshot(): VoiceActivityTimelineSnapshot {
    return {
      sampleRate: this.sampleRate,
      latestFrame: this.latestFrame,
      baseFrame: this.getBaseFrame(),
      trailingSilenceSeconds: this.getSilenceTailDuration(this.speechThreshold) / this.sampleRate,
      segments: this.segments,
    };
  }

  reset(): void {
    this.latestFrame = 0;
    this.segments = [];
  }

  private prune(): void {
    const baseFrame = this.getBaseFrame();
    this.segments = this.segments
      .filter((segment) => segment.endFrame > baseFrame)
      .map((segment) => {
        if (segment.startFrame >= baseFrame) {
          return segment;
        }
        return {
          ...segment,
          startFrame: baseFrame,
          startTimeSeconds: baseFrame / this.sampleRate,
        };
      });
  }
}
