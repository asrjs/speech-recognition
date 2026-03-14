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

function toSpeechFlag(
  observation: Pick<VoiceActivityObservation, 'speechProbability' | 'isSpeech'>,
  threshold: number
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
      throw new RangeError('VoiceActivityTimeline observation startFrame must be a non-negative finite number.');
    }
    if (!Number.isFinite(observation.endFrame) || observation.endFrame <= observation.startFrame) {
      throw new RangeError('VoiceActivityTimeline observation endFrame must be greater than startFrame.');
    }

    const segment: VoiceActivitySegment = {
      startFrame: observation.startFrame,
      endFrame: observation.endFrame,
      speechProbability: observation.speechProbability,
      isSpeech: toSpeechFlag(observation, this.speechThreshold),
      startTimeSeconds: observation.startFrame / this.sampleRate,
      endTimeSeconds: observation.endFrame / this.sampleRate
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
    isSpeech?: boolean
  ): VoiceActivitySegment {
    if (!Number.isFinite(frameCount) || frameCount <= 0) {
      throw new RangeError('VoiceActivityTimeline frameCount must be a positive finite number.');
    }
    return this.appendObservation({
      startFrame,
      endFrame: startFrame + frameCount,
      speechProbability,
      isSpeech
    });
  }

  appendVoiceActivityEvent(
    event: VoiceActivityEvent,
    defaultStartFrame: number,
    defaultEndFrame: number
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
      isSpeech: event.isSpeech
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
    return this.segments.some((segment) =>
      segment.endFrame > startFrame
      && segment.startFrame < endFrame
      && segment.speechProbability >= threshold
      && segment.isSpeech
    );
  }

  createSnapshot(): VoiceActivityTimelineSnapshot {
    return {
      sampleRate: this.sampleRate,
      latestFrame: this.latestFrame,
      baseFrame: this.getBaseFrame(),
      trailingSilenceSeconds: this.getSilenceTailDuration(this.speechThreshold) / this.sampleRate,
      segments: this.segments
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
          startTimeSeconds: baseFrame / this.sampleRate
        };
      });
  }
}
