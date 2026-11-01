import { AudioChunk, normalizePcmChunk, PcmAudioBuffer } from '../../audio/index.js';
import type { AudioInputLike } from '../../types/index.js';

export interface RollingAudioWindowOptions {
  readonly maxWindowMs?: number;
  readonly overlapMs?: number;
}

export class RollingAudioWindow {
  private readonly maxWindowSeconds: number;
  private readonly overlapSeconds: number;
  private readonly chunks: AudioChunk[] = [];
  private sequence = 0;
  private windowBaseSeconds = 0;
  private windowBaseSample = 0;

  constructor(options: RollingAudioWindowOptions = {}) {
    this.maxWindowSeconds = (options.maxWindowMs ?? 30000) / 1000;
    this.overlapSeconds = (options.overlapMs ?? 2000) / 1000;
  }

  getWindowBaseSeconds(): number {
    return this.windowBaseSeconds;
  }

  getWindowBaseSample(): number {
    return this.windowBaseSample;
  }

  getMaxWindowSeconds(): number {
    return this.maxWindowSeconds;
  }

  getOverlapSeconds(): number {
    return this.overlapSeconds;
  }

  wouldEvict(additionalSeconds = 0): boolean {
    return (
      this.getBufferedDurationSeconds() + additionalSeconds > this.maxWindowSeconds &&
      this.chunks.length > 0
    );
  }

  getEvictionDuration(additionalSeconds = 0): number {
    let projectedDuration = this.getBufferedDurationSeconds() + additionalSeconds;
    let evictedSeconds = 0;
    for (const chunk of this.chunks) {
      if (
        projectedDuration <= this.maxWindowSeconds ||
        projectedDuration - chunk.durationSeconds < this.overlapSeconds
      ) {
        break;
      }
      evictedSeconds += chunk.durationSeconds;
      projectedDuration -= chunk.durationSeconds;
    }
    return evictedSeconds;
  }

  push(
    input: AudioInputLike,
    startTimeSeconds?: number,
    options: { autoTrim?: boolean } = {},
  ): AudioChunk {
    const chunk = normalizePcmChunk(input, {
      sequence: this.sequence,
      startTimeSeconds,
    });

    this.sequence += 1;
    this.chunks.push(chunk);
    if (options.autoTrim !== false) {
      this.trim();
    }
    return chunk;
  }

  reset(): void {
    this.chunks.length = 0;
    this.sequence = 0;
    this.windowBaseSeconds = 0;
    this.windowBaseSample = 0;
  }

  getBufferedDurationSeconds(): number {
    return this.chunks.reduce((sum, chunk) => sum + chunk.durationSeconds, 0);
  }

  toPcmAudioBuffer(): PcmAudioBuffer {
    if (this.chunks.length === 0) {
      return new PcmAudioBuffer({
        sampleRate: 16000,
        channels: [new Float32Array(0)],
      });
    }

    const sampleRate = this.chunks[0]!.sampleRate;
    const numberOfChannels = this.chunks[0]!.numberOfChannels;
    const totalFrames = this.chunks.reduce((sum, chunk) => sum + chunk.numberOfFrames, 0);
    const channels = Array.from({ length: numberOfChannels }, () => new Float32Array(totalFrames));

    let cursor = 0;
    for (const chunk of this.chunks) {
      for (let channelIndex = 0; channelIndex < numberOfChannels; channelIndex += 1) {
        channels[channelIndex]!.set(chunk.channels[channelIndex]!, cursor);
      }
      cursor += chunk.numberOfFrames;
    }

    return new PcmAudioBuffer({
      sampleRate,
      channels,
    });
  }

  trim(): { evictedDurationSeconds: number; evictedFrames: number } {
    let evictedDurationSeconds = 0;
    let evictedFrames = 0;
    while (this.getBufferedDurationSeconds() > this.maxWindowSeconds && this.chunks.length > 1) {
      const nextDuration = this.getBufferedDurationSeconds() - this.chunks[0]!.durationSeconds;
      if (nextDuration < this.overlapSeconds) {
        break;
      }
      const shifted = this.chunks.shift()!;
      evictedDurationSeconds += shifted.durationSeconds;
      evictedFrames += shifted.numberOfFrames;
      this.windowBaseSeconds += shifted.durationSeconds;
      this.windowBaseSample += shifted.numberOfFrames;
    }
    return { evictedDurationSeconds, evictedFrames };
  }
}
