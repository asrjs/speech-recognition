import { AudioChunk, normalizePcmChunk, PcmAudioBuffer } from '../../processors/index.js';
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

  constructor(options: RollingAudioWindowOptions = {}) {
    this.maxWindowSeconds = (options.maxWindowMs ?? 30000) / 1000;
    this.overlapSeconds = (options.overlapMs ?? 2000) / 1000;
  }

  push(input: AudioInputLike, startTimeSeconds?: number): AudioChunk {
    const chunk = normalizePcmChunk(input, {
      sequence: this.sequence,
      startTimeSeconds
    });

    this.sequence += 1;
    this.chunks.push(chunk);
    this.trim();
    return chunk;
  }

  reset(): void {
    this.chunks.length = 0;
    this.sequence = 0;
  }

  getBufferedDurationSeconds(): number {
    return this.chunks.reduce((sum, chunk) => sum + chunk.durationSeconds, 0);
  }

  toPcmAudioBuffer(): PcmAudioBuffer {
    if (this.chunks.length === 0) {
      return new PcmAudioBuffer({
        sampleRate: 16000,
        channels: [new Float32Array(0)]
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
      channels
    });
  }

  private trim(): void {
    while (this.getBufferedDurationSeconds() > this.maxWindowSeconds && this.chunks.length > 1) {
      const nextDuration = this.getBufferedDurationSeconds() - this.chunks[0]!.durationSeconds;
      if (nextDuration < this.overlapSeconds) {
        break;
      }
      this.chunks.shift();
    }
  }
}
