import {
  chunkPcmAudio,
  normalizePcmInput,
  type AudioChunk,
  type PcmAudioBuffer,
} from '../audio/index.js';
import type { AudioInputLike } from '../types/index.js';

export interface AudioChunkerOptions {
  readonly chunkLengthMs?: number;
  readonly overlapMs?: number;
}

export interface LayeredAudioBufferOptions {
  readonly maxWindowMs?: number;
  readonly overlapMs?: number;
}

export interface LayeredAudioBufferEntry<
  TLayers extends Record<string, unknown> = Record<string, unknown>,
> {
  readonly chunk: AudioChunk;
  readonly layers: Partial<TLayers>;
}

export class AudioChunker {
  private readonly chunkLengthMs: number;
  private readonly overlapMs: number;

  constructor(options: AudioChunkerOptions = {}) {
    this.chunkLengthMs = options.chunkLengthMs ?? 30000;
    this.overlapMs = options.overlapMs ?? 2000;
  }

  split(input: AudioInputLike, startTimeSeconds = 0): AudioChunk[] {
    const audio = normalizePcmInput(input);
    const chunkLengthFrames = Math.max(
      1,
      Math.round((this.chunkLengthMs / 1000) * audio.sampleRate),
    );
    const overlapFrames = Math.max(0, Math.round((this.overlapMs / 1000) * audio.sampleRate));
    const chunks = chunkPcmAudio(audio, chunkLengthFrames, overlapFrames);

    return chunks.map((chunk, index) =>
      chunk.toChunk({
        sequence: index,
        startTimeSeconds:
          startTimeSeconds + (index * (chunkLengthFrames - overlapFrames)) / audio.sampleRate,
        isLast: index === chunks.length - 1,
      }),
    );
  }
}

export class LayeredAudioBuffer<TLayers extends Record<string, unknown> = Record<string, unknown>> {
  private readonly maxWindowSeconds: number;
  private readonly overlapSeconds: number;
  private readonly entries: Array<LayeredAudioBufferEntry<TLayers>> = [];
  private sequence = 0;

  constructor(options: LayeredAudioBufferOptions = {}) {
    this.maxWindowSeconds = (options.maxWindowMs ?? 30000) / 1000;
    this.overlapSeconds = (options.overlapMs ?? 2000) / 1000;
  }

  push(input: AudioInputLike, startTimeSeconds?: number): LayeredAudioBufferEntry<TLayers> {
    const chunk = normalizePcmInput(input).toChunk({
      sequence: this.sequence,
      startTimeSeconds,
    });
    const entry: LayeredAudioBufferEntry<TLayers> = {
      chunk,
      layers: {},
    };

    this.sequence += 1;
    this.entries.push(entry);
    this.trim();
    return entry;
  }

  setLayer<TKey extends keyof TLayers>(sequence: number, key: TKey, value: TLayers[TKey]): void {
    const entry = this.entries.find((candidate) => candidate.chunk.sequence === sequence);
    if (!entry) {
      throw new Error(`LayeredAudioBuffer could not find chunk with sequence ${sequence}.`);
    }

    Object.assign(entry.layers, { [key]: value });
  }

  getLayer<TKey extends keyof TLayers>(sequence: number, key: TKey): TLayers[TKey] | undefined {
    const entry = this.entries.find((candidate) => candidate.chunk.sequence === sequence);
    return entry?.layers[key];
  }

  getEntries(): readonly LayeredAudioBufferEntry<TLayers>[] {
    return this.entries;
  }

  getBufferedDurationSeconds(): number {
    return this.entries.reduce((sum, entry) => sum + entry.chunk.durationSeconds, 0);
  }

  toPcmAudioBuffer(): PcmAudioBuffer {
    if (this.entries.length === 0) {
      return normalizePcmInput(new Float32Array(0));
    }

    const sampleRate = this.entries[0]!.chunk.sampleRate;
    const numberOfChannels = this.entries[0]!.chunk.numberOfChannels;
    const totalFrames = this.entries.reduce((sum, entry) => sum + entry.chunk.numberOfFrames, 0);
    const channels = Array.from({ length: numberOfChannels }, () => new Float32Array(totalFrames));

    let cursor = 0;
    for (const entry of this.entries) {
      for (let channelIndex = 0; channelIndex < numberOfChannels; channelIndex += 1) {
        channels[channelIndex]!.set(entry.chunk.channels[channelIndex]!, cursor);
      }
      cursor += entry.chunk.numberOfFrames;
    }

    return normalizePcmInput({
      sampleRate,
      numberOfChannels,
      numberOfFrames: totalFrames,
      durationSeconds: totalFrames / sampleRate,
      channels,
    });
  }

  reset(): void {
    this.entries.length = 0;
    this.sequence = 0;
  }

  private trim(): void {
    while (this.getBufferedDurationSeconds() > this.maxWindowSeconds && this.entries.length > 1) {
      const nextDuration =
        this.getBufferedDurationSeconds() - this.entries[0]!.chunk.durationSeconds;
      if (nextDuration < this.overlapSeconds) {
        break;
      }

      this.entries.shift();
    }
  }
}
