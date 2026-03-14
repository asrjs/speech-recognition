import type { AudioBufferLike, AudioChunkLike, AudioInputLike, TranscriptWarning } from '../types/index.js';

export interface PcmAudioBufferInit {
  readonly sampleRate: number;
  readonly channels: readonly Float32Array[];
}

function assertConsistentChannelLengths(channels: readonly Float32Array[]): number {
  if (channels.length === 0) {
    throw new TypeError('PcmAudioBuffer requires at least one channel.');
  }

  const frameCount = channels[0]!.length;
  for (const channel of channels) {
    if (channel.length !== frameCount) {
      throw new TypeError('PcmAudioBuffer channels must all have the same frame length.');
    }
  }

  return frameCount;
}

export class PcmAudioBuffer implements AudioBufferLike {
  readonly sampleRate: number;
  readonly channels: readonly Float32Array[];
  readonly numberOfChannels: number;
  readonly numberOfFrames: number;
  readonly durationSeconds: number;
  readonly format = 'f32-planar' as const;

  constructor(init: PcmAudioBufferInit) {
    if (!Number.isFinite(init.sampleRate) || init.sampleRate <= 0) {
      throw new TypeError('PcmAudioBuffer requires a positive finite sampleRate.');
    }

    this.sampleRate = init.sampleRate;
    this.channels = init.channels.map((channel) => new Float32Array(channel));
    this.numberOfChannels = this.channels.length;
    this.numberOfFrames = assertConsistentChannelLengths(this.channels);
    this.durationSeconds = this.numberOfFrames / this.sampleRate;
  }

  static fromMono(data: Float32Array | Float64Array, sampleRate: number): PcmAudioBuffer {
    return new PcmAudioBuffer({
      sampleRate,
      channels: [Float32Array.from(data)]
    });
  }

  toMono(): PcmAudioBuffer {
    if (this.numberOfChannels === 1) {
      return this;
    }

    const mono = new Float32Array(this.numberOfFrames);
    for (let channelIndex = 0; channelIndex < this.numberOfChannels; channelIndex += 1) {
      const channel = this.channels[channelIndex]!;
      for (let frameIndex = 0; frameIndex < this.numberOfFrames; frameIndex += 1) {
        mono[frameIndex] = (mono[frameIndex] ?? 0) + ((channel[frameIndex] ?? 0) / this.numberOfChannels);
      }
    }

    return new PcmAudioBuffer({
      sampleRate: this.sampleRate,
      channels: [mono]
    });
  }

  sliceFrames(startFrame: number, endFrame: number): PcmAudioBuffer {
    const clampedStart = Math.max(0, Math.min(this.numberOfFrames, Math.floor(startFrame)));
    const clampedEnd = Math.max(clampedStart, Math.min(this.numberOfFrames, Math.ceil(endFrame)));

    return new PcmAudioBuffer({
      sampleRate: this.sampleRate,
      channels: this.channels.map((channel) => channel.subarray(clampedStart, clampedEnd))
    });
  }

  toChunk(
    options: Omit<Partial<AudioChunkInit>, 'sampleRate' | 'channels'> = {}
  ): AudioChunk {
    return new AudioChunk({
      sampleRate: this.sampleRate,
      channels: this.channels,
      sequence: options.sequence,
      startTimeSeconds: options.startTimeSeconds,
      endTimeSeconds: options.endTimeSeconds,
      isLast: options.isLast
    });
  }
}

export interface AudioChunkInit extends PcmAudioBufferInit {
  readonly sequence?: number;
  readonly startTimeSeconds?: number;
  readonly endTimeSeconds?: number;
  readonly isLast?: boolean;
}

export class AudioChunk extends PcmAudioBuffer implements AudioChunkLike {
  readonly sequence?: number;
  readonly startTimeSeconds?: number;
  readonly endTimeSeconds?: number;
  readonly isLast?: boolean;

  constructor(init: AudioChunkInit) {
    super(init);
    this.sequence = init.sequence;
    this.startTimeSeconds = init.startTimeSeconds;
    this.endTimeSeconds = init.endTimeSeconds ?? (
      init.startTimeSeconds !== undefined ? init.startTimeSeconds + this.durationSeconds : undefined
    );
    this.isLast = init.isLast;
  }
}

export interface NormalizePcmOptions {
  readonly sampleRate?: number;
  readonly sequence?: number;
  readonly startTimeSeconds?: number;
  readonly isLast?: boolean;
}

function splitInterleavedFloatData(
  data: Float32Array | Float64Array,
  numberOfChannels: number
): Float32Array[] {
  const frames = Math.floor(data.length / numberOfChannels);
  const channels = Array.from({ length: numberOfChannels }, () => new Float32Array(frames));

  for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
    for (let channelIndex = 0; channelIndex < numberOfChannels; channelIndex += 1) {
      channels[channelIndex]![frameIndex] = data[(frameIndex * numberOfChannels) + channelIndex] ?? 0;
    }
  }

  return channels;
}

function splitInterleavedInt16Data(data: Int16Array, numberOfChannels: number): Float32Array[] {
  const frames = Math.floor(data.length / numberOfChannels);
  const channels = Array.from({ length: numberOfChannels }, () => new Float32Array(frames));

  for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
    for (let channelIndex = 0; channelIndex < numberOfChannels; channelIndex += 1) {
      channels[channelIndex]![frameIndex] = (data[(frameIndex * numberOfChannels) + channelIndex] ?? 0) / 32768;
    }
  }

  return channels;
}

export function normalizePcmInput(input: AudioInputLike, options: NormalizePcmOptions = {}): PcmAudioBuffer {
  if (input instanceof Float32Array || input instanceof Float64Array) {
    return PcmAudioBuffer.fromMono(input, options.sampleRate ?? 16000);
  }

  if (input.channels && input.channels.length > 0) {
    return new PcmAudioBuffer({
      sampleRate: input.sampleRate,
      channels: input.channels.map((channel) => new Float32Array(channel))
    });
  }

  if (input.data instanceof Int16Array) {
    return new PcmAudioBuffer({
      sampleRate: input.sampleRate,
      channels: splitInterleavedInt16Data(input.data, input.numberOfChannels)
    });
  }

  if (input.data instanceof Float32Array || input.data instanceof Float64Array) {
    return new PcmAudioBuffer({
      sampleRate: input.sampleRate,
      channels: splitInterleavedFloatData(input.data, input.numberOfChannels)
    });
  }

  throw new TypeError('normalizePcmInput expected mono PCM or an AudioBufferLike with channel data.');
}

export function normalizePcmChunk(input: AudioInputLike, options: NormalizePcmOptions = {}): AudioChunk {
  const audio = normalizePcmInput(input, options);
  return new AudioChunk({
    sampleRate: audio.sampleRate,
    channels: audio.channels,
    sequence: options.sequence,
    startTimeSeconds: options.startTimeSeconds,
    isLast: options.isLast
  });
}

export function downmixToMono(input: AudioInputLike, options: NormalizePcmOptions = {}): PcmAudioBuffer {
  return normalizePcmInput(input, options).toMono();
}

export interface SampleRatePolicyResult {
  readonly audio: PcmAudioBuffer;
  readonly warning?: TranscriptWarning;
}

export interface SampleRatePolicy {
  readonly name: string;
  ensure(audio: PcmAudioBuffer, targetSampleRate: number): SampleRatePolicyResult;
}

export function createPassthroughSampleRatePolicy(): SampleRatePolicy {
  return {
    name: 'passthrough',
    ensure(audio, targetSampleRate) {
      if (audio.sampleRate === targetSampleRate) {
        return { audio };
      }

      return {
        audio,
        warning: {
          code: 'audio.sample-rate-passthrough',
          message: `Received ${audio.sampleRate} Hz audio while ${targetSampleRate} Hz was requested. The passthrough policy does not resample.`,
          recoverable: true
        }
      };
    }
  };
}

export function framePcmAudio(
  input: AudioBufferLike,
  frameLengthFrames: number,
  hopLengthFrames: number
): AudioChunk[] {
  if (frameLengthFrames <= 0 || hopLengthFrames <= 0) {
    throw new RangeError('framePcmAudio expects positive frameLengthFrames and hopLengthFrames.');
  }

  const audio = normalizePcmInput(input);
  const frames: AudioChunk[] = [];

  for (let start = 0, sequence = 0; start < audio.numberOfFrames; start += hopLengthFrames, sequence += 1) {
    const end = Math.min(audio.numberOfFrames, start + frameLengthFrames);
    const slice = audio.sliceFrames(start, end);
    frames.push(slice.toChunk({
      sequence,
      startTimeSeconds: start / audio.sampleRate,
      endTimeSeconds: end / audio.sampleRate,
      isLast: end >= audio.numberOfFrames
    }));

    if (end >= audio.numberOfFrames) {
      break;
    }
  }

  return frames;
}

export function chunkPcmAudio(
  input: AudioBufferLike,
  chunkLengthFrames: number,
  overlapFrames = 0
): AudioChunk[] {
  if (chunkLengthFrames <= 0) {
    throw new RangeError('chunkPcmAudio expects a positive chunkLengthFrames.');
  }

  if (overlapFrames < 0 || overlapFrames >= chunkLengthFrames) {
    throw new RangeError('chunkPcmAudio expects overlapFrames to be >= 0 and smaller than chunkLengthFrames.');
  }

  const hopLengthFrames = chunkLengthFrames - overlapFrames;
  return framePcmAudio(input, chunkLengthFrames, hopLengthFrames);
}
