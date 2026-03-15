export type AudioSampleFormat = 'f32-planar' | 'f32-interleaved' | 'i16-interleaved';

export interface AudioBufferLike {
  readonly sampleRate: number;
  readonly numberOfChannels: number;
  readonly numberOfFrames: number;
  readonly durationSeconds: number;
  readonly channels?: readonly Float32Array[];
  readonly data?: Float32Array | Float64Array | Int16Array;
  readonly format?: AudioSampleFormat;
}

export interface AudioChunkLike extends AudioBufferLike {
  readonly sequence?: number;
  readonly startTimeSeconds?: number;
  readonly endTimeSeconds?: number;
  readonly isLast?: boolean;
}

export type AudioInputLike = Float32Array | Float64Array | AudioBufferLike;
