export type RawAudio = Float32Array | Float64Array; // 16kHz Mono

export interface TensorMap {
  readonly data: Float32Array | Int32Array | BigInt64Array;
  readonly dims: readonly number[];
  readonly type: 'float32' | 'int32' | 'int64';
}

export interface AcousticFeatures {
  readonly tensor: TensorMap; // e.g. [B, n_features, T_frames]
  readonly sequenceLengths?: Int32Array | number[]; // Valid length per batch item
}

export interface EncodedSequence {
  readonly tensor: TensorMap; // e.g. [B, T_enc, d_model]
  readonly sequenceLengths?: Int32Array | number[];
}

export interface TopologyLogits {
  readonly tensor: TensorMap; // e.g. [B, T_dec, V+1]
  readonly sequenceLengths?: Int32Array | number[];
}
