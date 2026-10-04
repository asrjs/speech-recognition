/**
 * Local artifact URLs for the Moonshine encoder, cross-attention projection and
 * decoder. There is no seed cache: cross-attention memory is computed in-graph,
 * so utterance length is not pinned to an exported artifact.
 */
export interface MoonshineLoadOptions {
  readonly encoderUrl: string;
  /** Projects encoder states into per-layer cross-attention key and value. */
  readonly crossKvUrl: string;
  readonly decoderUrl: string;
  readonly tokenizerUrl: string;
  readonly generationUrl?: string;
  /** Supplies past-state geometry, which differs across checkpoints. */
  readonly receiptUrl?: string;
  readonly encoderBackend?: 'wasm' | 'webgpu';
  readonly maxTokens?: number;
  readonly sampleRate?: number;
}
