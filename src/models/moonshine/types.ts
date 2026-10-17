/**
 * Local artifact URLs for the Moonshine encoder, cross-attention projection and
 * decoder. There is no seed cache: cross-attention memory is computed in-graph,
 * so utterance length is not pinned to an exported artifact.
 */
/** URLs of the streaming graph set (chunked encoder, final-flush encoder, decoder pair). */
export interface MoonshineStreamingAssets {
  /** Pure-function chunked encoder graph URL. */
  readonly chunkedEncoderUrl: string;
  readonly chunkedEncoderDataUrl?: string;
  /** Whole-utterance encoder used for the final flush (dynamic time dim). */
  readonly encoderUrl: string;
  readonly encoderDataUrl?: string;
  readonly prefillUrl: string;
  readonly prefillDataUrl?: string;
  readonly stepUrl: string;
  readonly stepDataUrl?: string;
  readonly tokenizerUrl: string;
}

export interface MoonshineLoadOptions {
  readonly encoderUrl: string;
  readonly encoderDataUrl?: string;
  /** Projects encoder states into per-layer cross-attention key and value. */
  readonly crossKvUrl: string;
  readonly crossKvDataUrl?: string;
  readonly decoderUrl: string;
  readonly decoderDataUrl?: string;
  readonly tokenizerUrl: string;
  readonly generationUrl?: string;
  /** Supplies past-state geometry, which differs across checkpoints. */
  readonly receiptUrl?: string;
  readonly encoderBackend?: 'wasm' | 'webgpu';
  readonly maxTokens?: number;
  readonly sampleRate?: number;
  /** Present when the artifact source also carries the streaming graph set. */
  readonly streaming?: MoonshineStreamingAssets;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
}
