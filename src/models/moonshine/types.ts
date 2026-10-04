/** Local artifact URLs for the Moonshine encoder and decoder pair. */
export interface MoonshineLoadOptions {
  readonly encoderUrl: string;
  readonly decoderUrl: string;
  readonly tokenizerUrl: string;
  readonly generationUrl?: string;
  readonly seedUrl: string;
  readonly encoderBackend?: 'wasm' | 'webgpu';
  readonly maxTokens?: number;
  readonly sampleRate?: number;
}
