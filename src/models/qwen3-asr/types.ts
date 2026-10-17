/**
 * Local artifact URLs for the Qwen3-ASR fp32 graph set.
 *
 * The decode path is the audio front, then a cache-carrying decoder step, then
 * the token embedding table. Past state crosses the boundary as two stacked
 * tensors shaped [layers, batch, kv_heads, past_length, head_dim], and the step
 * must be traced against a non-empty past or its attention mask specialises to a
 * single position.
 */
export interface Qwen3AsrLoadOptions {
  readonly frontUrl: string;
  readonly decoderUrl: string;
  readonly embedUrl: string;
  readonly tokenizerUrl: string;
  readonly cuSeqlensUrl?: string;
  readonly inputFeaturesUrl?: string;
  readonly prefixEmbedsUrl?: string;
  readonly suffixEmbedsUrl?: string;
  readonly sampleRate?: number;
  readonly maxTokens?: number;
}
