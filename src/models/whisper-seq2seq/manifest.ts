import {
  parseWhisperGenerationConfig,
  parseWhisperModelConfig,
  type WhisperGenerationConfig,
  type WhisperModelConfig,
} from './generation-config.js';

export interface ParsedWhisperManifest {
  readonly generationConfig: WhisperGenerationConfig;
  readonly modelConfig: WhisperModelConfig;
  readonly format: string;
  readonly modelId: string;
}

export function parseWhisperManifest(raw: Record<string, unknown>): ParsedWhisperManifest {
  const format = typeof raw.format === 'string' ? raw.format : '';
  if (format !== 'whisper-browser-self-export-v1') {
    throw new Error(
      `Unsupported manifest format: "${format}". Expected "whisper-browser-self-export-v1".`,
    );
  }

  const modelId = typeof raw.model_id === 'string' ? raw.model_id : 'unknown';

  // Parse model config from manifest fields
  const modelConfig = parseWhisperModelConfig({
    decoder_layers: raw.decoder_layers,
    decoder_attention_heads: raw.decoder_attention_heads,
    d_model: raw.d_model,
    num_mel_bins: raw.num_mel_bins,
    median_filter_width: raw.median_filter_width,
  });

  // Handle head_dim from manifest (overrides computed value)
  if (typeof raw.head_dim === 'number') {
    (modelConfig as { headDim: number }).headDim = raw.head_dim;
  }

  // Parse generation config from manifest
  const specialTokens = (raw.special_tokens ?? {}) as Record<string, unknown>;
  const generationConfig = parseWhisperGenerationConfig({
    alignment_heads: raw.alignment_heads,
    no_timestamps_token_id: specialTokens.no_timestamps_token_id,
    begin_suppress_tokens: specialTokens.begin_suppress_tokens,
    max_length: raw.max_target_positions,
  });

  return { generationConfig, modelConfig, format, modelId };
}
