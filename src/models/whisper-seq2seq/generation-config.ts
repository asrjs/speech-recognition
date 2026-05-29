export interface WhisperGenerationConfig {
  readonly alignmentHeads: readonly { readonly layer: number; readonly head: number }[];
  readonly noTimestampsTokenId?: number;
  readonly suppressTokens?: readonly number[];
  readonly beginSuppressTokens?: readonly number[];
  readonly maxLength?: number;
}

export interface WhisperModelConfig {
  readonly medianFilterWidth: number;
  readonly decoderLayers: number;
  readonly decoderAttentionHeads: number;
  readonly dModel: number;
  readonly headDim: number;
  readonly numMelBins?: number;
}

export function parseWhisperGenerationConfig(
  raw: Record<string, unknown>,
): WhisperGenerationConfig {
  const alignmentHeadsRaw = Array.isArray(raw.alignment_heads) ? raw.alignment_heads : [];
  const alignmentHeads: { layer: number; head: number }[] = [];
  for (const entry of alignmentHeadsRaw) {
    if (
      Array.isArray(entry) &&
      entry.length === 2 &&
      typeof entry[0] === 'number' &&
      typeof entry[1] === 'number'
    ) {
      alignmentHeads.push({ layer: entry[0], head: entry[1] });
    }
  }

  return {
    alignmentHeads,
    noTimestampsTokenId:
      typeof raw.no_timestamps_token_id === 'number'
        ? raw.no_timestamps_token_id
        : undefined,
    suppressTokens: Array.isArray(raw.suppress_tokens)
      ? raw.suppress_tokens.filter((v): v is number => typeof v === 'number')
      : undefined,
    beginSuppressTokens: Array.isArray(raw.begin_suppress_tokens)
      ? raw.begin_suppress_tokens.filter((v): v is number => typeof v === 'number')
      : undefined,
    maxLength: typeof raw.max_length === 'number' ? raw.max_length : undefined,
  };
}

export function parseWhisperModelConfig(
  raw: Record<string, unknown>,
): WhisperModelConfig {
  const decoderLayers: number =
    typeof raw.decoder_layers === 'number' ? raw.decoder_layers : 4;
  const decoderAttentionHeads: number =
    typeof raw.decoder_attention_heads === 'number' ? raw.decoder_attention_heads : 6;
  const dModel: number =
    typeof raw.d_model === 'number' ? raw.d_model : 384;

  if (dModel % decoderAttentionHeads !== 0) {
    throw new Error(
      `d_model (${dModel}) is not divisible by decoder_attention_heads (${decoderAttentionHeads}). Whisper requires head_dim to be integral.`,
    );
  }

  return {
    medianFilterWidth:
      typeof raw.median_filter_width === 'number' ? raw.median_filter_width : 7,
    decoderLayers,
    decoderAttentionHeads,
    dModel,
    headDim: Math.floor(dModel / decoderAttentionHeads),
    numMelBins:
      typeof raw.num_mel_bins === 'number' ? raw.num_mel_bins : undefined,
  };
}
