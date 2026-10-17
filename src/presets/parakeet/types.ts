/** Legacy metrics shape preserved for compatibility with earlier Parakeet.js consumers. */
export interface LegacyParakeetMetrics {
  readonly preprocess_ms?: number;
  readonly encode_ms?: number;
  readonly decode_ms?: number;
  readonly tokenize_ms?: number;
  readonly total_ms?: number;
  readonly wall_ms?: number;
  readonly audio_duration_sec?: number;
  readonly rtf?: number;
  readonly rtfx?: number;
  readonly preprocessor_backend_requested?: string;
  readonly preprocessor_backend?: string;
  readonly audio_decode_ms?: number;
  readonly downmix_ms?: number;
  readonly resample_ms?: number;
  readonly audio_preparation_ms?: number;
  readonly input_sample_rate?: number;
  readonly output_sample_rate?: number;
  readonly resampler?: string;
  readonly resampler_quality?: string | null;
  readonly encoder_frame_count?: number;
  readonly decode_iterations?: number;
  readonly emitted_token_count?: number;
  readonly emitted_word_count?: number;
}

/** Legacy JSON transcript shape returned by the Parakeet compatibility wrapper. */
export interface LegacyParakeetTranscript {
  readonly utterance_text: string;
  readonly words: ReadonlyArray<{
    readonly text: string;
    readonly start_time: number;
    readonly end_time: number;
    readonly confidence?: number;
  }>;
  readonly tokens?: ReadonlyArray<{
    readonly id?: number;
    readonly token: string;
    readonly raw_text?: string;
    readonly start_time: number;
    readonly end_time: number;
    readonly confidence?: number;
    readonly frame_index?: number;
    readonly log_prob?: number;
    readonly tdt_step?: number;
  }>;
  readonly confidence_scores?: {
    readonly utterance?: number | null;
    readonly word_avg?: number | null;
    readonly token_avg?: number | null;
    readonly frame_avg?: number | null;
    readonly overall_log_prob?: number | null;
    readonly frame?: readonly number[] | null;
  };
  readonly metrics?: LegacyParakeetMetrics;
  readonly is_final: boolean;
}

/** Legacy transcription options accepted by the Parakeet compatibility wrapper. */
export interface LegacyParakeetTranscribeOptions {
  readonly returnTimestamps?: boolean;
  readonly returnConfidences?: boolean;
  readonly returnTokenIds?: boolean;
  readonly returnFrameIndices?: boolean;
  readonly returnLogProbs?: boolean;
  readonly returnTdtSteps?: boolean;
  readonly returnDecoderState?: boolean;
  readonly frameStride?: number;
  readonly enableProfiling?: boolean;
}
