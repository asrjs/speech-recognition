/**
 * Generic CTC (Connectionist Temporal Classification) types.
 *
 * Shared by all CTC-based models: MedASR, WAV2VEC2, and future architectures.
 * Model-specific type aliases (LasrCtcTokenSpan, Wav2Vec2TokenSpan, etc.)
 * should be re-exported as references to these shared types.
 *
 * @module ctc/types
 */

// ---------------------------------------------------------------------------
// Tokenizer contract — minimal interface a CTC tokenizer must satisfy
// ---------------------------------------------------------------------------

export interface CtcTokenizerLike {
  /** Decode a sequence of token IDs to text. */
  decode(ids: readonly number[]): string;
  /** Decode a single token ID to its string representation. */
  decodeTokenPiece?(tokenId: number): string;
}

// ---------------------------------------------------------------------------
// Raw token span (internal, before timing is applied)
// ---------------------------------------------------------------------------

export interface CtcRawTokenSpan {
  readonly tokenId: number;
  readonly startFrame: number;
  readonly endFrame: number;
  readonly frameCount: number;
  readonly averageLogProb: number;
  readonly confidence: number;
}

// ---------------------------------------------------------------------------
// Timed token span (after frame-to-seconds conversion)
// ---------------------------------------------------------------------------

export interface CtcTokenSpan {
  readonly tokenId: number;
  readonly text: string;
  readonly startFrame: number;
  readonly endFrame: number;
  readonly frameCount: number;
  readonly startTime: number;
  readonly endTime: number;
  readonly duration: number;
  readonly confidence: number;
  readonly averageLogProb: number;
}

// ---------------------------------------------------------------------------
// Utterance-level timing
// ---------------------------------------------------------------------------

export interface CtcUtteranceTiming {
  readonly hasSpeech: boolean;
  readonly startFrame: number | null;
  readonly endFrame: number | null;
  readonly startTime: number;
  readonly endTime: number;
  readonly duration: number;
  readonly confidence: number;
}

// ---------------------------------------------------------------------------
// Sentence-level timing
// ---------------------------------------------------------------------------

export interface CtcSentenceTiming {
  readonly text: string;
  readonly startTokenIndex: number;
  readonly endTokenIndex: number;
  readonly startFrame: number;
  readonly endFrame: number;
  readonly startTime: number;
  readonly endTime: number;
  readonly duration: number;
  readonly confidence: number;
}

// ---------------------------------------------------------------------------
// Native word (built from token spans)
// ---------------------------------------------------------------------------

export interface CtcNativeWord {
  readonly index: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
  readonly confidence?: number;
  readonly tokenIds?: readonly number[];
  readonly tokenIndices?: readonly number[];
}

// ---------------------------------------------------------------------------
// Argmax result
// ---------------------------------------------------------------------------

export interface CtcArgmaxResult {
  readonly frameIds: number[];
  readonly selectedLogProbs: Float32Array;
}

// ---------------------------------------------------------------------------
// Collapse result
// ---------------------------------------------------------------------------

export interface CtcCollapseResult {
  readonly collapsedIds: number[];
  readonly tokenSpans: CtcRawTokenSpan[];
}

// ---------------------------------------------------------------------------
// Decoder configuration
// ---------------------------------------------------------------------------

export interface CtcDecoderConfig {
  /** CTC blank token ID (e.g., 0 for WAV2VEC2 pad token). */
  readonly blankId: number;
  /** Vocabulary size (number of output classes). */
  readonly vocabSize: number;
  /** Tokenizer for decoding token IDs to text. */
  readonly tokenizer: CtcTokenizerLike;
  /**
   * Word separator token text.
   * For character-level CTC: ' ' (space, after '|' mapping).
   * For BPE/SentencePiece: undefined (words come from token boundaries).
   */
  readonly wordSeparator?: string;
}

// ---------------------------------------------------------------------------
// Frame-to-seconds estimation options
// ---------------------------------------------------------------------------

export interface CtcFrameTimingOptions {
  readonly audioDurationSec?: number | null;
  readonly inputFrames?: number | null;
  readonly inputFrameHopSeconds?: number;
  readonly outFrames?: number;
}

// ---------------------------------------------------------------------------
// Full decode result (output of CtcDecoder.decodeFromLogits)
// ---------------------------------------------------------------------------

export interface CtcDecodeResult {
  /** The decoded text. */
  readonly text: string;
  /** Collapsed (non-blank, deduplicated) token IDs. */
  readonly collapsedIds: number[];
  /** Argmax token ID per frame. */
  readonly frameIds: number[];
  /** Selected log probability per frame. */
  readonly selectedLogProbs: Float32Array;
  /** Raw token spans (before timing). */
  readonly rawTokenSpans: CtcRawTokenSpan[];
  /** Timed token spans (after frame-to-seconds). */
  readonly tokenSpans: CtcTokenSpan[];
  /** Utterance-level timing. */
  readonly utterance: CtcUtteranceTiming;
  /** Sentence-level timings. */
  readonly sentences: CtcSentenceTiming[];
  /** Word-level timings. */
  readonly words: CtcNativeWord[];
  /** Seconds per output frame. */
  readonly secondsPerFrame: number;
}
