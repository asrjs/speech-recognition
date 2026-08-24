/**
 * Quality Gate Types — shared contracts for hallucination suppression.
 *
 * Model-agnostic. Works with any ASR model that produces text + token logits.
 */

// ---------------------------------------------------------------------------
// Verdicts
// ---------------------------------------------------------------------------

/** Outcome of a quality gate evaluation. */
export type QualityVerdict = 'accept' | 'reject' | 'no_speech';

// ---------------------------------------------------------------------------
// Quality Gate
// ---------------------------------------------------------------------------

/** Scalar per-token measurements for a selected decode sequence. */
export interface TokenQualityTrace {
  readonly tokenId: number;
  readonly logProb: number;
  readonly entropy: number;
}

/** Context from the model runtime for quality checks that need raw logits. */
export interface QualityGateContext {
  /** Raw decoder-init logits used by Whisper's no-speech test. */
  readonly noSpeechLogits?: Float32Array | readonly number[];
  /** Model-specific no-speech token ID. */
  readonly noSpeechTokenId?: number;
  /**
   * Selected-sequence scalar traces. When present, logprob/entropy gates use
   * these instead of retaining full-vocabulary logits.
   */
  readonly tokenTraces?: readonly TokenQualityTrace[];
}

/** A quality gate function — evaluates a decode result. */
export interface QualityGate {
  (
    text: string,
    tokens: readonly number[],
    logits: readonly Float32Array[],
    vocabSize: number,
    context?: QualityGateContext,
  ): QualityGateResult;
}

/** Result of running one or more quality gates on a decode result. */
export interface QualityGateResult {
  readonly verdict: QualityVerdict;
  readonly compressionRatio?: number;
  readonly avgLogProb?: number;
  readonly noSpeechProb?: number;
  readonly entropy?: number;
  readonly reason?: string;
}

// ---------------------------------------------------------------------------
// Per-segment metrics
// ---------------------------------------------------------------------------

/** Quality metrics collected during a single decode run. */
export interface SegmentQualityMetrics {
  readonly compressionRatio: number;
  readonly avgLogProb: number;
  readonly noSpeechProb: number;
  readonly entropy: number;
  readonly temperature: number;
}
