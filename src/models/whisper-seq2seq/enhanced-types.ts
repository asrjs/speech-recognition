/**
 * Enhanced Whisper types — quality gates, temperature fallback, VAD segmenter config.
 *
 * These are the shared contract between:
 *   - quality-gates.ts (compression, logprob, entropy, no-speech)
 *   - temperature-fallback.ts (retry loop)
 *   - chunk-context.ts (condition-on-previous-text)
 *   - vad-segmenter.ts (VAD pre-segmentation)
 *   - enhanced-executor.ts (composition wrapper)
 *
 */

// ---------------------------------------------------------------------------
// Quality verdicts
// ---------------------------------------------------------------------------

/** Outcome of a quality gate evaluation. */
export type QualityVerdict = 'accept' | 'reject' | 'no_speech';

/** A quality gate function — evaluates a decode result. */
export interface QualityGate {
  (text: string, tokens: readonly number[], logits: readonly Float32Array[], vocabSize: number): QualityGateResult;
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

// ---------------------------------------------------------------------------
// Enhanced decode result
// ---------------------------------------------------------------------------

/** Decode result with quality metrics attached. */
export interface EnhancedDecodeResult {
  readonly tokens: readonly number[];
  readonly text: string;
  readonly metrics: SegmentQualityMetrics;
}

// ---------------------------------------------------------------------------
// Enhanced decode options
// ---------------------------------------------------------------------------

/** Options for the enhanced decoder — extends vanilla options with quality gates. */
export interface EnhancedDecodeOptions {
  /** Compression ratio threshold — reject if higher (default: 2.4) */
  readonly compressionRatioThreshold?: number;

  /** Average log probability threshold — reject if lower (default: -1.0) */
  readonly logProbThreshold?: number;

  /** No-speech probability threshold (default: 0.6) */
  readonly noSpeechThreshold?: number;

  /** Entropy threshold in nats — reject if higher (default: 2.4) */
  readonly entropyThreshold?: number;

  /** Enable temperature fallback on quality rejection (default: true) */
  readonly temperatureFallback?: boolean;

  /** Temperature schedule for fallback (default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) */
  readonly temperatures?: readonly number[];

  /** Include previous segment text in subsequent chunk prompts (default: true) */
  readonly conditionOnPreviousText?: boolean;

  /** Max tokens to include from previous segments (default: maxTargetPositions / 2) */
  readonly maxContextTokens?: number;
}

// ---------------------------------------------------------------------------
// VAD segmenter config
// ---------------------------------------------------------------------------

/** Configuration for VAD-based audio pre-segmentation. */
export interface VadSegmenterConfig {
  /** VAD backend to use */
  readonly backend: 'ten-vad' | 'firered-vad';

  /** Speech probability threshold (default: 0.5) */
  readonly speechThreshold?: number;

  /** Minimum speech segment duration in ms (default: 250) */
  readonly minSpeechDurationMs?: number;

  /** Minimum silence between segments in ms (default: 100) */
  readonly minSilenceDurationMs?: number;

  /** Padding added to each side of speech segments in ms (default: 400) */
  readonly speechPadMs?: number;

  /** Maximum segment duration in ms — caps at ~29s for Whisper's 30s window (default: 29000) */
  readonly maxSegmentDurationMs?: number;
}

// ---------------------------------------------------------------------------
// Default factories
// ---------------------------------------------------------------------------

/** Return EnhancedDecodeOptions with defaults filled in. */
export function makeDefaultEnhancedDecodeOptions(
  overrides?: Partial<EnhancedDecodeOptions>,
): EnhancedDecodeOptions {
  return {
    compressionRatioThreshold: 2.4,
    logProbThreshold: -1.0,
    noSpeechThreshold: 0.6,
    entropyThreshold: 2.4,
    temperatureFallback: true,
    temperatures: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    conditionOnPreviousText: true,
    ...overrides,
  };
}

/** Return VadSegmenterConfig with defaults filled in. */
export function makeDefaultVadSegmenterConfig(
  overrides: Partial<VadSegmenterConfig> & { backend: VadSegmenterConfig['backend'] },
): VadSegmenterConfig {
  return {
    speechThreshold: 0.5,
    minSpeechDurationMs: 250,
    minSilenceDurationMs: 100,
    speechPadMs: 400,
    maxSegmentDurationMs: 29000,
    ...overrides,
    backend: overrides.backend, // always required, ensure it's last
  };
}
