/**
 * Controls how much canonical structure a transcription response includes.
 *
 * Higher detail levels add nested `segments`, `words`, and `tokens`, but the
 * meaning of those fields stays stable across model families and backends.
 */
export type TranscriptDetailLevel = 'text' | 'segments' | 'words' | 'detailed';

/**
 * Selects whether callers receive canonical output, native output, or both.
 */
export type TranscriptResponseFlavor = 'canonical' | 'canonical+native' | 'native';

/** Warning emitted during normalization or inference that does not fit into the main transcript body. */
export interface TranscriptWarning {
  readonly code: string;
  readonly message: string;
  readonly recoverable?: boolean;
}

/** Canonical token-level detail for models that can expose token timing or confidence. */
export interface TranscriptToken {
  readonly index: number;
  readonly text: string;
  readonly rawText?: string;
  readonly id?: number;
  readonly isWordStart?: boolean;
  readonly startTime?: number;
  readonly endTime?: number;
  readonly confidence?: number;
  readonly frameIndex?: number;
  readonly logProb?: number;
  readonly tdtStep?: number;
}

/** Canonical word-level detail with stable timestamp semantics in seconds. */
export interface TranscriptWord {
  readonly index: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
  readonly confidence?: number;
  readonly tokenIndices?: readonly number[];
}

/** Canonical segment-level detail for phrase or utterance-like spans. */
export interface TranscriptSegment {
  readonly index: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
  readonly confidence?: number;
  readonly wordIndices?: readonly number[];
}

/** Optional performance metadata attached to canonical transcript results. */
export interface TranscriptMetrics {
  readonly preprocessMs?: number;
  readonly encodeMs?: number;
  readonly decodeMs?: number;
  readonly tokenizeMs?: number;
  readonly postprocessMs?: number;
  readonly totalMs?: number;
  readonly wallMs?: number;
  readonly audioDurationSec?: number;
  readonly rtf?: number;
  readonly rtfx?: number;
  readonly requestedPreprocessorBackend?: string;
  readonly preprocessorBackend?: string;
  readonly decodeAudioMs?: number;
  readonly downmixMs?: number;
  readonly resampleMs?: number;
  readonly audioPreparationMs?: number;
  readonly inputSampleRate?: number;
  readonly outputSampleRate?: number;
  readonly resampler?: string;
  readonly resamplerQuality?: string | null;
  readonly encoderFrameCount?: number;
  readonly decodeIterations?: number;
  readonly emittedTokenCount?: number;
  readonly emittedWordCount?: number;
}

/** Stable metadata shared by full and streaming transcript objects. */
export interface TranscriptMeta {
  readonly detailLevel: TranscriptDetailLevel;
  readonly isFinal: boolean;
  readonly modelFamily?: string;
  readonly modelId?: string;
  readonly backendId?: string;
  readonly language?: string;
  readonly sampleRate?: number;
  readonly durationSeconds?: number;
  readonly tokenCount?: number;
  readonly wordCount?: number;
  readonly segmentCount?: number;
  readonly averageConfidence?: number;
  readonly averageSegmentConfidence?: number;
  readonly averageWordConfidence?: number;
  readonly averageTokenConfidence?: number;
  readonly nativeAvailable?: boolean;
  readonly producedAt?: string;
  readonly backendNotes?: readonly string[];
  readonly metrics?: TranscriptMetrics;
}

/** Context passed to native-to-canonical transcript normalizers. */
export interface TranscriptNormalizationContext extends Omit<
  Partial<TranscriptMeta>,
  'detailLevel' | 'isFinal'
> {
  readonly detailLevel?: TranscriptDetailLevel;
}

/**
 * Stable app-facing transcript format used across model families and backends.
 *
 * This object is intentionally structured-clone-safe so it can cross worker
 * boundaries without custom serialization.
 */
export interface TranscriptResult {
  readonly text: string;
  readonly warnings: readonly TranscriptWarning[];
  readonly meta: TranscriptMeta;
  readonly segments?: readonly TranscriptSegment[];
  readonly words?: readonly TranscriptWord[];
  readonly tokens?: readonly TranscriptToken[];
}

/**
 * Streaming transcript snapshot.
 *
 * `committedText` is the stable portion of the transcript, while
 * `previewText` can still change as more audio arrives.
 */
export interface PartialTranscript {
  readonly kind: 'partial' | 'final';
  readonly revision: number;
  readonly text: string;
  readonly committedText: string;
  readonly previewText: string;
  readonly warnings: readonly TranscriptWarning[];
  readonly meta: TranscriptMeta;
  readonly segments?: readonly TranscriptSegment[];
  readonly words?: readonly TranscriptWord[];
  readonly tokens?: readonly TranscriptToken[];
}

/** Canonical transcript paired with optional model-native output. */
export interface TranscriptionEnvelope<TNative = unknown> {
  readonly canonical: TranscriptResult;
  readonly native?: TNative;
}

/** Converts model-native output into the shared canonical transcript format. */
export interface TranscriptNormalizer<TNative = unknown> {
  readonly id: string;
  toCanonical(native: TNative, context?: TranscriptNormalizationContext): TranscriptResult;
  toEnvelope(
    native: TNative,
    context?: TranscriptNormalizationContext,
  ): TranscriptionEnvelope<TNative>;
}

/** Response type helper keyed by `responseFlavor`. */
export type TranscriptResponse<
  TNative = unknown,
  TFlavor extends TranscriptResponseFlavor = 'canonical',
> = TFlavor extends 'native'
  ? TNative
  : TFlavor extends 'canonical+native'
    ? TranscriptionEnvelope<TNative>
    : TranscriptResult;
