/**
 * Types for the production Whisper pipeline output.
 *
 * @module pipeline/types
 */

import type { Sentence } from '../post-processing/extras.js';

export interface ProductionTranscriptSubtitles {
  readonly srt: string;
  readonly vtt: string;
}

export interface ProductionTranscriptMetrics {
  readonly duration: number;
  readonly wordCount: number;
  readonly sentenceCount: number;
}

export interface ProductionWord {
  readonly word: string;
  readonly start: number;
  readonly end: number;
  readonly probability: number;
}

export interface ProductionSegment {
  readonly text: string;
  readonly start: number;
  readonly end: number;
  readonly words?: readonly ProductionWord[];
}

export interface ProductionTranscript {
  /** Full transcript text with sentences separated by newlines */
  readonly text: string;
  /** Normalized text (collapsed spaces) */
  readonly normalized: string;
  /** Raw utterance text from the executor */
  readonly raw: string;
  /** Detected language */
  readonly language: string;
  /** Sentences with word-level timestamps */
  readonly sentences: readonly Sentence[];
  /** Words with timestamps */
  readonly words: readonly ProductionWord[];
  /** Segments from the executor */
  readonly segments: readonly ProductionSegment[];
  /** Subtitles (SRT + VTT) */
  readonly subtitles: ProductionTranscriptSubtitles;
  /** Metrics */
  readonly metrics: ProductionTranscriptMetrics;
}
