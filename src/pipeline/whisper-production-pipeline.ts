/**
 * Production Whisper Transcription Pipeline — end-to-end from audio to formatted output.
 *
 * Composes:
 *   1. EnhancedWhisperExecutor (VAD + quality gates + temp fallback + drift + context)
 *   2. Post-processing (sentence boundary + normalization)
 *   3. Subtitle generation (SRT + VTT)
 *   4. Metrics collection
 *
 * Modeled after WhisperX + whisper.cpp production best practices.
 *
 * @module pipeline/whisper-production-pipeline
 */

import {
  formatTranscript,
  type FormattedTranscript,
  type Sentence,
} from '../post-processing/extras.js';
import type {
  ProductionTranscript,
  ProductionTranscriptMetrics,
  ProductionTranscriptSubtitles,
} from './types.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ProductionWhisperPipelineOptions {
  /** Enhanced whisper executor (VAD+gates+fallback+drift wired) */
  readonly enhancedExecutor: ProductionWhisperExecutorLike;
  /** Output formats to generate */
  readonly outputFormats?: readonly ('sentences' | 'srt' | 'vtt')[];
  /** Default VAD backend */
  readonly vadConfig?: Record<string, unknown>;
  /** Default temperature fallback sequence */
  readonly temperatures?: readonly number[];
}

export interface ProductionWhisperTranscribeOptions {
  readonly sampleRate: number;
  readonly language?: string;
  readonly returnWordTimestamps?: boolean;
  /** Override per-call VAD config */
  readonly vadConfig?: Record<string, unknown>;
}

/**
 * Minimal interface for the enhanced executor.
 * Accepts any object with transcribe() matching EnhancedWhisperExecutor signature.
 */
export interface ProductionWhisperExecutorLike {
  transcribe(
    audio: Float32Array,
    options?: any,
    context?: any,
  ): Promise<WhisperNativeTranscriptLike>;
  dispose(): Promise<void>;
}

interface WhisperNativeTranscriptLike {
  readonly utteranceText: string;
  readonly isFinal: boolean;
  readonly language: string;
  readonly segments?: readonly SegmentLike[];
  readonly words?: readonly WordLike[];
}

interface SegmentLike {
  readonly text: string;
  readonly start: number;
  readonly end: number;
  readonly words?: readonly WordLike[];
}

interface WordLike {
  readonly word: string;
  readonly start: number;
  readonly end: number;
  readonly probability?: number;
}

// ---------------------------------------------------------------------------
// Production Pipeline
// ---------------------------------------------------------------------------

export class ProductionWhisperPipeline {
  private readonly executor: ProductionWhisperExecutorLike;
  private readonly outputFormats: readonly string[];
  private readonly defaultVadConfig?: Record<string, unknown>;
  private readonly defaultTemperatures?: readonly number[];

  constructor(options: ProductionWhisperPipelineOptions) {
    this.executor = options.enhancedExecutor;
    this.outputFormats = options.outputFormats ?? ['sentences'];
    this.defaultVadConfig = options.vadConfig;
    this.defaultTemperatures = options.temperatures;
  }

  async transcribe(
    audio: Float32Array,
    options: ProductionWhisperTranscribeOptions,
  ): Promise<ProductionTranscript> {
    const audioDurationSeconds = audio.length / options.sampleRate;

    // 1. Run enhanced executor (VAD → gates → fallback → drift → context → merge)
    const nativeResult = await this.executor.transcribe(audio, {
      language: options.language ?? 'en',
      returnWordTimestamps: options.returnWordTimestamps ?? true,
      vadConfig: options.vadConfig ?? this.defaultVadConfig,
      temperatures: this.defaultTemperatures,
    } as any);

    // 2. Extract words for post-processing
    const words: DedupWordLike[] = (nativeResult.words ?? []).map((w: WordLike) => ({
      word: w.word,
      start: w.start,
      end: w.end,
      probability: w.probability ?? 0.9,
    }));

    // 3. Format transcript (sentence boundary + normalization)
    let formatted: FormattedTranscript;
    if (words.length > 0) {
      formatted = formatTranscript(words, audioDurationSeconds);
    } else {
      formatted = {
        text: nativeResult.utteranceText ?? '',
        normalized: nativeResult.utteranceText ?? '',
        sentences: [],
        duration: audioDurationSeconds,
        wordCount: 0,
      } as FormattedTranscript;
    }

    // 4. Generate subtitles (SRT + VTT)
    const subtitles = this.generateSubtitles(formatted.sentences, words);

    // 5. Collect metrics using raw word data
    const metrics = this.collectMetrics(formatted, nativeResult);

    return {
      // Use raw text from executor (preserves punctuation/casing)
      // Fall back to formatted text if words have the full content
      text: nativeResult.utteranceText || formatted.text,
      // Normalized version from sentence processing
      normalized: formatted.normalized,
      raw: nativeResult.utteranceText,
      language: nativeResult.language,
      sentences: formatted.sentences,
      words: words as any,
      segments: nativeResult.segments as any,
      subtitles,
      metrics: {
        ...metrics,
        // Word count from the raw words array, not from deduped
        wordCount: (nativeResult.words ?? []).length,
      },
    };
  }

  async dispose(): Promise<void> {
    await this.executor.dispose();
  }

  // -----------------------------------------------------------------------
  // Private helpers
  // -----------------------------------------------------------------------

  private generateSubtitles(
    sentences: readonly Sentence[],
    _words: readonly DedupWordLike[],
  ): ProductionTranscriptSubtitles {
    let srt = '';
    let vtt = '';

    if (this.outputFormats.includes('srt') || this.outputFormats.includes('vtt')) {
      srt = sentencesToSrt(sentences);
      vtt = sentencesToVtt(sentences);
    }

    return { srt, vtt };
  }

  private collectMetrics(
    formatted: FormattedTranscript,
    _nativeResult: WhisperNativeTranscriptLike,
  ): ProductionTranscriptMetrics {
    return {
      duration: formatted.duration,
      wordCount: formatted.wordCount,
      sentenceCount: formatted.sentences.length,
    };
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createWhisperProductionPipeline(
  options: ProductionWhisperPipelineOptions,
): ProductionWhisperPipeline {
  return new ProductionWhisperPipeline(options);
}

// ---------------------------------------------------------------------------
// Types used for post-processing
// ---------------------------------------------------------------------------

interface DedupWordLike {
  word: string;
  start: number;
  end: number;
  probability: number;
}

// ---------------------------------------------------------------------------
// Subtitle formatters (inline — reuse existing pipeline/subtitles.ts when wired)
// ---------------------------------------------------------------------------

function formatSrtTimestamp(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = Math.floor(totalSeconds % 60);
  const millis = Math.floor((totalSeconds % 1) * 1000);
  return (
    String(hours).padStart(2, '0') +
    ':' +
    String(minutes).padStart(2, '0') +
    ':' +
    String(seconds).padStart(2, '0') +
    ',' +
    String(millis).padStart(3, '0')
  );
}

function sentencesToSrt(sentences: readonly Sentence[]): string {
  return sentences
    .map((sentence, i) => {
      const start = formatSrtTimestamp(sentence.start);
      const end = formatSrtTimestamp(sentence.end);
      return `${i + 1}\n${start} --> ${end}\n${sentence.text.trim()}\n`;
    })
    .join('\n');
}

function sentencesToVtt(sentences: readonly Sentence[]): string {
  const header = 'WEBVTT\n\n';
  const cues = sentences
    .map((sentence, i) => {
      const start = formatVttTimestamp(sentence.start);
      const end = formatVttTimestamp(sentence.end);
      return `${i + 1}\n${start} --> ${end}\n${sentence.text.trim()}\n`;
    })
    .join('\n');
  return header + cues;
}

function formatVttTimestamp(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = Math.floor(totalSeconds % 60);
  const millis = Math.floor((totalSeconds % 1) * 1000);
  return (
    String(hours).padStart(2, '0') +
    ':' +
    String(minutes).padStart(2, '0') +
    ':' +
    String(seconds).padStart(2, '0') +
    '.' +
    String(millis).padStart(3, '0')
  );
}
