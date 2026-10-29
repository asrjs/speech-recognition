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

import type { DedupWord, FormattedTranscript, Sentence } from '../post-processing/extras.js';
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
  ): Promise<ProductionWhisperTranscriptInput>;
  dispose(): Promise<void>;
}

/**
 * Transcript returned by the executor.
 *
 * `EnhancedWhisperExecutor` returns the Whisper native shape, whose words use
 * `text`, `startTime`, `endTime` and `confidence`. Older custom executors may
 * still return `word`, `start`, `end` and `probability`; both are accepted.
 */
export interface ProductionWhisperTranscriptInput {
  readonly utteranceText: string;
  readonly isFinal: boolean;
  readonly language?: string;
  readonly segments?: readonly unknown[];
  readonly words?: readonly ProductionWhisperWordInput[];
}

/** One executor word in either the Whisper native or the legacy shape. */
export interface ProductionWhisperWordInput {
  /** Whisper native word text. */
  readonly text?: string;
  readonly startTime?: number;
  readonly endTime?: number;
  readonly confidence?: number;
  /** Legacy field names accepted from older custom executors. */
  readonly word?: string;
  readonly start?: number;
  readonly end?: number;
  readonly probability?: number;
}

/**
 * Converts executor words to the post-processing shape.
 *
 * Words without text or finite timing cannot be placed in sentences or
 * subtitles, so they are skipped. A missing confidence becomes 0 rather than an
 * invented value, so a scored duplicate always wins de-duplication.
 */
function toDedupWords(words: readonly ProductionWhisperWordInput[]): DedupWord[] {
  const result: DedupWord[] = [];
  for (const input of words) {
    const word = input.text ?? input.word;
    const start = input.startTime ?? input.start;
    const end = input.endTime ?? input.end;
    if (typeof word !== 'string' || !Number.isFinite(start) || !Number.isFinite(end)) {
      continue;
    }
    const confidence = input.confidence ?? input.probability;
    result.push({
      word,
      start: start as number,
      end: end as number,
      probability: Number.isFinite(confidence) ? (confidence as number) : 0,
    });
  }
  return result;
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
    const words = toDedupWords(nativeResult.words ?? []);

    // 3. Format transcript (sentence boundary + normalization)
    let formatted: FormattedTranscript;
    if (words.length > 0) {
      const { formatTranscript } = await import('../post-processing/extras.js');
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
      language: nativeResult.language ?? options.language ?? 'en',
      sentences: formatted.sentences,
      words,
      segments: nativeResult.segments ?? [],
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
    _words: readonly DedupWord[],
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
    _nativeResult: ProductionWhisperTranscriptInput,
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
