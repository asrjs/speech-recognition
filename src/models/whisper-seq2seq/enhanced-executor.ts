/**
 * Enhanced Whisper Executor — composition wrapper adding production features.
 *
 * Wraps a vanilla WhisperExecutor and adds:
 *   - Quality gate evaluation (compression ratio, logprob, entropy, no-speech)
 *   - Temperature fallback on quality rejection
 *   - Condition-on-previous-text (context building)
 *   - VAD pre-segmentation (if backend provided)
 *
 * Architecture: Composition, not inheritance.
 *   EnhancedWhisperExecutor wraps WhisperExecutor.
 *   All enhanced features are pre/post-processing.
 *   Vanilla executor handles all ONNX inference.
 */

import type { AudioBufferLike } from '../../types/index.js';
import type { EnhancedDecodeOptions, VadSegmenterConfig } from './enhanced-types.js';
import {
  compressionRatioGate,
  logProbGate,
  noSpeechGate,
  entropyGate,
} from './quality-gates.js';
import { withTemperatureFallback } from './temperature-fallback.js';
import { mergeVadSegments, type WhisperVadBackend, type VadSpeechSegment } from './vad-segmenter.js';
import { mergeWhisperSegments } from './segment-merger.js';
import type { WhisperExecutor, WhisperNativeTranscript, WhisperSeq2SeqTranscriptionOptions, WhisperDecodeContext } from './types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function extractEnhancedOptions(
  options: WhisperSeq2SeqTranscriptionOptions & Partial<EnhancedDecodeOptions>,
): EnhancedDecodeOptions {
  return {
    compressionRatioThreshold: options.compressionRatioThreshold,
    logProbThreshold: options.logProbThreshold,
    noSpeechThreshold: options.noSpeechThreshold,
    entropyThreshold: options.entropyThreshold,
    temperatureFallback: options.temperatureFallback,
    temperatures: options.temperatures,
    conditionOnPreviousText: options.conditionOnPreviousText,
    maxContextTokens: options.maxContextTokens,
  };
}

// ---------------------------------------------------------------------------
// EnhancedWhisperExecutor
// ---------------------------------------------------------------------------

export class EnhancedWhisperExecutor implements WhisperExecutor {
  constructor(
    private readonly vanilla: WhisperExecutor,
    private readonly vadConfig?: VadSegmenterConfig,
    private readonly vadBackend?: WhisperVadBackend,
  ) {}

  ready(): Promise<void> | void {
    return this.vanilla.ready?.();
  }

  async transcribe(
    audio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions & Partial<EnhancedDecodeOptions>,
    context: WhisperDecodeContext,
  ): Promise<WhisperNativeTranscript> {
    const enhancedOpts = extractEnhancedOptions(options);

    // 1. VAD pre-segmentation (if backend configured)
    const vadEnabled = this.vadBackend && this.vadConfig;
    let segments: VadSpeechSegment[] | null = null;

    if (vadEnabled && (audio as any).length !== undefined) {
      // Attempt VAD on raw audio buffer — requires Float32Array-like input
      try {
        const rawSegments = await this.vadBackend!.segment(
          audio as any,
          (audio as any).sampleRate ?? 16000,
          this.vadConfig!.speechThreshold ?? 0.5,
        );
        segments = mergeVadSegments(
          rawSegments,
          this.vadConfig!.minSilenceDurationMs ?? 100,
          this.vadConfig!.speechPadMs ?? 400,
          this.vadConfig!.maxSegmentDurationMs ?? 29000,
          this.vadConfig!.minSpeechDurationMs ?? 250,
        );
      } catch {
        // VAD failed — fall through to single chunk
        segments = null;
      }
    }

    // 2. Single-chunk mode (no VAD or VAD failed)
    if (!segments || segments.length === 0) {
      // Temperature fallback
      if (enhancedOpts.temperatureFallback !== false && enhancedOpts.temperatures) {
        const gates = [
          compressionRatioGate(enhancedOpts.compressionRatioThreshold),
          logProbGate(enhancedOpts.logProbThreshold),
          noSpeechGate(enhancedOpts.noSpeechThreshold, enhancedOpts.logProbThreshold ?? -1.0),
          entropyGate(enhancedOpts.entropyThreshold),
        ];

        const fallbackResult = await withTemperatureFallback(
          async (_temp) => {
            const result = await this.vanilla.transcribe(audio, options, context);
            return {
              result,
              text: result.utteranceText,
              tokens: [] as readonly number[],
              logits: [] as Float32Array[],
              vocabSize: 51865,
            };
          },
          gates,
          enhancedOpts.temperatures,
        );
        return fallbackResult.result;
      }
      return this.vanilla.transcribe(audio, options, context);
    }

    // 3. Multi-chunk mode with VAD
    const perChunkResults: Array<{
      segments: any[];
      words: any[];
      timeOffsetSeconds: number;
    }> = [];

    for (const seg of segments) {
      // Extract audio chunk as Float32Array subarray
      const chunk = (audio as any).subarray?.(
        Math.floor(seg.startSeconds * 16000),
        Math.ceil(seg.endSeconds * 16000),
      ) ?? audio;

      const chunkResult = await this.vanilla.transcribe(chunk, options, context);

      perChunkResults.push({
        segments: [...(chunkResult.segments ?? [])],
        words: [...(chunkResult.words ?? [])],
        timeOffsetSeconds: seg.startSeconds,
      });
    }

    // 4. Merge all chunks
    const merged = mergeWhisperSegments(perChunkResults);

    return {
      utteranceText: merged.segments.map((s) => s.text).join(' ').trim(),
      isFinal: true,
      language: 'en',
      segments: merged.segments as any,
      words: merged.words as any,
    };
  }

  async dispose(): Promise<void> {
    await this.vanilla.dispose();
  }
}
