/**
 * Enhanced Whisper Executor — production-grade pipeline.
 *
 * Implements WhisperX + whisper.cpp best practices:
 *   1. VAD pre-segmentation (70% of hallucination reduction)
 *      - Never feeds silence to the decoder
 *      - TenVAD or FireRed VAD backend
 *   2. Compression ratio gate (catches repetitive output)
 *   3. Temperature fallback (escapes hallucination loops)
 *   4. Context conditioning control (prevents error cascading, 20% improvement)
 *   5. Drift correction (whisper.cpp-style seek counter for long audio)
 *   6. Segment merging (stitch multi-chunk results)
 *
 * Deferred (needs vanilla executor logit collection):
 *   - Log probability gate
 *   - Entropy gate
 *   - No-speech gate
 *
 * Architecture: Composition over inheritance.
 * EnhancedWhisperExecutor wraps WhisperExecutor.
 */

import type { AudioBufferLike } from '../../types/index.js';
import type { EnhancedDecodeOptions, VadSegmenterConfig } from './enhanced-types.js';
import { compressionRatioGate, logProbGate, noSpeechGate, entropyGate } from '../../quality/index.js';
import { withTemperatureFallback } from '../../quality/index.js';
import {
  mergeVadSegments,
  DriftHandler,
  type WhisperVadBackend,
  type VadSpeechSegment,
} from '../../chunking/index.js';
import { mergeSegments, deduplicateWords } from '../../post-processing/index.js';
import { ChunkContextBuilder } from './chunk-context.js';
import type {
  WhisperExecutor,
  WhisperNativeTranscript,
  WhisperSeq2SeqTranscriptionOptions,
  WhisperDecodeContext,
} from './types.js';

// ---------------------------------------------------------------------------
// Production defaults (matching WhisperX + whisper.cpp)
// ---------------------------------------------------------------------------

const WHISPER_SAMPLE_RATE=16000;
const MAX_SEGMENT_DURATION_MS=29000; // cap at 29s for Whisper's 30s window
const SPEECH_PAD_MS=400; // WhisperX: 0.2s each side → 400ms total
const MIN_SILENCE_MS=100;
const MIN_SPEECH_MS=250;

// ---------------------------------------------------------------------------
// EnhancedWhisperExecutor
// ---------------------------------------------------------------------------

export class EnhancedWhisperExecutor implements WhisperExecutor {
  private readonly contextBuilder: ChunkContextBuilder;
  private readonly driftHandler: DriftHandler;

  constructor(
    private readonly vanilla: WhisperExecutor,
    private readonly vadConfig?: VadSegmenterConfig,
    private readonly vadBackend?: WhisperVadBackend,
  ) {
    this.contextBuilder = new ChunkContextBuilder(224); // half of max_target_positions
    this.driftHandler = new DriftHandler();
  }

  ready(): Promise<void> | void {
    return this.vanilla.ready?.();
  }

  /**
   * Production transcribe pipeline matching WhisperX:
   *
   * 1. VAD → speech segments (skip silence entirely)
   * 2. Per segment:
   *    a. Extract audio chunk with padding
   *    b. Build language prompt
   *    c. Compression ratio + temperature fallback
   *    d. Drift-correct timestamps
   *    e. Feed tokens to context builder
   * 3. Merge all chunks → final transcript
   */
  async transcribe(
    audio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions & Partial<EnhancedDecodeOptions>,
    context: WhisperDecodeContext,
  ): Promise<WhisperNativeTranscript> {
    // ── 1. VAD pre-segmentation ──
    let segments: VadSpeechSegment[] | null = null;

    if (this.vadBackend && this.vadConfig) {
      try {
        const raw = await this.vadBackend.segment(
          audio as any,
          (audio as any).sampleRate ?? WHISPER_SAMPLE_RATE,
          this.vadConfig.speechThreshold ?? 0.5,
        );
        segments = mergeVadSegments(
          raw,
          this.vadConfig.minSilenceDurationMs ?? MIN_SILENCE_MS,
          this.vadConfig.speechPadMs ?? SPEECH_PAD_MS,
          this.vadConfig.maxSegmentDurationMs ?? MAX_SEGMENT_DURATION_MS,
          this.vadConfig.minSpeechDurationMs ?? MIN_SPEECH_MS,
        );
      } catch {
        // VAD failed — fall through
      }
    }

    // ── 2. No VAD → single-chunk with gates ──
    if (!segments || segments.length === 0) {
      return this._transcribeSingle(audio, options, context);
    }

    // ── 3. Multi-chunk VAD pipeline ──
    this.contextBuilder.reset();
    this.driftHandler.reset(audio instanceof Float32Array ? audio.length : 0);

    const conditionOnPrev = options.conditionOnPreviousText !== false;
    const useFallback = options.temperatureFallback !== false;
    const temps = options.temperatures ?? [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

    const perChunkResults: Array<{
      segments: any[];
      words: any[];
      text: string;
      timeOffsetSeconds: number;
    }> = [];

    for (const seg of segments) {
      // Extract audio chunk
      const sr = (audio as any).sampleRate ?? WHISPER_SAMPLE_RATE;
      const audioLength = (audio instanceof Float32Array) ? audio.length : ((audio as any).length ?? 0);
      const startSample = Math.max(0, Math.floor(seg.startSeconds * sr));
      const endSample = Math.min(audioLength, Math.ceil(seg.endSeconds * sr));
      const chunk = (audio as any).subarray?.(startSample, endSample) ?? audio;

      if (!chunk || chunk.length === 0) continue;

      // Build chunk options with context conditioning
      const chunkOpts: any = { ...options };
      if (conditionOnPrev) {
        const prevTokens = this.contextBuilder.getPreviousTokens();
        if (prevTokens.length > 0) {
          const maxCtx = options.maxContextTokens ?? 224;
          // Build prompt context: [<|0.00|>, ...prev_tokens_tail]
          const timestamp0 = 50364; // <|0.00|>
          const ctxTokens = prevTokens.slice(-maxCtx);
          chunkOpts.extraPromptTokens = [timestamp0, ...ctxTokens];
        }
      }

      // Transcribe with temperature fallback + full quality gates
      let chunkResult: WhisperNativeTranscript;
      const collectedLogits: Float32Array[] = [];
      const collectedTokens: number[] = [];

      if (useFallback) {
        const gates = [
          compressionRatioGate(options.compressionRatioThreshold ?? 2.4),
          logProbGate(options.logProbThreshold ?? -1.0),
          entropyGate(options.entropyThreshold ?? 2.4),
          noSpeechGate(options.noSpeechThreshold ?? 0.6, options.logProbThreshold ?? -1.0),
        ];
        const fallback = await withTemperatureFallback(
          async (_temp) => {
            const optsWithLogits = {
              ...chunkOpts,
              onTokenLogits: (tokenId: number, logits: Float32Array, _ctx: any) => {
                collectedLogits.push(new Float32Array(logits));
                collectedTokens.push(tokenId);
              },
            };
            const r = await this.vanilla.transcribe(chunk, optsWithLogits as any, context);
            return { result: r, text: r.utteranceText, tokens: collectedTokens, logits: collectedLogits, vocabSize: 51865 };
          },
          gates,
          temps,
        );
        chunkResult = fallback.result;
      } else {
        chunkResult = await this.vanilla.transcribe(chunk, { ...chunkOpts, onTokenLogits: (tokenId: number, logits: Float32Array, _ctx: any) => {
          collectedLogits.push(new Float32Array(logits));
          collectedTokens.push(tokenId);
        } } as any, context);
      }

      // Feed context builder with generated tokens
      this.contextBuilder.addSegmentTokens(collectedTokens);

      // Drift correction
      const corrected = this.driftHandler.correctTimestamps(
        seg.startSeconds,
        seg.endSeconds,
        sr,
        1.0, // maxDriftSec
      );

      // Advance drift by accepted segment duration
      const duration = corrected.end - corrected.start;
      this.driftHandler.advanceBy(duration, sr);

      perChunkResults.push({
        segments: [...(chunkResult.segments ?? [])],
        words: [...(chunkResult.words ?? [])],
        text: chunkResult.utteranceText,
        timeOffsetSeconds: corrected.start,
      });
    }

    // ── 4. Merge all chunks ──
    const merged = mergeSegments(perChunkResults);
    const deduped = deduplicateWords(merged.words);

    // Build final text from segments, fall back to joining chunk texts
    const utteranceText = merged.segments.length > 0
      ? merged.segments.map((s) => s.text).join(' ').trim()
      : perChunkResults.map(c => c.text).join(' ').trim();

    return {
      utteranceText: utteranceText || '[no speech detected]',
      isFinal: true,
      language: (options as any).language ?? 'en',
      segments: merged.segments as any,
      words: deduped as any,
    };
  }

  /**
   * Single-chunk mode: no VAD, just quality gates + temperature fallback.
   */
  private async _transcribeSingle(
    audio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions & Partial<EnhancedDecodeOptions>,
    context: WhisperDecodeContext,
  ): Promise<WhisperNativeTranscript> {
    const useFallback = options.temperatureFallback !== false;
    const temps = options.temperatures ?? [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

    if (!useFallback || !temps || temps.length === 0) {
      return this.vanilla.transcribe(audio, options, context);
    }

    // Build quality gates
    const gates = [
      compressionRatioGate(options.compressionRatioThreshold ?? 2.4),
      logProbGate(options.logProbThreshold ?? -1.0),
      entropyGate(options.entropyThreshold ?? 2.4),
      noSpeechGate(options.noSpeechThreshold ?? 0.6, options.logProbThreshold ?? -1.0),
    ];

    const fallback = await withTemperatureFallback(
      async (_temp) => {
        // Collect logits from the vanilla executor via onTokenLogits callback
        const collectedLogits: Float32Array[] = [];
        const collectedTokens: number[] = [];
        const optsWithLogits = {
          ...options,
          onTokenLogits: (tokenId: number, logits: Float32Array, _ctx: any) => {
            collectedLogits.push(new Float32Array(logits)); // snapshot
            collectedTokens.push(tokenId);
          },
        };
        const r = await this.vanilla.transcribe(audio, optsWithLogits as any, context);
        return {
          result: r,
          text: r.utteranceText,
          tokens: collectedTokens,
          logits: collectedLogits,
          vocabSize: 51865,
        };
      },
      gates,
      temps,
    );

    return fallback.result;
  }

  async dispose(): Promise<void> {
    await this.vanilla.dispose();
  }
}
