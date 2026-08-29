/**
 * Generic CTC decoder — model-agnostic greedy CTC decoding pipeline.
 *
 * Provides both a stateless function API (backward compatible with lasr-ctc/ctc.ts)
 * and a stateful CtcDecoder class that encapsulates blankId, vocabSize, and tokenizer.
 *
 * All CTC-based models (MedASR, WAV2VEC2, future) should use this module.
 *
 * @module ctc/decoder
 */

import type {
  CtcArgmaxResult,
  CtcCollapseResult,
  CtcDecodeResult,
  CtcDecoderConfig,
  CtcFrameTimingOptions,
  CtcNativeWord,
  CtcRawTokenSpan,
  CtcSentenceTiming,
  CtcTokenSpan,
  CtcTokenizerLike,
  CtcUtteranceTiming,
} from './types.js';

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

const SENTENCE_RE = /[^.!?]+[.!?]+|[^.!?]+$/g;

function clampProbability(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

function splitSentencesWithOffsets(
  text: string,
): Array<{ readonly text: string; readonly startChar: number; readonly endChar: number }> {
  if (!text) {
    return [];
  }

  const sentences: Array<{ text: string; startChar: number; endChar: number }> = [];
  const regex = new RegExp(SENTENCE_RE.source, 'g');
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    const full = match[0] ?? '';
    const trimmed = full.trim();
    if (!trimmed) {
      continue;
    }

    const leadingTrim = full.length - full.trimStart().length;
    const trailingTrim = full.length - full.trimEnd().length;
    const startChar = match.index + leadingTrim;
    const endChar = match.index + full.length - trailingTrim;

    sentences.push({
      text: trimmed,
      startChar,
      endChar,
    });
  }

  return sentences;
}

function decodedCharEndsByToken(
  tokenizer: CtcTokenizerLike,
  tokenIds: readonly number[],
): number[] {
  const charEnds = new Array<number>(tokenIds.length).fill(0);
  const prefix: number[] = [];

  for (let index = 0; index < tokenIds.length; index += 1) {
    prefix.push(tokenIds[index] ?? 0);
    const decoded = tokenizer.decode(prefix);
    charEnds[index] = decoded.length;
  }

  return charEnds;
}

function findStartTokenIndex(charEnds: readonly number[], startChar: number): number {
  for (let index = 0; index < charEnds.length; index += 1) {
    if ((charEnds[index] ?? 0) > startChar) {
      return index;
    }
  }

  return -1;
}

function findEndTokenIndex(charEnds: readonly number[], endCharExclusive: number): number {
  for (let index = 0; index < charEnds.length; index += 1) {
    if ((charEnds[index] ?? 0) >= endCharExclusive) {
      return index;
    }
  }

  return Math.max(0, charEnds.length - 1);
}

function aggregateSpanLogProb(
  spans: readonly CtcTokenSpan[],
  startIndex: number,
  endIndex: number,
): number {
  let weightedSum = 0;
  let frameCount = 0;

  for (let index = startIndex; index <= endIndex; index += 1) {
    const span = spans[index];
    if (!span) {
      continue;
    }

    weightedSum += span.averageLogProb * span.frameCount;
    frameCount += span.frameCount;
  }

  if (frameCount <= 0) {
    return 0;
  }

  return clampProbability(Math.exp(weightedSum / frameCount));
}

// ===========================================================================
// Stateless functions — backward compatible with lasr-ctc/ctc.ts
// ===========================================================================

/**
 * Perform argmax + log-softmax on CTC logits.
 *
 * @param logits Flat logits array [frameCount * vocabSize], row-major.
 * @param frameCount Number of output frames.
 * @param vocabSize Vocabulary size.
 * @returns Per-frame argmax IDs and selected log probabilities.
 */
export function argmaxAndSelectedLogProbs(
  logits: ArrayLike<number>,
  frameCount: number,
  vocabSize: number,
): CtcArgmaxResult {
  const frameIds = new Array<number>(frameCount).fill(0);
  const selectedLogProbs = new Float32Array(frameCount);
  const exp = Math.exp;
  const log = Math.log;

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const rowOffset = frameIndex * vocabSize;
    const rowEnd = rowOffset + vocabSize;
    let bestId = 0;
    let bestValue = Number.NEGATIVE_INFINITY;

    if (rowEnd <= logits.length) {
      for (let index = rowOffset; index < rowEnd; index += 1) {
        const value = logits[index]!;
        if (value > bestValue) {
          bestValue = value;
          bestId = index - rowOffset;
        }
      }
    } else {
      for (let index = rowOffset; index < rowEnd; index += 1) {
        const value = index < logits.length ? logits[index]! : Number.NEGATIVE_INFINITY;
        if (value > bestValue) {
          bestValue = value;
          bestId = index - rowOffset;
        }
      }
    }

    // rowMax in the original formulation is bitwise identical to bestValue
    // (one strict maximum over the same elements), so the normalized score
    // keeps the original expression shape while the exp pass reuses a single
    // hoisted Math.exp over contiguous typed-array access.
    let expSum = 0;
    if (rowEnd <= logits.length) {
      for (let index = rowOffset; index < rowEnd; index += 1) {
        expSum += exp(logits[index]! - bestValue);
      }
    } else {
      for (let index = rowOffset; index < rowEnd; index += 1) {
        expSum += exp((index < logits.length ? logits[index]! : Number.NEGATIVE_INFINITY) - bestValue);
      }
    }

    frameIds[frameIndex] = bestId;
    selectedLogProbs[frameIndex] = bestValue - (bestValue + log(expSum || 1));
  }

  return {
    frameIds,
    selectedLogProbs,
  };
}

// ---------------------------------------------------------------------------
// float16 fast path for CTC argmax + log-softmax
// ---------------------------------------------------------------------------

/**
 * Conservative safe zone for the fp16 exp lookup table: as long as a frame
 * maximum stays within [-FP16_EXP_LUT_SAFE_MAX, FP16_EXP_LUT_SAFE_MAX] the
 * unshifted exp sum cannot overflow or underflow float64 for realistic
 * vocabularies. Rows whose maximum leaves the zone fall back to the generic
 * float pipeline, so raw-logit graphs keep identical semantics.
 */
const FP16_EXP_LUT_SAFE_MAX = 80;

function fp16BitsToFloat(bits: number): number {
  const sign = (bits & 0x8000) << 16;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) {
    if (mantissa === 0) {
      return sign ? -0 : 0;
    }
    let normalized = mantissa;
    let exponentValue = -14;
    while ((normalized & 0x400) === 0) {
      normalized <<= 1;
      exponentValue -= 1;
    }
    normalized &= 0x3ff;
    return (sign ? -1 : 1) * (1 + normalized / 1024) * 2 ** exponentValue;
  }
  if (exponent === 0x1f) {
    return mantissa === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }
  return (sign ? -1 : 1) * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

let fp16ExpLutCache: Float64Array | null = null;

function fp16ExpLookupTable(): Float64Array {
  if (fp16ExpLutCache === null) {
    const table = new Float64Array(0x10000);
    for (let bits = 0; bits < 0x10000; bits += 1) {
      table[bits] = Math.exp(fp16BitsToFloat(bits));
    }
    fp16ExpLutCache = table;
  }
  return fp16ExpLutCache;
}

/**
 * Ordering key for finite IEEE-754 half bits: a larger key means a larger
 * float value, and -0/+0 collapse to the same key so the first strict
 * maximum wins exactly like the generic float scan. NaN and infinity codes
 * never reach this function; callers detect them via isFp16SpecialCode.
 */
function fp16FiniteOrderingKey(bits: number): number {
  return (bits & 0x8000) !== 0 ? 0x8000 - (bits & 0x7fff) : 0x8000 + bits;
}

/** True for plus/minus infinity and NaN half codes (all exponent bits set). */
function isFp16SpecialCode(bits: number): boolean {
  return (bits & 0x7c00) === 0x7c00;
}

function convertFp16BitsRange(bits: Uint16Array, start: number, end: number): Float32Array {
  const out = new Float32Array(end - start);
  for (let index = start; index < end; index += 1) {
    out[index - start] = fp16BitsToFloat(bits[index]!);
  }
  return out;
}

/**
 * Argmax + selected log-probability directly on raw float16 bit patterns.
 *
 * ONNX CTC graphs that emit float16 log-probabilities hand back a
 * Uint16Array of half-precision bits. The generic pipeline first converts
 * every element to float32 and then runs two float passes (max, then sum of
 * exp(x - max)). Because an fp16 code is a 16-bit index, both passes can
 * instead run on integer keys and a precomputed 65536-entry exp lookup
 * table:
 *
 * - max: strict scan over the integer ordering key of the raw bits;
 * - sum: table lookups of Math.exp(fp16ToFloat(bits)) accumulated per row;
 * - score: best - log(sum) is algebraically identical to the reference
 *   -log(sum(exp(x - best))) formulation in argmaxAndSelectedLogProbs.
 *
 * Parity fallbacks: rows containing NaN or infinity codes and rows whose
 * maximum leaves the exp safe zone are recomputed through the converting
 * generic pipeline so the result stays faithful to the float path.
 *
 * @param bits Flat float16 bit patterns [frameCount * vocabSize], row-major.
 * @param frameCount Number of output frames.
 * @param vocabSize Vocabulary size.
 * @returns Per-frame argmax IDs and selected log probabilities.
 */
export function argmaxAndSelectedLogProbsFp16(
  bits: Uint16Array,
  frameCount: number,
  vocabSize: number,
): CtcArgmaxResult {
  const frameIds = new Array<number>(frameCount).fill(0);
  const selectedLogProbs = new Float32Array(frameCount);
  const expLut = fp16ExpLookupTable();
  const log = Math.log;

  const decodeFallbackRow = (rowOffset: number, rowEnd: number, frameIndex: number): void => {
    const converted = convertFp16BitsRange(bits, rowOffset, rowEnd);
    const generic = argmaxAndSelectedLogProbs(converted, 1, rowEnd - rowOffset);
    frameIds[frameIndex] = generic.frameIds[0]!;
    selectedLogProbs[frameIndex] = generic.selectedLogProbs[0]!;
  };

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const rowOffset = frameIndex * vocabSize;
    if (rowOffset >= bits.length) {
      // Row entirely past the buffer: mirror the generic padded scan,
      // which selects id 0 and produces a NaN score.
      selectedLogProbs[frameIndex] = NaN;
      continue;
    }
    const rowEnd = Math.min(rowOffset + vocabSize, bits.length);
    let bestId = 0;
    let bestBits = bits[rowOffset]!;
    let bestKey = -1;
    let sawSpecial = false;
    for (let index = rowOffset; index < rowEnd; index += 1) {
      const code = bits[index]!;
      if (isFp16SpecialCode(code)) {
        sawSpecial = true;
        continue;
      }
      const key = fp16FiniteOrderingKey(code);
      if (key > bestKey) {
        bestKey = key;
        bestBits = code;
        bestId = index - rowOffset;
      }
    }
    const bestValue = bestKey >= 0 ? fp16BitsToFloat(bestBits) : Number.NEGATIVE_INFINITY;
    if (sawSpecial || !(bestValue >= -FP16_EXP_LUT_SAFE_MAX && bestValue <= FP16_EXP_LUT_SAFE_MAX)) {
      decodeFallbackRow(rowOffset, rowEnd, frameIndex);
      continue;
    }
    let expSum = 0;
    for (let index = rowOffset; index < rowEnd; index += 1) {
      expSum += expLut[bits[index]!] ?? 0;
    }
    frameIds[frameIndex] = bestId;
    selectedLogProbs[frameIndex] = bestValue - log(expSum || 1);
  }

  return {
    frameIds,
    selectedLogProbs,
  };
}

/**
 * CTC collapse: remove blanks and consecutive duplicate tokens, producing
 * raw token spans with frame-level timing and confidence.
 */
export function ctcCollapseWithSpans(
  frameIds: readonly number[],
  frameLogProbs: ArrayLike<number>,
  blankId: number,
): CtcCollapseResult {
  const collapsedIds: number[] = [];
  const tokenSpans: CtcRawTokenSpan[] = [];

  if (frameIds.length === 0) {
    return {
      collapsedIds,
      tokenSpans,
    };
  }

  let runId = frameIds[0] ?? blankId;
  let runStart = 0;
  let runFrameCount = 1;
  let runLogProbSum = frameLogProbs[0] ?? 0;

  const flushRun = (endFrame: number): void => {
    if (runId === blankId) {
      return;
    }

    const averageLogProb = runFrameCount > 0 ? runLogProbSum / runFrameCount : 0;
    collapsedIds.push(runId);
    tokenSpans.push({
      tokenId: runId,
      startFrame: runStart,
      endFrame,
      frameCount: runFrameCount,
      averageLogProb,
      confidence: clampProbability(Math.exp(averageLogProb)),
    });
  };

  for (let frameIndex = 1; frameIndex < frameIds.length; frameIndex += 1) {
    const frameId = frameIds[frameIndex] ?? blankId;
    const frameLogProb = frameLogProbs[frameIndex] ?? 0;

    if (frameId !== runId) {
      flushRun(frameIndex - 1);
      runId = frameId;
      runStart = frameIndex;
      runFrameCount = 1;
      runLogProbSum = frameLogProb;
      continue;
    }

    runFrameCount += 1;
    runLogProbSum += frameLogProb;
  }

  flushRun(frameIds.length - 1);

  return {
    collapsedIds,
    tokenSpans,
  };
}

/**
 * Estimate seconds per output frame from timing metadata.
 */
export function estimateSecondsPerOutputFrame(
  options: CtcFrameTimingOptions = {},
): number {
  const outFrames = options.outFrames ?? 0;
  if (!Number.isFinite(outFrames) || outFrames <= 0) {
    return 0;
  }

  if (Number.isFinite(options.audioDurationSec) && (options.audioDurationSec ?? 0) > 0) {
    return (options.audioDurationSec as number) / outFrames;
  }

  if (
    Number.isFinite(options.inputFrames) &&
    (options.inputFrames ?? 0) > 0 &&
    Number.isFinite(options.inputFrameHopSeconds) &&
    (options.inputFrameHopSeconds ?? 0) > 0
  ) {
    return ((options.inputFrames as number) * (options.inputFrameHopSeconds as number)) / outFrames;
  }

  return 0;
}

/**
 * Add timing information (seconds) to raw token spans via a tokenizer.
 */
export function addTimesToTokenSpans(
  tokenizer: CtcTokenizerLike,
  tokenSpans: readonly CtcRawTokenSpan[],
  secondsPerFrame: number,
): CtcTokenSpan[] {
  const safeSecondsPerFrame =
    Number.isFinite(secondsPerFrame) && secondsPerFrame > 0 ? secondsPerFrame : 0;

  return tokenSpans.map((span) => {
    const startTime = span.startFrame * safeSecondsPerFrame;
    const endTime = (span.endFrame + 1) * safeSecondsPerFrame;
    const fallbackPiece = tokenizer.decode([span.tokenId]);

    return {
      tokenId: span.tokenId,
      text: tokenizer.decodeTokenPiece?.(span.tokenId) ?? fallbackPiece,
      startFrame: span.startFrame,
      endFrame: span.endFrame,
      frameCount: span.frameCount,
      startTime,
      endTime,
      duration: Math.max(0, endTime - startTime),
      confidence: span.confidence,
      averageLogProb: span.averageLogProb,
    };
  });
}

/**
 * Build utterance-level timing from frame IDs and log probabilities.
 */
export function buildUtteranceTiming(
  frameIds: readonly number[],
  frameLogProbs: ArrayLike<number>,
  blankId: number,
  secondsPerFrame: number,
): CtcUtteranceTiming {
  let startFrame = -1;
  let endFrame = -1;
  let logProbSum = 0;
  let count = 0;

  for (let frameIndex = 0; frameIndex < frameIds.length; frameIndex += 1) {
    if ((frameIds[frameIndex] ?? blankId) === blankId) {
      continue;
    }

    if (startFrame < 0) {
      startFrame = frameIndex;
    }
    endFrame = frameIndex;
    logProbSum += frameLogProbs[frameIndex] ?? 0;
    count += 1;
  }

  if (startFrame < 0) {
    return {
      hasSpeech: false,
      startFrame: null,
      endFrame: null,
      startTime: 0,
      endTime: 0,
      duration: 0,
      confidence: 0,
    };
  }

  const safeSecondsPerFrame =
    Number.isFinite(secondsPerFrame) && secondsPerFrame > 0 ? secondsPerFrame : 0;
  const startTime = startFrame * safeSecondsPerFrame;
  const endTime = (endFrame + 1) * safeSecondsPerFrame;

  return {
    hasSpeech: true,
    startFrame,
    endFrame,
    startTime,
    endTime,
    duration: Math.max(0, endTime - startTime),
    confidence: count > 0 ? clampProbability(Math.exp(logProbSum / count)) : 0,
  };
}

/**
 * Build sentence-level timings from decoded text and token spans.
 */
export function buildSentenceTimings(
  text: string,
  tokenizer: CtcTokenizerLike,
  collapsedIds: readonly number[],
  tokenSpans: readonly CtcTokenSpan[],
): CtcSentenceTiming[] {
  if (!text || collapsedIds.length === 0 || tokenSpans.length === 0) {
    return [];
  }

  if (collapsedIds.length !== tokenSpans.length) {
    return [];
  }

  const sentenceOffsets = splitSentencesWithOffsets(text);
  if (sentenceOffsets.length === 0) {
    return [];
  }

  const charEnds = decodedCharEndsByToken(tokenizer, collapsedIds);
  const sentenceTimings: CtcSentenceTiming[] = [];

  for (const sentence of sentenceOffsets) {
    const startTokenIndex = findStartTokenIndex(charEnds, sentence.startChar);
    if (startTokenIndex < 0) {
      continue;
    }

    const endTokenIndex = findEndTokenIndex(charEnds, sentence.endChar);
    if (endTokenIndex < startTokenIndex) {
      continue;
    }

    const startSpan = tokenSpans[startTokenIndex];
    const endSpan = tokenSpans[endTokenIndex];
    if (!startSpan || !endSpan) {
      continue;
    }

    sentenceTimings.push({
      text: sentence.text,
      startTokenIndex,
      endTokenIndex,
      startFrame: startSpan.startFrame,
      endFrame: endSpan.endFrame,
      startTime: startSpan.startTime,
      endTime: endSpan.endTime,
      duration: Math.max(0, endSpan.endTime - startSpan.startTime),
      confidence: aggregateSpanLogProb(tokenSpans, startTokenIndex, endTokenIndex),
    });
  }

  return sentenceTimings;
}

/**
 * Build words from character-level CTC token spans.
 *
 * Space tokens (text === wordSeparator) separate words.
 * Tokens between spaces are concatenated into word text.
 *
 * @param tokenSpans Timed token spans with decoded text.
 * @param wordSeparator The character that separates words (default: ' ').
 * @returns Array of words with timing and confidence.
 */
export function buildWordsFromCharSpans(
  tokenSpans: readonly CtcTokenSpan[],
  wordSeparator = ' ',
): CtcNativeWord[] {
  const words: CtcNativeWord[] = [];
  let wordIndex = 0;
  let currentWordText = '';
  let currentWordStart: number | undefined;
  let currentWordEnd: number | undefined;
  let currentWordConfSum = 0;
  let currentWordConfCount = 0;
  let currentWordTokenIndices: number[] = [];
  let currentWordTokenIds: number[] = [];

  const flushWord = (): void => {
    if (currentWordText.length === 0) {
      return;
    }

    words.push({
      index: wordIndex,
      text: currentWordText,
      startTime: currentWordStart ?? 0,
      endTime: currentWordEnd ?? 0,
      confidence:
        currentWordConfCount > 0
          ? currentWordConfSum / currentWordConfCount
          : undefined,
      tokenIds: currentWordTokenIds,
      tokenIndices: currentWordTokenIndices,
    });
    wordIndex += 1;
  };

  for (let spanIndex = 0; spanIndex < tokenSpans.length; spanIndex += 1) {
    const span = tokenSpans[spanIndex];
    if (!span) continue;

    if (span.text === wordSeparator) {
      flushWord();
      currentWordText = '';
      currentWordStart = undefined;
      currentWordEnd = undefined;
      currentWordConfSum = 0;
      currentWordConfCount = 0;
      currentWordTokenIndices = [];
      currentWordTokenIds = [];
      continue;
    }

    if (currentWordStart === undefined) {
      currentWordStart = span.startTime;
    }
    currentWordEnd = span.endTime;
    currentWordText += span.text;
    currentWordConfSum += span.confidence;
    currentWordConfCount += 1;
    currentWordTokenIndices.push(spanIndex);
    currentWordTokenIds.push(span.tokenId);
  }

  flushWord();
  return words;
}

// ===========================================================================
// CtcDecoder class — stateful, encapsulates model-specific CTC config
// ===========================================================================

/**
 * Generic CTC decoder for any model that produces CTC logits.
 *
 * Encapsulates blankId, vocabSize, tokenizer, and word separator.
 * Provides both a one-call `decodeFromLogits()` and individual step methods
 * for fine-grained control.
 *
 * @example
 * ```ts
 * // WAV2VEC2
 * const decoder = new CtcDecoder({
 *   blankId: 0,
 *   vocabSize: 32,
 *   tokenizer: wav2vec2CharTokenizer,
 *   wordSeparator: ' ',
 * });
 *
 * // MedASR
 * const decoder = new CtcDecoder({
 *   blankId: 0,
 *   vocabSize: 512,
 *   tokenizer: medAsrBpeTokenizer,
 * });
 * ```
 */
export class CtcDecoder {
  readonly blankId: number;
  readonly vocabSize: number;
  readonly tokenizer: CtcTokenizerLike;
  readonly wordSeparator: string | undefined;

  constructor(config: CtcDecoderConfig) {
    this.blankId = config.blankId;
    this.vocabSize = config.vocabSize;
    this.tokenizer = config.tokenizer;
    this.wordSeparator = config.wordSeparator;
  }

  // -------------------------------------------------------------------------
  // Full pipeline — one call
  // -------------------------------------------------------------------------

  /**
   * Run the full CTC decode pipeline: argmax → collapse → timing → words.
   *
   * @param logits Flat CTC logits [frameCount * vocabSize], row-major.
   * @param frameCount Number of output frames.
   * @param timingOpts Timing estimation options.
   */
  decodeFromLogits(
    logits: ArrayLike<number>,
    frameCount: number,
    timingOpts: CtcFrameTimingOptions = {},
  ): CtcDecodeResult {
    const { frameIds, selectedLogProbs } = this.argmax(logits, frameCount);
    const { collapsedIds, tokenSpans: rawTokenSpans } = this.collapse(frameIds, selectedLogProbs);
    const text = this.tokenizer.decode(collapsedIds);
    const secondsPerFrame = this.estimateSecondsPerFrame({
      ...timingOpts,
      outFrames: frameCount,
    });
    const tokenSpans = this.addTiming(rawTokenSpans, secondsPerFrame);
    const utterance = this.buildUtterance(frameIds, selectedLogProbs, secondsPerFrame);
    const sentences = this.buildSentences(text, collapsedIds, tokenSpans);
    const words = this.buildWords(tokenSpans);

    return {
      text,
      collapsedIds,
      frameIds,
      selectedLogProbs,
      rawTokenSpans,
      tokenSpans,
      utterance,
      sentences,
      words,
      secondsPerFrame,
    };
  }

  // -------------------------------------------------------------------------
  // Individual steps
  // -------------------------------------------------------------------------

  /** Argmax + log-softmax on logits. */
  argmax(logits: ArrayLike<number>, frameCount: number): CtcArgmaxResult {
    return argmaxAndSelectedLogProbs(logits, frameCount, this.vocabSize);
  }

  /** CTC collapse: remove blanks and consecutive duplicates. */
  collapse(
    frameIds: readonly number[],
    frameLogProbs: ArrayLike<number>,
  ): CtcCollapseResult {
    return ctcCollapseWithSpans(frameIds, frameLogProbs, this.blankId);
  }

  /** Estimate seconds per output frame. */
  estimateSecondsPerFrame(options: CtcFrameTimingOptions = {}): number {
    return estimateSecondsPerOutputFrame(options);
  }

  /** Add timing (seconds) to raw token spans. */
  addTiming(
    rawSpans: readonly CtcRawTokenSpan[],
    secondsPerFrame: number,
  ): CtcTokenSpan[] {
    return addTimesToTokenSpans(this.tokenizer, rawSpans, secondsPerFrame);
  }

  /** Build utterance-level timing. */
  buildUtterance(
    frameIds: readonly number[],
    frameLogProbs: ArrayLike<number>,
    secondsPerFrame: number,
  ): CtcUtteranceTiming {
    return buildUtteranceTiming(frameIds, frameLogProbs, this.blankId, secondsPerFrame);
  }

  /** Build sentence-level timings. */
  buildSentences(
    text: string,
    collapsedIds: readonly number[],
    tokenSpans: readonly CtcTokenSpan[],
  ): CtcSentenceTiming[] {
    return buildSentenceTimings(text, this.tokenizer, collapsedIds, tokenSpans);
  }

  /** Build word-level timings. Uses char-level strategy if wordSeparator is set. */
  buildWords(tokenSpans: readonly CtcTokenSpan[]): CtcNativeWord[] {
    if (this.wordSeparator !== undefined) {
      return buildWordsFromCharSpans(tokenSpans, this.wordSeparator);
    }
    // No word separator → BPE/other tokenization; no automatic word building.
    return [];
  }
}
