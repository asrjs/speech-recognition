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

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const rowOffset = frameIndex * vocabSize;
    let bestId = 0;
    let bestValue = Number.NEGATIVE_INFINITY;
    let rowMax = Number.NEGATIVE_INFINITY;

    for (let vocabIndex = 0; vocabIndex < vocabSize; vocabIndex += 1) {
      const value = logits[rowOffset + vocabIndex] ?? Number.NEGATIVE_INFINITY;
      if (value > bestValue) {
        bestValue = value;
        bestId = vocabIndex;
      }
      if (value > rowMax) {
        rowMax = value;
      }
    }

    let expSum = 0;
    for (let vocabIndex = 0; vocabIndex < vocabSize; vocabIndex += 1) {
      expSum += Math.exp((logits[rowOffset + vocabIndex] ?? Number.NEGATIVE_INFINITY) - rowMax);
    }

    frameIds[frameIndex] = bestId;
    selectedLogProbs[frameIndex] = bestValue - (rowMax + Math.log(expSum || 1));
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
