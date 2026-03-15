import type { LasrCtcSentenceTiming, LasrCtcTokenSpan, LasrCtcUtteranceTiming } from './types.js';

interface RawTokenSpan {
  readonly tokenId: number;
  readonly startFrame: number;
  readonly endFrame: number;
  readonly frameCount: number;
  readonly averageLogProb: number;
  readonly confidence: number;
}

interface TokenDecoderLike {
  decode(ids: readonly number[]): string;
  decodeTokenPiece?(tokenId: number): string;
}

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
  tokenizer: TokenDecoderLike,
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
  spans: readonly LasrCtcTokenSpan[],
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

export function argmaxAndSelectedLogProbs(
  logits: ArrayLike<number>,
  frameCount: number,
  vocabSize: number,
): { readonly frameIds: number[]; readonly selectedLogProbs: Float32Array } {
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

export function ctcCollapseWithSpans(
  frameIds: readonly number[],
  frameLogProbs: ArrayLike<number>,
  blankId: number,
): { readonly collapsedIds: number[]; readonly tokenSpans: RawTokenSpan[] } {
  const collapsedIds: number[] = [];
  const tokenSpans: RawTokenSpan[] = [];

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

export function estimateSecondsPerOutputFrame(
  options: {
    readonly audioDurationSec?: number | null;
    readonly inputFrames?: number | null;
    readonly inputFrameHopSeconds?: number;
    readonly outFrames?: number;
  } = {},
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

export function addTimesToTokenSpans(
  tokenizer: TokenDecoderLike,
  tokenSpans: readonly RawTokenSpan[],
  secondsPerFrame: number,
): LasrCtcTokenSpan[] {
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

export function buildUtteranceTiming(
  frameIds: readonly number[],
  frameLogProbs: ArrayLike<number>,
  blankId: number,
  secondsPerFrame: number,
): LasrCtcUtteranceTiming {
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

export function buildSentenceTimings(
  text: string,
  tokenizer: TokenDecoderLike,
  collapsedIds: readonly number[],
  tokenSpans: readonly LasrCtcTokenSpan[],
): LasrCtcSentenceTiming[] {
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
  const sentenceTimings: LasrCtcSentenceTiming[] = [];

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
