import {
  collateWhisperWordTimestamps,
  isWhisperTimestampToken,
  whisperTimestampTokenToSeconds,
  type WhisperTimestampTokenOptions,
  type WhisperTokenTimestamp,
} from '../../pipeline/whisper-timestamps.js';
import type { WhisperNativeToken, WhisperNativeWord } from './types.js';

export interface BuildWhisperWordTimestampOptions extends WhisperTimestampTokenOptions {
  readonly language?: string | null;
}

interface InterpolatedTokenTimestamp extends WhisperTokenTimestamp {
  readonly sourceIndex: number;
  readonly confidence?: number;
}

export function buildWhisperWordTimestampsFromTokenDetails(
  tokens: readonly WhisperNativeToken[],
  options: BuildWhisperWordTimestampOptions,
): WhisperNativeWord[] {
  const tokenTimestamps = interpolateTokenTimestamps(tokens, options);
  if (tokenTimestamps.length === 0) return [];

  const words = collateWhisperWordTimestamps(tokenTimestamps, { language: options.language });
  let cursor = 0;
  return words.map((word) => {
    const sourceTokens = tokenTimestamps.slice(cursor, cursor + word.tokenIds.length);
    cursor += word.tokenIds.length;
    const confidences = sourceTokens
      .map((token) => token.confidence)
      .filter((confidence): confidence is number => confidence !== undefined);
    return {
      index: word.index,
      text: word.text,
      startTime: word.startTime,
      endTime: word.endTime,
      tokenIds: word.tokenIds,
      tokenIndices: sourceTokens.map((token) => token.sourceIndex),
      ...(confidences.length > 0
        ? { confidence: confidences.reduce((sum, confidence) => sum + confidence, 0) / confidences.length }
        : {}),
    };
  });
}

function interpolateTokenTimestamps(
  tokens: readonly WhisperNativeToken[],
  options: BuildWhisperWordTimestampOptions,
): InterpolatedTokenTimestamp[] {
  const output: InterpolatedTokenTimestamp[] = [];
  let segmentStart: number | null = null;
  let segmentTokens: WhisperNativeToken[] = [];

  for (const token of tokens) {
    const tokenId = token.id;
    if (tokenId === undefined || !isWhisperTimestampToken(tokenId, options)) {
      if (segmentStart !== null && tokenId !== undefined && !token.special) {
        segmentTokens.push(token);
      }
      continue;
    }

    const timestamp = whisperTimestampTokenToSeconds(tokenId, options);
    if (segmentStart === null) {
      segmentStart = timestamp;
      segmentTokens = [];
      continue;
    }

    output.push(...interpolateSegmentTokens(segmentTokens, segmentStart, timestamp));
    segmentStart = timestamp;
    segmentTokens = [];
  }

  return output;
}

function interpolateSegmentTokens(
  tokens: readonly WhisperNativeToken[],
  startTime: number,
  endTime: number,
): InterpolatedTokenTimestamp[] {
  if (tokens.length === 0 || endTime <= startTime) return [];
  const duration = endTime - startTime;
  return tokens.flatMap((token, index) => {
    if (token.id === undefined || token.text.length === 0) return [];
    const tokenStart = roundTime(startTime + (duration * index) / tokens.length);
    const tokenEnd = roundTime(startTime + (duration * (index + 1)) / tokens.length);
    return [
      {
        tokenId: token.id,
        text: token.text,
        startTime: tokenStart,
        endTime: tokenEnd,
        sourceIndex: token.index,
        confidence: token.confidence,
      },
    ];
  });
}

function roundTime(seconds: number): number {
  return Math.round(seconds * 1000) / 1000;
}
