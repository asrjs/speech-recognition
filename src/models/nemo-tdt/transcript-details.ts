import type { TranscriptWarning } from '../../types/index.js';
import type { NemoTdtNativeToken, NemoTdtNativeTranscript } from './types.js';
import { ParakeetTokenizer } from './tokenizer.js';

function isFiniteNumber(value: number | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

export function buildWordAndTokenDetails(
  tokenizer: ParakeetTokenizer,
  tokenIds: readonly number[],
  tokenTimestamps: readonly (readonly [number, number])[],
  tokenConfidences: readonly number[],
  tokenFrameIndices: readonly number[],
  tokenLogProbs: readonly number[],
  tokenTdtSteps: readonly number[],
): Pick<NemoTdtNativeTranscript, 'words' | 'tokens'> {
  const rawTokens = tokenizer.idsToTokens(tokenIds);
  const tokens: NemoTdtNativeToken[] = [];
  const words: Array<{
    index: number;
    text: string;
    startTime: number;
    endTime: number;
    confidence?: number;
  }> = [];

  let activeWord:
    | {
        index: number;
        parts: string[];
        startTime: number;
        endTime: number;
        confidences: number[];
      }
    | undefined;

  for (let index = 0; index < tokenIds.length; index += 1) {
    const rawToken = rawTokens[index] ?? '';
    const tokenText = rawToken.replace(/\u2581/g, ' ');
    const isWordStart = rawToken.startsWith('\u2581') || !activeWord;
    const [startTime, endTime] = tokenTimestamps[index] ?? [0, 0];
    const confidence = tokenConfidences[index];

    tokens.push({
      index,
      id: tokenIds[index],
      text: tokenText.trim().length > 0 ? tokenText.trim() : tokenText,
      rawText: rawToken,
      isWordStart,
      startTime,
      endTime,
      confidence,
      frameIndex: tokenFrameIndices[index],
      logProb: tokenLogProbs[index],
      tdtStep: tokenTdtSteps[index],
    });

    if (isWordStart) {
      if (activeWord) {
        words.push({
          index: activeWord.index,
          text: activeWord.parts.join('').trim(),
          startTime: activeWord.startTime,
          endTime: activeWord.endTime,
          confidence:
            activeWord.confidences.length > 0
              ? activeWord.confidences.reduce((sum, value) => sum + value, 0) /
                activeWord.confidences.length
              : undefined,
        });
      }

      activeWord = {
        index: words.length,
        parts: [tokenText],
        startTime,
        endTime,
        confidences: isFiniteNumber(confidence) ? [confidence] : [],
      };
      continue;
    }

    if (!activeWord) {
      activeWord = {
        index: words.length,
        parts: [tokenText],
        startTime,
        endTime,
        confidences: isFiniteNumber(confidence) ? [confidence] : [],
      };
      continue;
    }

    activeWord.parts.push(tokenText);
    activeWord.endTime = endTime;
    if (isFiniteNumber(confidence)) {
      activeWord.confidences.push(confidence);
    }
  }

  if (activeWord) {
    words.push({
      index: activeWord.index,
      text: activeWord.parts.join('').trim(),
      startTime: activeWord.startTime,
      endTime: activeWord.endTime,
      confidence:
        activeWord.confidences.length > 0
          ? activeWord.confidences.reduce((sum, value) => sum + value, 0) /
            activeWord.confidences.length
          : undefined,
    });
  }

  return {
    words,
    tokens,
  };
}

export function buildEmptyTranscript(
  warnings: readonly TranscriptWarning[],
): NemoTdtNativeTranscript {
  return {
    utteranceText: '',
    isFinal: true,
    words: [],
    tokens: [],
    warnings,
  };
}
