import { roundTimestampSeconds } from '../../runtime/timing.js';
import type { TranscriptWarning } from '../../types/index.js';
import { ParakeetTokenizer } from '../nemo-tdt/tokenizer.js';
import type {
  NemoRnntNativeSpecialToken,
  NemoRnntNativeToken,
  NemoRnntNativeTranscript,
} from './types.js';

function isFiniteNumber(value: number | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function classifySpecialToken(
  tokenId: number,
  options: {
    readonly eouTokenId?: number;
    readonly eobTokenId?: number;
  },
): NemoRnntNativeSpecialToken['kind'] {
  if (tokenId === options.eouTokenId) {
    return 'eou';
  }
  if (tokenId === options.eobTokenId) {
    return 'eob';
  }
  return 'control';
}

interface ActiveWordState {
  index: number;
  parts: string[];
  startTime: number;
  endTime: number;
  confidences: number[];
}

function calculateWordConfidence(confidences: number[]): number | undefined {
  if (confidences.length === 0) {
    return undefined;
  }
  return confidences.reduce((sum, value) => sum + value, 0) / confidences.length;
}

function flushActiveWord(
  activeWord: ActiveWordState,
  words: Array<{
    index: number;
    text: string;
    startTime: number;
    endTime: number;
    confidence?: number;
  }>,
): void {
  words.push({
    index: activeWord.index,
    text: activeWord.parts.join('').trim(),
    startTime: activeWord.startTime,
    endTime: activeWord.endTime,
    confidence: calculateWordConfidence(activeWord.confidences),
  });
}

function createActiveWord(
  index: number,
  startTime: number,
  tokenText: string,
  confidence?: number,
): ActiveWordState {
  return {
    index,
    parts: [tokenText],
    startTime,
    endTime: startTime,
    confidences: isFiniteNumber(confidence) ? [confidence] : [],
  };
}

export function buildRnntTranscriptDetails(
  tokenizer: ParakeetTokenizer,
  tokenIds: readonly number[],
  tokenFrameIndices: readonly number[],
  tokenConfidences: readonly number[],
  tokenLogProbs: readonly number[],
  options: {
    readonly frameTimeSeconds: number;
    readonly eouTokenId?: number;
    readonly eobTokenId?: number;
  },
): Pick<
  NemoRnntNativeTranscript,
  'rawUtteranceText' | 'tokens' | 'specialTokens' | 'words' | 'control'
> & {
  readonly utteranceText: string;
} {
  const rawTokens = tokenizer.idsToTokens(tokenIds);
  const tokens: NemoRnntNativeToken[] = [];
  const specialTokens: NemoRnntNativeSpecialToken[] = [];
  const words: Array<{
    index: number;
    text: string;
    startTime: number;
    endTime: number;
    confidence?: number;
  }> = [];

  let activeWord: ActiveWordState | undefined;

  for (let index = 0; index < tokenIds.length; index += 1) {
    const tokenId = tokenIds[index];
    if (tokenId === undefined) {
      continue;
    }

    const rawToken = rawTokens[index] ?? '';
    const frameIndex = tokenFrameIndices[index] ?? 0;
    const time = roundTimestampSeconds(frameIndex * options.frameTimeSeconds);
    const confidence = tokenConfidences[index];
    const logProb = tokenLogProbs[index];

    if (tokenizer.isControlTokenId(tokenId)) {
      specialTokens.push({
        index: specialTokens.length,
        id: tokenId,
        text: rawToken,
        rawText: rawToken,
        isWordStart: true,
        startTime: time,
        endTime: time,
        confidence,
        frameIndex,
        logProb,
        kind: classifySpecialToken(tokenId, options),
      });
      continue;
    }

    const tokenText = rawToken.replace(/\u2581/g, ' ');
    const isWordStart = rawToken.startsWith('\u2581') || !activeWord;

    tokens.push({
      index: tokens.length,
      id: tokenId,
      text: tokenText.trim().length > 0 ? tokenText.trim() : tokenText,
      rawText: rawToken,
      isWordStart,
      startTime: time,
      endTime: time,
      confidence,
      frameIndex,
      logProb,
    });

    if (isWordStart) {
      if (activeWord) {
        flushActiveWord(activeWord, words);
      }

      activeWord = createActiveWord(words.length, time, tokenText, confidence);
      continue;
    }

    if (!activeWord) {
      activeWord = createActiveWord(words.length, time, tokenText, confidence);
      continue;
    }

    activeWord.parts.push(tokenText);
    activeWord.endTime = time;
    if (isFiniteNumber(confidence)) {
      activeWord.confidences.push(confidence);
    }
  }

  if (activeWord) {
    flushActiveWord(activeWord, words);
  }

  return {
    utteranceText: tokenizer.decode(tokenIds, { skipControlTokens: true }),
    rawUtteranceText: tokenizer.decode(tokenIds),
    words,
    tokens,
    specialTokens,
    control: {
      containsEou: specialTokens.some((token) => token.kind === 'eou'),
      containsEob: specialTokens.some((token) => token.kind === 'eob'),
      eouTokenId: options.eouTokenId,
      eobTokenId: options.eobTokenId,
    },
  };
}

export function buildEmptyTranscript(
  warnings: readonly TranscriptWarning[],
): NemoRnntNativeTranscript {
  return {
    utteranceText: '',
    rawUtteranceText: '',
    isFinal: true,
    words: [],
    tokens: [],
    specialTokens: [],
    warnings,
    control: {
      containsEou: false,
      containsEob: false,
    },
  };
}
