import type { TranscriptWarning } from '../../types/index.js';
import type { ParakeetTokenizer } from '../nemo-tdt/tokenizer.js';
import type {
  NemotronRnntNativeSpecialToken,
  NemotronRnntNativeToken,
  NemotronRnntNativeTranscript,
  NemotronRnntNativeWord,
} from './types.js';

const LANG_SEGMENT_REGEX = /^<(en-US|tr-TR)>$/;

function isFiniteNumber(value: number | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

/**
 * Walks the per-frame blank-dominant output to fold blank-frames into the
 * neighboring non-blank emission's confidence statistics. Used by the
 * streaming executor to expose per-frame confidences alongside per-token
 * confidences.
 */
export function aggregateNemotronRnntFrameConfidences(
  frameConfidenceStats: Map<number, { sum: number; count: number }>,
): number[] {
  if (frameConfidenceStats.size === 0) {
    return [];
  }
  const maxFrame = Math.max(...frameConfidenceStats.keys());
  const result: number[] = new Array(maxFrame + 1);
  for (let frame = 0; frame <= maxFrame; frame += 1) {
    const entry = frameConfidenceStats.get(frame);
    result[frame] = entry ? entry.sum / entry.count : 0;
  }
  return result;
}

/**
 * Re-uses the nemo-rnnt transcript-details builder so the Nemotron
 * executor inherits the same word/timestamp reconstruction, then maps
 * the `<en-US>`/`<tr-TR>` control tokens to the Nemotron-specific
 * `lang-segment` kind so downstream consumers can split multi-language
 * streams.
 */
export function buildNemotronRnntTranscriptDetails(
  tokenizer: ParakeetTokenizer,
  tokenIds: readonly number[],
  tokenFrameIndices: readonly number[],
  tokenConfidences: readonly number[],
  tokenLogProbs: readonly number[],
  options: {
    readonly frameTimeSeconds: number;
  },
): {
  readonly utteranceText: string;
  readonly rawUtteranceText: string;
  readonly tokens: readonly NemotronRnntNativeToken[];
  readonly specialTokens: readonly NemotronRnntNativeSpecialToken[];
  readonly words: ReadonlyArray<NemotronRnntNativeWord>;
} {
  const rawTokens = tokenizer.idsToTokens(tokenIds);
  const tokens: NemotronRnntNativeToken[] = [];
  const specialTokens: NemotronRnntNativeSpecialToken[] = [];
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
    const tokenId = tokenIds[index];
    if (tokenId === undefined) {
      continue;
    }

    const rawToken = rawTokens[index] ?? '';
    const frameIndex = tokenFrameIndices[index] ?? 0;
    const startTime = frameIndex * options.frameTimeSeconds;
    const confidence = tokenConfidences[index];
    const logProb = tokenLogProbs[index];

    const isControl = tokenizer.isControlTokenId(tokenId);
    const tokenText = rawToken.replace(/\u2581/g, ' ');

    if (isControl) {
      const langMatch = tokenText.match(LANG_SEGMENT_REGEX);
      const kind: NemotronRnntNativeSpecialToken['kind'] = langMatch
        ? 'lang-segment'
        : 'control';

      if (activeWord) {
        words.push({
          index: words.length,
          text: activeWord.parts.join('').trim(),
          startTime: activeWord.startTime,
          endTime: activeWord.endTime,
          confidence:
            activeWord.confidences.length > 0
              ? activeWord.confidences.reduce((a, b) => a + b, 0) /
                activeWord.confidences.length
              : undefined,
        });
        activeWord = undefined;
      }

      specialTokens.push({
        index: specialTokens.length,
        id: tokenId,
        kind,
        text: tokenText.trim(),
        rawText: tokenText,
        frameIndex,
        confidence,
        logProb,
        startTime,
        endTime: startTime,
      });
      continue;
    }

    tokens.push({
      index: tokens.length,
      id: tokenId,
      text: tokenText.replace(/^\s+/, ''),
      rawText: tokenText,
      isWordStart: tokenText.startsWith(' '),
      startTime,
      endTime: startTime + options.frameTimeSeconds,
      confidence,
      logProb,
      frameIndex,
    });

    if (tokenText.startsWith(' ')) {
      if (activeWord) {
        words.push({
          index: words.length,
          text: activeWord.parts.join('').trim(),
          startTime: activeWord.startTime,
          endTime: activeWord.endTime,
          confidence:
            activeWord.confidences.length > 0
              ? activeWord.confidences.reduce((a, b) => a + b, 0) /
                activeWord.confidences.length
              : undefined,
        });
      }
      activeWord = {
        index: words.length,
        parts: [tokenText.replace(/^\s+/, '')],
        startTime,
        endTime: startTime + options.frameTimeSeconds,
        confidences: isFiniteNumber(confidence) ? [confidence] : [],
      };
    } else if (activeWord) {
      activeWord.parts.push(tokenText);
      activeWord.endTime = startTime + options.frameTimeSeconds;
      if (isFiniteNumber(confidence)) {
        activeWord.confidences.push(confidence);
      }
    } else {
      activeWord = {
        index: words.length,
        parts: [tokenText],
        startTime,
        endTime: startTime + options.frameTimeSeconds,
        confidences: isFiniteNumber(confidence) ? [confidence] : [],
      };
    }
  }

  if (activeWord) {
    words.push({
      index: words.length,
      text: activeWord.parts.join('').trim(),
      startTime: activeWord.startTime,
      endTime: activeWord.endTime,
      confidence:
        activeWord.confidences.length > 0
          ? activeWord.confidences.reduce((a, b) => a + b, 0) /
            activeWord.confidences.length
          : undefined,
    });
  }

  const rawUtteranceText = tokens.map((t) => t.rawText ?? '').join('');
  const utteranceText = rawUtteranceText.replace(/\s+/g, ' ').trim();

  return { utteranceText, rawUtteranceText, tokens, specialTokens, words };
}

/**
 * Wraps a built transcript with the Nemotron-specific control summary
 * so consumers can detect multi-language outputs without scanning the
 * special-token stream.
 */
export function withNemotronRnntControl(
  transcript: NemotronRnntNativeTranscript,
): NemotronRnntNativeTranscript {
  const langSegmentIds: number[] = [];
  for (const special of transcript.specialTokens ?? []) {
    if (special.kind === 'lang-segment' && typeof special.id === 'number') {
      langSegmentIds.push(special.id);
    }
  }

  return {
    ...transcript,
    control: {
      containsLangSegment: langSegmentIds.length > 0,
      langSegmentTokenIds: langSegmentIds.length > 0 ? langSegmentIds : undefined,
    },
  };
}

/**
 * Empty placeholder returned when the executor runs on silent audio or
 * short input that produces no encodable features.
 */
export function buildEmptyNemotronRnntTranscript(
  warnings: readonly TranscriptWarning[],
): NemotronRnntNativeTranscript {
  return {
    utteranceText: '',
    isFinal: true,
    tokens: [],
    specialTokens: [],
    control: { containsLangSegment: false },
    confidence: {},
    metrics: {
      emittedTokenCount: 0,
      emittedWordCount: 0,
      encoderFrameCount: 0,
      decodeIterations: 0,
    },
    warnings,
    debug: {},
  };
}
