import { partitionWordsIntoSentences } from '../../pipeline/index.js';
import type {
  ModelClassification,
  TranscriptDetailLevel,
  TranscriptMeta,
  TranscriptResult,
  TranscriptSentence,
  TranscriptSegment,
  TranscriptToken,
  TranscriptWord,
  TranscriptWarning,
} from '../../types/index.js';
import type {
  NemoConfidenceReconstructor,
  NemoNativeToken,
  NemoNativeTranscript,
  NemoTimestampReconstructor,
} from './types.js';

function buildDefaultSentences(words: readonly TranscriptWord[]): TranscriptSentence[] {
  return partitionWordsIntoSentences(words);
}

function buildDefaultSegments(words: readonly TranscriptWord[]): TranscriptSegment[] {
  return buildDefaultSentences(words);
}

function mapNativeToken(token: NemoNativeToken): TranscriptToken {
  const extended = token as {
    readonly frameIndex?: number;
    readonly logProb?: number;
    readonly tdtStep?: number;
  };
  return {
    index: token.index,
    id: token.id,
    text: token.text,
    rawText: token.rawText,
    isWordStart: token.isWordStart,
    startTime: token.startTime,
    endTime: token.endTime,
    confidence: token.confidence,
    frameIndex: extended.frameIndex,
    logProb: extended.logProb,
    tdtStep: extended.tdtStep,
  };
}

export const defaultNemoTimestampReconstructor: NemoTimestampReconstructor = {
  reconstruct(nativeTranscript, detail) {
    const tokens: TranscriptToken[] = (nativeTranscript.tokens ?? []).map((token) =>
      mapNativeToken(token),
    );

    const words: TranscriptWord[] = (nativeTranscript.words ?? []).map((word) => {
      const tokenIndices: number[] = [];
      for (const token of tokens) {
        if (
          token.startTime !== undefined &&
          token.endTime !== undefined &&
          token.startTime >= word.startTime &&
          token.endTime <= word.endTime
        ) {
          tokenIndices.push(token.index);
        }
      }
      return {
        index: word.index,
        text: word.text,
        startTime: word.startTime,
        endTime: word.endTime,
        confidence: word.confidence,
        tokenIndices,
      };
    });

    const sentences = buildDefaultSentences(words);
    const segments = buildDefaultSegments(words);

    if (detail === 'text') {
      return {};
    }
    if (detail === 'sentences') {
      return { sentences };
    }
    if (detail === 'segments') {
      return { segments };
    }
    if (detail === 'words') {
      return {
        segments,
        sentences,
        words,
      };
    }
    if (detail === 'sentences+words') {
      return {
        sentences,
        words,
      };
    }

    return {
      segments,
      sentences,
      words,
      tokens,
    };
  },
};

export const defaultNemoConfidenceReconstructor: NemoConfidenceReconstructor = {
  summarize(nativeTranscript) {
    return {
      averageConfidence: nativeTranscript.confidence?.utterance,
      averageWordConfidence: nativeTranscript.confidence?.wordAverage,
      averageTokenConfidence: nativeTranscript.confidence?.tokenAverage,
    };
  },
};

function mapWarnings(nativeTranscript: NemoNativeTranscript): TranscriptWarning[] {
  return (nativeTranscript.warnings ?? []).map((warning) => ({
    code: warning.code,
    message: warning.message,
    recoverable: true,
  }));
}

function resolveCanonicalFamily(classification: ModelClassification): string {
  return (
    classification.family ?? `${classification.ecosystem}-${classification.decoder ?? 'model'}`
  );
}

export function mapNemoNativeToCanonical(
  nativeTranscript: NemoNativeTranscript,
  classification: ModelClassification,
  meta: Omit<TranscriptMeta, 'detailLevel' | 'isFinal'> & {
    readonly detailLevel: TranscriptDetailLevel;
  },
  timestampReconstructor: NemoTimestampReconstructor = defaultNemoTimestampReconstructor,
  confidenceReconstructor: NemoConfidenceReconstructor = defaultNemoConfidenceReconstructor,
): TranscriptResult {
  const detail = meta.detailLevel;
  const warnings = mapWarnings(nativeTranscript);
  const reconstructed = timestampReconstructor.reconstruct(nativeTranscript, detail);
  const confidence = confidenceReconstructor.summarize(nativeTranscript);

  const canonicalMeta: TranscriptMeta = {
    ...meta,
    ...confidence,
    detailLevel: detail,
    isFinal: nativeTranscript.isFinal,
    modelFamily: resolveCanonicalFamily(classification),
    tokenCount: reconstructed.tokens?.length,
    wordCount: reconstructed.words?.length,
    sentenceCount: reconstructed.sentences?.length,
    segmentCount: reconstructed.segments?.length,
    nativeAvailable: true,
  };

  const result: TranscriptResult = {
    text: nativeTranscript.utteranceText,
    warnings,
    meta: canonicalMeta,
  };

  if (reconstructed.segments && reconstructed.segments.length > 0) {
    Object.assign(result, { segments: reconstructed.segments });
  }
  if (reconstructed.sentences && reconstructed.sentences.length > 0) {
    Object.assign(result, { sentences: reconstructed.sentences });
  }
  if (reconstructed.words && reconstructed.words.length > 0) {
    Object.assign(result, { words: reconstructed.words });
  }
  if (reconstructed.tokens && reconstructed.tokens.length > 0) {
    Object.assign(result, { tokens: reconstructed.tokens });
  }

  return result;
}
