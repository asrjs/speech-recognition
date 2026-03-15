import type {
  ModelClassification,
  TranscriptDetailLevel,
  TranscriptMeta,
  TranscriptResult,
  TranscriptSegment,
  TranscriptToken,
  TranscriptWord,
  TranscriptWarning,
} from '../../types/index.js';
import type {
  NemoConfidenceReconstructor,
  NemoNativeTranscript,
  NemoTimestampReconstructor,
} from './types.js';

function buildDefaultSegments(words: readonly TranscriptWord[]): TranscriptSegment[] {
  if (words.length === 0) {
    return [];
  }

  return [
    {
      index: 0,
      text: words
        .map((word) => word.text)
        .join(' ')
        .trim(),
      startTime: words[0]!.startTime,
      endTime: words[words.length - 1]!.endTime,
      confidence: words.every((word) => typeof word.confidence === 'number')
        ? words.reduce((sum, word) => sum + (word.confidence ?? 0), 0) / words.length
        : undefined,
      wordIndices: words.map((word) => word.index),
    },
  ];
}

export const defaultNemoTimestampReconstructor: NemoTimestampReconstructor = {
  reconstruct(nativeTranscript, detail) {
    const tokens: TranscriptToken[] = (nativeTranscript.tokens ?? []).map((token) => ({
      index: token.index,
      id: token.id,
      text: token.text,
      rawText: token.rawText,
      isWordStart: token.isWordStart,
      startTime: token.startTime,
      endTime: token.endTime,
      confidence: token.confidence,
    }));

    const words: TranscriptWord[] = (nativeTranscript.words ?? []).map((word) => ({
      index: word.index,
      text: word.text,
      startTime: word.startTime,
      endTime: word.endTime,
      confidence: word.confidence,
      tokenIndices: tokens
        .filter(
          (token) =>
            token.startTime !== undefined &&
            token.endTime !== undefined &&
            token.startTime >= word.startTime &&
            token.endTime <= word.endTime,
        )
        .map((token) => token.index),
    }));

    const segments = buildDefaultSegments(words);

    if (detail === 'text') {
      return {};
    }
    if (detail === 'segments') {
      return { segments };
    }
    if (detail === 'words') {
      return {
        segments,
        words,
      };
    }

    return {
      segments,
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
  if (reconstructed.words && reconstructed.words.length > 0) {
    Object.assign(result, { words: reconstructed.words });
  }
  if (reconstructed.tokens && reconstructed.tokens.length > 0) {
    Object.assign(result, { tokens: reconstructed.tokens });
  }

  return result;
}
