import type {
  ModelClassification,
  TranscriptMeta,
  TranscriptResult,
  TranscriptToken,
  TranscriptWord,
} from '../../types/index.js';
import type { LasrCtcNativeTranscript, LasrCtcTranscriptionOptions } from './types.js';

export function mapLasrCtcNativeToCanonical(
  nativeTranscript: LasrCtcNativeTranscript,
  classification: ModelClassification,
  meta: Omit<TranscriptMeta, 'detailLevel' | 'isFinal'> & {
    readonly detailLevel: LasrCtcTranscriptionOptions['detail'];
  },
): TranscriptResult {
  const detail = meta.detailLevel ?? 'segments';
  const tokens: TranscriptToken[] = (nativeTranscript.tokens ?? []).map((token) => ({
    index: token.index,
    id: token.id,
    text: token.text,
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
  const segmentStart =
    words.length > 0 ? words[0]?.startTime : tokens.length > 0 ? tokens[0]?.startTime : undefined;
  const segmentEnd =
    words.length > 0
      ? words[words.length - 1]?.endTime
      : tokens.length > 0
        ? tokens[tokens.length - 1]?.endTime
        : undefined;
  const segments =
    segmentStart !== undefined && segmentEnd !== undefined
      ? [
          {
            index: 0,
            text: nativeTranscript.utteranceText,
            startTime: segmentStart,
            endTime: segmentEnd,
            confidence: nativeTranscript.confidence?.utterance,
            wordIndices: words.length > 0 ? words.map((word) => word.index) : undefined,
          },
        ]
      : undefined;

  const result: TranscriptResult = {
    text: nativeTranscript.utteranceText,
    warnings: (nativeTranscript.warnings ?? []).map((warning) => ({
      code: warning.code,
      message: warning.message,
      recoverable: true,
    })),
    meta: {
      ...meta,
      detailLevel: detail,
      isFinal: nativeTranscript.isFinal,
      modelFamily: classification.family ?? 'lasr-ctc',
      tokenCount: tokens.length || undefined,
      wordCount: words.length || undefined,
      segmentCount: segments?.length,
      averageConfidence: nativeTranscript.confidence?.utterance,
      averageWordConfidence: nativeTranscript.confidence?.wordAverage,
      averageTokenConfidence: nativeTranscript.confidence?.tokenAverage,
      nativeAvailable: true,
      metrics: nativeTranscript.metrics ?? meta.metrics,
    },
  };

  if (detail !== 'text' && segments) {
    Object.assign(result, { segments });
  }
  if ((detail === 'words' || detail === 'detailed') && words.length > 0) {
    Object.assign(result, { words });
  }
  if (detail === 'detailed' && tokens.length > 0) {
    Object.assign(result, { tokens });
  }

  return result;
}
