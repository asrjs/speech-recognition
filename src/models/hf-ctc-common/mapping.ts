import type {
  ModelClassification,
  TranscriptMeta,
  TranscriptResult,
  TranscriptToken,
  TranscriptWord,
} from '../../types/index.js';
import type { HfCtcNativeTranscript, HfCtcTranscriptionOptions } from './types.js';

export function mapHfCtcNativeToCanonical(
  nativeTranscript: HfCtcNativeTranscript,
  classification: ModelClassification,
  meta: Omit<TranscriptMeta, 'detailLevel' | 'isFinal'> & {
    readonly detailLevel: HfCtcTranscriptionOptions['detail'];
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
  const segments =
    words.length > 0
      ? [
          {
            index: 0,
            text: nativeTranscript.utteranceText,
            startTime: words[0]!.startTime,
            endTime: words[words.length - 1]!.endTime,
            confidence: nativeTranscript.confidence?.utterance,
            wordIndices: words.map((word) => word.index),
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
      modelFamily: classification.family ?? 'hf-ctc',
      tokenCount: tokens.length || undefined,
      wordCount: words.length || undefined,
      segmentCount: segments?.length,
      averageConfidence: nativeTranscript.confidence?.utterance,
      averageWordConfidence: nativeTranscript.confidence?.wordAverage,
      averageTokenConfidence: nativeTranscript.confidence?.tokenAverage,
      nativeAvailable: true,
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
