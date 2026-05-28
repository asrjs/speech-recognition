import type {
  ModelClassification,
  TranscriptMeta,
  TranscriptResult,
  TranscriptToken,
  TranscriptWord,
} from '../../types/index.js';
import type { WhisperNativeTranscript, WhisperSeq2SeqTranscriptionOptions } from './types.js';

export function mapWhisperNativeToCanonical(
  nativeTranscript: WhisperNativeTranscript,
  classification: ModelClassification,
  meta: Omit<TranscriptMeta, 'detailLevel' | 'isFinal'> & {
    readonly detailLevel: WhisperSeq2SeqTranscriptionOptions['detail'];
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
    tokenIndices: word.tokenIndices,
  }));

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
      modelFamily: classification.family ?? 'whisper',
      tokenCount: tokens.length || undefined,
      wordCount: words.length || undefined,
      segmentCount: nativeTranscript.segments?.length,
      nativeAvailable: true,
      language: nativeTranscript.language ?? meta.language,
    },
  };

  if (detail !== 'text' && nativeTranscript.segments && nativeTranscript.segments.length > 0) {
    Object.assign(result, { segments: nativeTranscript.segments });
  }
  if ((detail === 'words' || detail === 'detailed') && words.length > 0) {
    Object.assign(result, { words });
  }
  if (detail === 'detailed' && tokens.length > 0) {
    Object.assign(result, { tokens });
  }

  return result;
}
