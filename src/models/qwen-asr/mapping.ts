import type {
  ModelClassification,
  TranscriptMeta,
  TranscriptResult,
  TranscriptToken,
} from '../../types/index.js';
import type { Qwen3AsrNativeTranscript, Qwen3AsrTranscriptionOptions } from './types.js';

export function mapQwen3AsrNativeToCanonical(
  nativeTranscript: Qwen3AsrNativeTranscript,
  classification: ModelClassification,
  meta: Omit<TranscriptMeta, 'detailLevel' | 'isFinal'> & {
    readonly detailLevel?: Qwen3AsrTranscriptionOptions['detail'];
  },
): TranscriptResult {
  const detail = meta.detailLevel ?? 'segments';
  const tokens: TranscriptToken[] = (nativeTranscript.tokens ?? []).map((token) => ({
    index: token.index,
    id: token.id,
    text: token.text,
  }));
  const result: TranscriptResult = {
    text: nativeTranscript.utteranceText,
    warnings: (nativeTranscript.warnings ?? []).map((warning) => ({
      code: warning.code,
      message: warning.message,
      recoverable: warning.recoverable ?? true,
    })),
    meta: {
      ...meta,
      detailLevel: detail,
      isFinal: nativeTranscript.isFinal,
      modelFamily: classification.family ?? 'qwen-asr',
      language: nativeTranscript.language ?? meta.language,
      tokenCount: tokens.length || undefined,
      segmentCount: nativeTranscript.segments?.length,
      nativeAvailable: true,
    },
  };
  if (detail !== 'text' && nativeTranscript.segments && nativeTranscript.segments.length > 0) {
    Object.assign(result, { segments: nativeTranscript.segments });
  }
  if (detail === 'detailed' && tokens.length > 0) {
    Object.assign(result, { tokens });
  }
  return result;
}
