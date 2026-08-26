import type { ModelClassification, TranscriptMeta, TranscriptResult, TranscriptToken } from '../../types/index.js';
import type { SenseVoiceNativeTranscript, SenseVoiceTranscriptionOptions } from './types.js';

export function mapSenseVoiceNativeToCanonical(
  native: SenseVoiceNativeTranscript,
  classification: ModelClassification,
  meta: Omit<TranscriptMeta, 'detailLevel' | 'isFinal'> & {
    readonly detailLevel: SenseVoiceTranscriptionOptions['detail'];
  },
): TranscriptResult {
  const detail = meta.detailLevel ?? 'segments';
  const tokens: TranscriptToken[] = (native.tokens ?? []).map((token) => ({
    index: token.index,
    id: token.id,
    text: token.text,
    startTime: token.startTime,
    endTime: token.endTime,
    confidence: token.confidence,
  }));
  const first = tokens[0];
  const last = tokens[tokens.length - 1];
  const segments =
    first?.startTime !== undefined && last?.endTime !== undefined
      ? [{
          index: 0,
          text: native.utteranceText,
          startTime: first.startTime,
          endTime: last.endTime,
          confidence: native.confidence?.utterance,
          wordIndices: undefined,
        }]
      : undefined;
  const result: TranscriptResult = {
    text: native.utteranceText,
    warnings: (native.warnings ?? []).map((warning) => ({
      code: warning.code,
      message: warning.message,
      recoverable: true,
    })),
    meta: {
      ...meta,
      detailLevel: detail,
      isFinal: native.isFinal,
      modelFamily: classification.family ?? 'sensevoice',
      language: native.language,
      tokenCount: tokens.length || undefined,
      segmentCount: segments?.length,
      averageConfidence: native.confidence?.utterance,
      averageTokenConfidence: native.confidence?.tokenAverage,
      nativeAvailable: true,
      backendNotes: [
        native.metadata?.emotion ? `emotion:${native.metadata.emotion}` : '',
        native.metadata?.event ? `event:${native.metadata.event}` : '',
      ].filter(Boolean),
      metrics: native.metrics ?? meta.metrics,
    },
  };
  if (detail !== 'text' && segments) Object.assign(result, { segments });
  if (detail === 'detailed' && tokens.length > 0) Object.assign(result, { tokens });
  return result;
}
