import type {
  ModelClassification,
  TranscriptMeta,
  TranscriptResult,
  TranscriptSegment,
  TranscriptToken,
  TranscriptWord,
} from '../../types/index.js';
import type { Wav2Vec2NativeTranscript, Wav2Vec2TranscriptionOptions } from './types.js';

function average(values: readonly (number | undefined)[]): number | undefined {
  const finite = values.filter((value): value is number => Number.isFinite(value));
  if (finite.length === 0) {
    return undefined;
  }
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function tokenIndicesWithinWord(
  tokens: readonly TranscriptToken[],
  word: { readonly startTime: number; readonly endTime: number },
): readonly number[] | undefined {
  const indices: number[] = [];
  for (const token of tokens) {
    if (
      token.startTime !== undefined &&
      token.endTime !== undefined &&
      token.startTime >= word.startTime &&
      token.endTime <= word.endTime
    ) {
      indices.push(token.index);
    }
  }
  return indices.length > 0 ? indices : undefined;
}

function wordIndicesWithinSegment(
  words: readonly TranscriptWord[],
  segment: { readonly startTime: number; readonly endTime: number },
): readonly number[] | undefined {
  const indices: number[] = [];
  for (const word of words) {
    if (word.startTime >= segment.startTime && word.endTime <= segment.endTime) {
      indices.push(word.index);
    }
  }
  return indices.length > 0 ? indices : undefined;
}

export function mapWav2Vec2NativeToCanonical(
  nativeTranscript: Wav2Vec2NativeTranscript,
  classification: ModelClassification,
  meta: Omit<TranscriptMeta, 'detailLevel' | 'isFinal'> & {
    readonly detailLevel: Wav2Vec2TranscriptionOptions['detail'];
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
    tokenIndices: word.tokenIndices ?? tokenIndicesWithinWord(tokens, word),
  }));
  const segments: TranscriptSegment[] = (nativeTranscript.segments ?? []).map((segment) => ({
    index: segment.index,
    text: segment.text,
    startTime: segment.startTime,
    endTime: segment.endTime,
    confidence: segment.confidence,
    wordIndices: wordIndicesWithinSegment(words, segment),
  }));

  const averageTokenConfidence = average(tokens.map((token) => token.confidence));
  const averageWordConfidence = average(words.map((word) => word.confidence));
  const averageSegmentConfidence = average(segments.map((segment) => segment.confidence));

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
      language: nativeTranscript.language ?? meta.language,
      modelFamily: classification.family ?? 'wav2vec2',
      tokenCount: tokens.length || undefined,
      wordCount: words.length || undefined,
      segmentCount: segments.length || undefined,
      averageConfidence: averageSegmentConfidence ?? averageWordConfidence ?? averageTokenConfidence,
      averageSegmentConfidence,
      averageWordConfidence,
      averageTokenConfidence,
      nativeAvailable: true,
    },
  };

  if (detail !== 'text' && segments.length > 0) {
    Object.assign(result, { segments });
  }
  if ((detail === 'words' || detail === 'sentences+words' || detail === 'detailed') && words.length > 0) {
    Object.assign(result, { words });
  }
  if (detail === 'detailed' && tokens.length > 0) {
    Object.assign(result, { tokens });
  }

  return result;
}
