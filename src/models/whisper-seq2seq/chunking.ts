import type { WhisperNativeSegment, WhisperNativeToken, WhisperNativeTranscript, WhisperNativeWord } from './types.js';

export interface WhisperChunkTranscriptInput {
  readonly chunkStartTime: number;
  readonly transcript: WhisperNativeTranscript;
}

export function mergeWhisperChunkTranscripts(chunks: readonly WhisperChunkTranscriptInput[]): WhisperNativeTranscript {
  const segments: WhisperNativeSegment[] = [];
  const words: WhisperNativeWord[] = [];
  const tokens: WhisperNativeToken[] = [];
  const warnings = chunks.flatMap((chunk) => chunk.transcript.warnings ?? []);
  const language = chunks.find((chunk) => chunk.transcript.language)?.transcript.language;

  for (const chunk of chunks) {
    const offset = chunk.chunkStartTime;
    for (const segment of chunk.transcript.segments ?? []) {
      segments.push({
        ...segment,
        index: segments.length,
        startTime: offsetTime(segment.startTime, offset),
        endTime: offsetTime(segment.endTime, offset),
      });
    }
    for (const word of chunk.transcript.words ?? []) {
      words.push({
        ...word,
        index: words.length,
        startTime: offsetTime(word.startTime, offset),
        endTime: offsetTime(word.endTime, offset),
      });
    }
    for (const token of chunk.transcript.tokens ?? []) {
      tokens.push({
        ...token,
        index: tokens.length,
        startTime: token.startTime === undefined ? undefined : offsetTime(token.startTime, offset),
        endTime: token.endTime === undefined ? undefined : offsetTime(token.endTime, offset),
      });
    }
  }

  const utteranceText = segments.length > 0
    ? segments.map((segment) => segment.text).join(' ').trim()
    : chunks.map((chunk) => chunk.transcript.utteranceText).filter(Boolean).join(' ').trim();

  return {
    utteranceText,
    isFinal: chunks.every((chunk) => chunk.transcript.isFinal),
    ...(language ? { language } : {}),
    ...(segments.length > 0 ? { segments } : {}),
    ...(words.length > 0 ? { words } : {}),
    ...(tokens.length > 0 ? { tokens } : {}),
    ...(warnings.length > 0 ? { warnings } : {}),
  };
}

function offsetTime(time: number, offset: number): number {
  return Math.round((time + offset) * 1000) / 1000;
}
