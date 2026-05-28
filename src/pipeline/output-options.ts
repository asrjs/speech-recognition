import type { BaseTranscriptionOptions, TranscriptDetailLevel } from '../types/index.js';

export function resolveTranscriptDetail(
  options: Pick<
    BaseTranscriptionOptions,
    'detail' | 'returnTimestamps' | 'returnWords' | 'returnTokens'
  > = {},
): TranscriptDetailLevel | undefined {
  if (options.detail) {
    return options.detail;
  }
  if (options.returnTokens) {
    return 'detailed';
  }
  if (options.returnWords || options.returnTimestamps === 'word') {
    return 'words';
  }
  if (options.returnTimestamps === 'sentence' || options.returnTimestamps === 'sentences') {
    return 'sentences';
  }
  if (
    options.returnTimestamps === true ||
    options.returnTimestamps === 'segment' ||
    options.returnTimestamps === 'segments'
  ) {
    return 'segments';
  }
  return undefined;
}

export function withResolvedTranscriptDetail<TOptions extends BaseTranscriptionOptions>(
  options: TOptions | undefined,
): TOptions | undefined {
  const detail = resolveTranscriptDetail(options);
  if (!detail || options?.detail === detail) {
    return options;
  }
  return { ...(options ?? {}), detail } as TOptions;
}
