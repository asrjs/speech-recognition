import { PcmAudioBuffer, normalizePcmInput } from '../audio/index.js';
import type {
  AudioInputLike,
  BaseTranscriptionOptions,
  ModelInferenceLimits,
  TranscriptResult,
  TranscriptSegment,
  TranscriptWarning,
  TranscriptWord,
} from '../types/index.js';
import { joinTranscriptWords, partitionWordsIntoSegments } from './sentence-segmenter.js';
import { resolveWindowPolicy, shouldUseWindowing, type ResolvedWindowPolicy } from './window-policy.js';
import {
  addWindowMetrics,
  buildWindowedMetrics,
  createWindowedMetricsAccumulator,
} from './windowed-metrics.js';

const EPSILON_SECONDS = 1e-6;
const CURSOR_MIN_ADVANCE_SECONDS = 1;
const CURSOR_GAP_THRESHOLD_SECONDS = 0.2;
const CURSOR_SNAP_WINDOW_SECONDS = 0.5;
const SEGMENT_DEDUP_TOLERANCE_SECONDS = 0.15;

export interface WindowedTranscriptionContext<TOptions extends BaseTranscriptionOptions> {
  readonly input: AudioInputLike;
  readonly options?: TOptions;
  readonly inference?: ModelInferenceLimits;
  readonly transcribeWindow: (input: PcmAudioBuffer, options: TOptions) => Promise<TranscriptResult>;
}

export interface WindowedTranscriptionDecision<TOptions extends BaseTranscriptionOptions> {
  readonly shouldWindow: boolean;
  readonly audio: PcmAudioBuffer;
  readonly policy: ResolvedWindowPolicy;
  readonly options: TOptions;
}

function normalizeWordText(text: string | undefined): string {
  return String(text ?? '')
    .normalize('NFKC')
    .toLowerCase()
    .replace(/^[("'“‘\[{]+/g, '')
    .replace(/[.,!?;:)"'”’\]}]+$/g, '')
    .trim();
}

function normalizeRawWordText(text: string | undefined): string {
  return String(text ?? '').normalize('NFKC').toLowerCase().trim();
}

export function dedupeWindowWords(words: readonly TranscriptWord[]): TranscriptWord[] {
  const merged: TranscriptWord[] = [];
  for (const word of words) {
    const prev = merged.at(-1);
    const prevText = normalizeWordText(prev?.text);
    const wordText = normalizeWordText(word.text);
    if (
      prev &&
      prevText === wordText &&
      (prevText.length > 0 || normalizeRawWordText(prev.text) === normalizeRawWordText(word.text)) &&
      word.startTime < prev.endTime
    ) {
      const prevDuration = prev.endTime - prev.startTime;
      const nextDuration = word.endTime - word.startTime;
      if (nextDuration > prevDuration) {
        merged[merged.length - 1] = word;
      }
      continue;
    }
    merged.push(word);
  }
  return merged;
}

function withTimeOffset(result: TranscriptResult, offsetSeconds: number): TranscriptResult {
  if (Math.abs(offsetSeconds) <= EPSILON_SECONDS) {
    return result;
  }

  const words = result.words?.map((word) => ({
    ...word,
    startTime: word.startTime + offsetSeconds,
    endTime: word.endTime + offsetSeconds,
  }));
  const segments = result.segments?.map((segment) => ({
    ...segment,
    startTime: segment.startTime + offsetSeconds,
    endTime: segment.endTime + offsetSeconds,
  }));
  const tokens = result.tokens?.map((token) => ({
    ...token,
    startTime: token.startTime === undefined ? undefined : token.startTime + offsetSeconds,
    endTime: token.endTime === undefined ? undefined : token.endTime + offsetSeconds,
  }));

  return {
    ...result,
    ...(segments ? { segments } : {}),
    ...(words ? { words } : {}),
    ...(tokens ? { tokens } : {}),
  };
}

function mergePendingAndCurrentWords(
  pendingWords: readonly TranscriptWord[],
  currentWords: readonly TranscriptWord[],
): TranscriptWord[] {
  if (pendingWords.length === 0) {
    return dedupeWindowWords(currentWords);
  }
  if (currentWords.length === 0) {
    return dedupeWindowWords(pendingWords);
  }

  const pendingStart = pendingWords[0]!.startTime;
  const currentStart = currentWords[0]!.startTime;
  if (currentStart <= pendingStart + EPSILON_SECONDS) {
    return dedupeWindowWords(currentWords);
  }
  return dedupeWindowWords([...pendingWords, ...currentWords]);
}

function relocateCursorToNearbyGap(targetSeconds: number, words: readonly TranscriptWord[]): number {
  let best = targetSeconds;
  let bestDistance = CURSOR_SNAP_WINDOW_SECONDS + 1;
  for (let i = 0; i < words.length - 1; i += 1) {
    const current = words[i]!;
    const next = words[i + 1]!;
    const gapStart = current.endTime;
    const gapEnd = next.startTime;
    const gap = gapEnd - gapStart;
    if (gap < CURSOR_GAP_THRESHOLD_SECONDS) {
      continue;
    }
    for (const candidate of [gapStart, gapEnd]) {
      if (candidate + EPSILON_SECONDS < targetSeconds) {
        continue;
      }
      const distance = candidate - targetSeconds;
      if (distance <= CURSOR_SNAP_WINDOW_SECONDS && distance < bestDistance) {
        best = candidate;
        bestDistance = distance;
      }
    }
  }
  return best;
}

function normalizeSegmentText(text: string): string {
  return text
    .normalize('NFKC')
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
}

function appendFinalizedSegment(segments: TranscriptSegment[], segment: TranscriptSegment): void {
  const normalized = normalizeSegmentText(segment.text);
  if (!normalized) {
    return;
  }
  const duplicate = segments.some(
    (candidate) =>
      normalizeSegmentText(candidate.text) === normalized &&
      Math.abs(candidate.endTime - segment.endTime) < SEGMENT_DEDUP_TOLERANCE_SECONDS,
  );
  if (!duplicate) {
    segments.push({ ...segment, index: segments.length });
  }
}

function flattenSegmentWords(segments: readonly TranscriptSegment[], words: readonly TranscriptWord[]): TranscriptWord[] {
  const byIndex = new Map(words.map((word) => [word.index, word]));
  const flattened: TranscriptWord[] = [];
  for (const segment of segments) {
    for (const index of segment.wordIndices ?? []) {
      const word = byIndex.get(index);
      if (word) {
        flattened.push(word);
      }
    }
  }
  return flattened;
}

function addWarning(warnings: readonly TranscriptWarning[], code: string, message: string): TranscriptWarning[] {
  return [...warnings, { code, message, recoverable: true }];
}

export function planWindowedTranscription<TOptions extends BaseTranscriptionOptions>(
  input: AudioInputLike,
  options: TOptions | undefined,
  inference: WindowedTranscriptionContext<TOptions>['inference'],
): WindowedTranscriptionDecision<TOptions> {
  const policy = resolveWindowPolicy({ ...(options ?? {}), inference });
  const audio = normalizePcmInput(input, { sampleRate: policy.sampleRate }).toMono();
  const shouldWindow = shouldUseWindowing(audio.durationSeconds, policy);
  return {
    shouldWindow,
    audio,
    policy,
    options: { ...(options ?? {}), detail: options?.detail ?? 'words' } as TOptions,
  };
}

export async function transcribeWithWindowing<TOptions extends BaseTranscriptionOptions>(
  context: WindowedTranscriptionContext<TOptions>,
): Promise<TranscriptResult> {
  const decision = planWindowedTranscription(context.input, context.options, context.inference);
  if (!decision.shouldWindow) {
    return context.transcribeWindow(decision.audio, decision.options);
  }

  const { audio, policy } = decision;
  const accumulator = createWindowedMetricsAccumulator(audio.durationSeconds);
  const finalizedSegments: TranscriptSegment[] = [];
  const allSegments: TranscriptSegment[] = [];
  let allWords: TranscriptWord[] = [];
  let pendingWords: TranscriptWord[] = [];
  let lastTextFallback = '';
  let startSeconds = 0;
  let shouldMergePending = false;
  let warnings: readonly TranscriptWarning[] = [];
  const maxWindows = Math.max(
    4,
    Math.ceil(Math.max(0, audio.durationSeconds - policy.windowDurationSec) / CURSOR_MIN_ADVANCE_SECONDS) +
      2,
  );

  for (
    let windowIndex = 0;
    windowIndex < maxWindows && startSeconds < audio.durationSeconds - EPSILON_SECONDS;
    windowIndex += 1
  ) {
    const endSeconds = Math.min(audio.durationSeconds, startSeconds + policy.windowDurationSec);
    const startFrame = Math.max(0, Math.min(audio.numberOfFrames - 1, Math.floor(startSeconds * audio.sampleRate)));
    const endFrame = Math.max(startFrame + 1, Math.min(audio.numberOfFrames, Math.ceil(endSeconds * audio.sampleRate)));
    const isLastWindow = endSeconds >= audio.durationSeconds - EPSILON_SECONDS;
    const windowAudio = audio.sliceFrames(startFrame, endFrame);
    const windowResult = withTimeOffset(
      await context.transcribeWindow(windowAudio, {
        ...decision.options,
        detail: 'words',
      }),
      startSeconds,
    );

    addWindowMetrics(accumulator, windowResult.meta.metrics);
    warnings = [...warnings, ...windowResult.warnings];
    lastTextFallback = windowResult.text || lastTextFallback;

    const currentWords = windowResult.words ?? [];
    const currentSegments = windowResult.segments ?? [];
    if (currentWords.length === 0 && currentSegments.length > 0) {
      for (const segment of currentSegments) {
        appendFinalizedSegment(allSegments, segment);
      }
    }
    const windowWords = shouldMergePending
      ? mergePendingAndCurrentWords(pendingWords, currentWords)
      : dedupeWindowWords(currentWords);
    const segments = partitionWordsIntoSegments(windowWords);

    if (isLastWindow) {
      for (const segment of segments) {
        appendFinalizedSegment(finalizedSegments, segment);
      }
      pendingWords = [];
      allWords = dedupeWindowWords([...allWords, ...windowWords]);
      break;
    }

    if (segments.length > 1) {
      const pendingSegment = segments[segments.length - 1]!;
      if (pendingSegment.startTime >= startSeconds + CURSOR_MIN_ADVANCE_SECONDS - EPSILON_SECONDS) {
        for (const segment of segments.slice(0, -1)) {
          appendFinalizedSegment(finalizedSegments, segment);
        }
        pendingWords = dedupeWindowWords(
          windowWords.filter((word) => pendingSegment.wordIndices?.includes(word.index)),
        );
        const nextStartSeconds = Math.min(
          audio.durationSeconds,
          relocateCursorToNearbyGap(pendingSegment.startTime, windowWords),
        );
        shouldMergePending = nextStartSeconds > pendingSegment.startTime + EPSILON_SECONDS;
        allWords = dedupeWindowWords([...allWords, ...flattenSegmentWords(segments.slice(0, -1), windowWords)]);
        if (nextStartSeconds > startSeconds + EPSILON_SECONDS) {
          startSeconds = nextStartSeconds;
          continue;
        }
      }
    }

    const fallbackStartSeconds = Math.min(
      audio.durationSeconds,
      startSeconds + Math.max(1, policy.windowDurationSec - policy.overlapSec),
    );
    pendingWords = dedupeWindowWords(windowWords);
    shouldMergePending = true;
    allWords = dedupeWindowWords([...allWords, ...windowWords]);
    if (fallbackStartSeconds <= startSeconds + EPSILON_SECONDS) {
      warnings = addWarning(
        warnings,
        'pipeline.windowing.cursor-stalled',
        'Windowed transcription cursor stopped advancing; returning partial merged output.',
      );
      break;
    }
    startSeconds = fallbackStartSeconds;
  }

  const mergedWords = dedupeWindowWords([...allWords, ...pendingWords]).map((word, index) => ({
    ...word,
    index,
  }));

  let segments: TranscriptSegment[];
  if (mergedWords.length > 0) {
    segments = partitionWordsIntoSegments(mergedWords).map((segment, index) => ({ ...segment, index }));
  } else if (allSegments.length > 0) {
    segments = [...allSegments]
      .sort((a, b) => a.startTime - b.startTime)
      .map((segment, index) => ({ ...segment, index }));
  } else {
    segments = finalizedSegments.map((segment, index) => ({ ...segment, index }));
  }

  const text = mergedWords.length > 0
    ? joinTranscriptWords(mergedWords)
    : segments.length > 0
      ? segments.map((segment) => segment.text).join(' ').trim()
      : lastTextFallback.trim();
  const metrics = buildWindowedMetrics(accumulator);

  return {
    text,
    warnings,
    meta: {
      detailLevel: decision.options.detail ?? 'words',
      isFinal: true,
      sampleRate: audio.sampleRate,
      durationSeconds: audio.durationSeconds,
      wordCount: mergedWords.length || undefined,
      segmentCount: segments.length || undefined,
      metrics,
    },
    ...(segments.length > 0 && decision.options.detail !== 'text' ? { segments } : {}),
    ...((decision.options.detail === 'words' || decision.options.detail === 'detailed') &&
    mergedWords.length > 0
      ? { words: mergedWords }
      : {}),
  };
}
