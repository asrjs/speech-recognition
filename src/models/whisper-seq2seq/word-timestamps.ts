import {
  collateWhisperWordTimestamps,
  isWhisperTimestampToken,
  whisperTimestampTokenToSeconds,
  type WhisperTimestampTokenOptions,
  type WhisperTokenTimestamp,
} from '../../pipeline/whisper-timestamps.js';
import type {
  WhisperForcedAlignmentWord,
  WhisperNativeToken,
  WhisperNativeWord,
} from './types.js';

export interface BuildWhisperWordTimestampOptions extends WhisperTimestampTokenOptions {
  readonly language?: string | null;
}

interface InterpolatedTokenTimestamp extends WhisperTokenTimestamp {
  readonly sourceIndex: number;
  readonly confidence?: number;
}

export function coalesceWhisperWordTimestamps(
  alignedWords: readonly WhisperNativeWord[] | undefined,
  tokens: readonly WhisperNativeToken[],
  options: BuildWhisperWordTimestampOptions,
): WhisperNativeWord[] {
  if (alignedWords && alignedWords.length > 0) {
    return constrainWhisperWordDurations(alignedWords);
  }
  return buildWhisperWordTimestampsFromTokenDetails(tokens, options);
}

/**
 * Keep word timestamps inside the actual audio window.
 *
 * Whisper always pads the feature input to its 30-second model window, so
 * generated timestamp tokens can legitimately refer to padded silence after a
 * short clip.  Native transcript consumers should receive clip-relative times
 * instead: retain a word that overlaps the clip, clip its end, and discard
 * words that are entirely in the padded tail.
 */
export function constrainWhisperWordTimestampsToDuration(
  words: readonly WhisperNativeWord[],
  durationSeconds: number,
): WhisperNativeWord[] {
  if (!Number.isFinite(durationSeconds) || durationSeconds < 0) {
    return words.map((word) => ({ ...word }));
  }

  const limit = durationSeconds;
  const bounded: WhisperNativeWord[] = [];
  for (const word of words) {
    const rawStart = Number.isFinite(word.startTime) ? word.startTime : 0;
    const rawEnd = Number.isFinite(word.endTime) ? word.endTime : rawStart;
    const lower = Math.min(rawStart, rawEnd);
    const upper = Math.max(rawStart, rawEnd);
    if (upper <= 0 || lower >= limit) continue;

    const startTime = Math.min(limit, Math.max(0, lower));
    const endTime = Math.min(limit, Math.max(startTime, upper));
    if (endTime <= 0 || startTime >= limit) continue;
    bounded.push({
      ...word,
      index: bounded.length,
      startTime,
      endTime,
    });
  }
  return bounded;
}

export interface WhisperDtwTokenTimestampInput {
  readonly id: number;
  readonly text: string;
  readonly sourceIndex?: number;
  readonly confidence?: number;
}

export function buildWhisperWordTimestampsFromDtwTokens(
  tokens: readonly WhisperDtwTokenTimestampInput[],
  dtwTimestamps: readonly number[],
  options: Pick<BuildWhisperWordTimestampOptions, 'language'> = {},
): WhisperNativeWord[] {
  const tokenTimestamps: InterpolatedTokenTimestamp[] = tokens.flatMap((token, index) => {
    if (token.text.length === 0) return [];
    const startTime = dtwTimestamps[index] ?? 0;
    const endTime = dtwTimestamps[index + 1] ?? startTime;
    return [
      {
        tokenId: token.id,
        text: token.text,
        startTime,
        endTime,
        sourceIndex: token.sourceIndex ?? index,
        ...(token.confidence !== undefined ? { confidence: token.confidence } : {}),
      },
    ];
  });
  if (tokenTimestamps.length === 0) return [];

  const words = collateWhisperWordTimestamps(tokenTimestamps, { language: options.language });
  let cursor = 0;
  const collated = words.map((word) => {
    const sourceTokens = tokenTimestamps.slice(cursor, cursor + word.tokenIds.length);
    cursor += word.tokenIds.length;
    const confidences = sourceTokens
      .map((token) => token.confidence)
      .filter((confidence): confidence is number => confidence !== undefined);
    return {
      index: word.index,
      text: word.text,
      startTime: word.startTime,
      endTime: word.endTime,
      tokenIds: word.tokenIds,
      tokenIndices: sourceTokens.map((token) => token.sourceIndex),
      ...(confidences.length > 0
        ? { confidence: confidences.reduce((sum, confidence) => sum + confidence, 0) / confidences.length }
        : {}),
    };
  });
  return constrainWhisperWordDurations(collated);
}

export function buildWhisperWordTimestampsFromTokenDetails(
  tokens: readonly WhisperNativeToken[],
  options: BuildWhisperWordTimestampOptions,
): WhisperNativeWord[] {
  const tokenTimestamps = interpolateTokenTimestamps(tokens, options);
  if (tokenTimestamps.length === 0) return [];

  const words = collateWhisperWordTimestamps(tokenTimestamps, { language: options.language });
  let cursor = 0;
  const collated = words.map((word) => {
    const sourceTokens = tokenTimestamps.slice(cursor, cursor + word.tokenIds.length);
    cursor += word.tokenIds.length;
    const confidences = sourceTokens
      .map((token) => token.confidence)
      .filter((confidence): confidence is number => confidence !== undefined);
    return {
      index: word.index,
      text: word.text,
      startTime: word.startTime,
      endTime: word.endTime,
      tokenIds: word.tokenIds,
      tokenIndices: sourceTokens.map((token) => token.sourceIndex),
      ...(confidences.length > 0
        ? { confidence: confidences.reduce((sum, confidence) => sum + confidence, 0) / confidences.length }
        : {}),
    };
  });
  return constrainWhisperWordDurations(collated);
}

function interpolateTokenTimestamps(
  tokens: readonly WhisperNativeToken[],
  options: BuildWhisperWordTimestampOptions,
): InterpolatedTokenTimestamp[] {
  const output: InterpolatedTokenTimestamp[] = [];
  let segmentStart: number | null = null;
  let segmentTokens: WhisperNativeToken[] = [];

  for (const token of tokens) {
    const tokenId = token.id;
    if (tokenId === undefined || !isWhisperTimestampToken(tokenId, options)) {
      if (segmentStart !== null && tokenId !== undefined && !token.special) {
        segmentTokens.push(token);
      }
      continue;
    }

    const timestamp = whisperTimestampTokenToSeconds(tokenId, options);
    if (segmentStart === null) {
      segmentStart = timestamp;
      segmentTokens = [];
      continue;
    }

    output.push(...interpolateSegmentTokens(segmentTokens, segmentStart, timestamp));
    segmentStart = timestamp;
    segmentTokens = [];
  }

  return output;
}

function interpolateSegmentTokens(
  tokens: readonly WhisperNativeToken[],
  startTime: number,
  endTime: number,
): InterpolatedTokenTimestamp[] {
  if (tokens.length === 0 || endTime <= startTime) return [];
  const duration = endTime - startTime;
  return tokens.flatMap((token, index) => {
    if (token.id === undefined || token.text.length === 0) return [];
    const tokenStart = roundTime(startTime + (duration * index) / tokens.length);
    const tokenEnd = roundTime(startTime + (duration * (index + 1)) / tokens.length);
    return [
      {
        tokenId: token.id,
        text: token.text,
        startTime: tokenStart,
        endTime: tokenEnd,
        sourceIndex: token.index,
        confidence: token.confidence,
      },
    ];
  });
}

function roundTime(seconds: number): number {
  return Math.round(seconds * 1000) / 1000;
}

const SENTENCE_END_MARKS = new Set(['.', '。', '!', '！', '?', '？']);
const PAUSE_END_PATTERN = /[,.!?;:…，。！？：；]$/;
const MIN_WORD_DURATION = 0.08;
/** Close DTW holes larger than this even when the median word is long. */
const MAX_INTER_WORD_GAP = 0.4;
/** After wav2vec2 overlay, only collapse egregious mid-phrase holes. */
const MAX_POST_ALIGN_GAP = 1.2;
/** Wav2Vec2 CNN context is ~400ms; WhisperX pads VAD slices by 0.5s+. */
const ALIGNMENT_SEGMENT_PAD_SECONDS = 0.5;
/**
 * If CTC's first word in a pause group starts this far later than Whisper/DTW,
 * the Viterbi path likely squeezed the phrase toward a later region. Keep DTW.
 */
const MAX_ALIGN_FIRST_WORD_LATE_DRIFT = 0.8;

function isSentenceEndMark(text: string): boolean {
  const trimmed = text.trim();
  return trimmed.length > 0 && [...trimmed].every((char) => SENTENCE_END_MARKS.has(char));
}

function hasTrailingPause(text: string): boolean {
  return PAUSE_END_PATTERN.test(text.trim());
}

function medianDuration(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2
    : (sorted[middle] ?? 0);
}

/**
 * OpenAI Whisper timing heuristic: cap median word duration at 0.7s and
 * truncate words longer than 2x that median. Sentence-boundary words are
 * clipped first; remaining DTW outliers (common on turbo's single 0–N span)
 * are clipped so one token cannot swallow multiple seconds.
 */
export function constrainWhisperWordDurations(
  words: readonly WhisperNativeWord[],
): WhisperNativeWord[] {
  if (words.length === 0) return [];
  const durations = words
    .map((word) => word.endTime - word.startTime)
    .filter((duration) => duration > 0);
  const median = Math.min(0.7, medianDuration(durations));
  const maxDuration = median * 2;
  if (!(maxDuration > 0)) return words.map((word) => ({ ...word }));

  const next = words.map((word) => ({ ...word }));
  // A short clip can contain a long leading pause. Whisper's DTW path then
  // assigns that pause to the first word, and OpenAI/faster-whisper shift the
  // word start toward its end instead of clipping the end toward zero. Keep
  // that boundary intact so the later gap-closing pass cannot pull the whole
  // phrase into the leading silence.
  const first = next[0];
  if (first) {
    const firstDuration = first.endTime - first.startTime;
    if (firstDuration > maxDuration * 2) {
      const startTime = Math.max(first.startTime, first.endTime - maxDuration);
      next[0] = {
        ...first,
        startTime: Math.min(startTime, first.endTime - MIN_WORD_DURATION),
      };
    }
  }
  for (let index = 0; index < next.length; index++) {
    const word = next[index]!;
    const duration = word.endTime - word.startTime;
    if (duration <= maxDuration) continue;
    const previous = index > 0 ? next[index - 1] : undefined;
    if (isSentenceEndMark(word.text)) {
      next[index] = { ...word, endTime: roundTime(word.startTime + maxDuration) };
    } else if (previous && isSentenceEndMark(previous.text)) {
      next[index] = { ...word, startTime: roundTime(Math.max(previous.endTime, word.endTime - maxDuration)) };
    } else if (duration > maxDuration * 2) {
      next[index] = { ...word, endTime: roundTime(word.startTime + maxDuration) };
    }
  }

  for (let index = 1; index < next.length; index++) {
    const word = next[index]!;
    const previous = next[index - 1]!;
    if (word.startTime < previous.startTime) {
      next[index] = { ...word, startTime: previous.startTime };
    }
    if (next[index]!.endTime < next[index]!.startTime) {
      next[index] = { ...next[index]!, endTime: next[index]!.startTime };
    }
  }
  return clipShortWhisperWordDurations(
    expandWhisperStubWordDurations(
      closeWhisperWordTimestampGaps(next, MAX_INTER_WORD_GAP),
    ),
  );
}

/**
 * WhisperX aligns each VAD/pause segment on its own audio slice. Splitting
 * here on the DTW pause gap prevents a mid-clip silence from shoving the
 * first phrase several seconds later.
 */
export function splitWhisperWordsByPause(
  words: readonly WhisperNativeWord[],
  minGapSeconds: number = MAX_INTER_WORD_GAP,
): WhisperNativeWord[][] {
  if (words.length === 0) return [];
  const groups: WhisperNativeWord[][] = [[words[0]!]];
  for (let index = 1; index < words.length; index += 1) {
    const word = words[index]!;
    const previous = words[index - 1]!;
    const gap = word.startTime - previous.endTime;
    if (gap >= minGapSeconds) {
      groups.push([word]);
    } else {
      groups[groups.length - 1]!.push(word);
    }
  }
  return groups;
}

export function alignmentWindowForWhisperWords(
  words: readonly WhisperNativeWord[],
  audioDurationSeconds: number,
  padSeconds: number = ALIGNMENT_SEGMENT_PAD_SECONDS,
): { readonly startSeconds: number; readonly endSeconds: number } {
  const startSeconds = Math.max(0, (words[0]?.startTime ?? 0) - padSeconds);
  const endSeconds = Math.min(
    audioDurationSeconds,
    Math.max(startSeconds + MIN_WORD_DURATION, (words[words.length - 1]?.endTime ?? startSeconds) + padSeconds),
  );
  return { startSeconds, endSeconds };
}

/**
 * Turbo DTW often leaves multi-second holes after duration clipping
 * (`long` ends at 1.06s while `history` still starts at 2.35s). Pull the
 * next word forward across those holes, but stop at punctuation pauses so
 * a comma after `world,` does not swallow the following phrase.
 */
function closeWhisperWordTimestampGaps(
  words: readonly WhisperNativeWord[],
  maxGapSeconds: number,
): WhisperNativeWord[] {
  if (words.length < 2) return [...words];
  const hop = 0.02;
  const next = words.map((word) => ({ ...word }));
  for (let index = 0; index < next.length - 1; index++) {
    const word = next[index]!;
    const following = next[index + 1]!;
    if (hasTrailingPause(word.text)) continue;
    const gap = following.startTime - word.endTime;
    if (gap <= maxGapSeconds) continue;
    const duration = Math.max(MIN_WORD_DURATION, hop, following.endTime - following.startTime);
    const startTime = roundTime(word.endTime + hop);
    next[index + 1] = {
      ...following,
      startTime,
      endTime: roundTime(startTime + duration),
    };
  }
  for (let index = 1; index < next.length; index++) {
    const previous = next[index - 1]!;
    const word = next[index]!;
    if (normalizeAlignableWord(previous.text).length === 0) continue;
    if (normalizeAlignableWord(word.text).length === 0) continue;
    if (word.startTime >= previous.endTime) continue;
    const duration = Math.max(hop, word.endTime - word.startTime);
    const startTime = roundTime(previous.endTime);
    next[index] = {
      ...word,
      startTime,
      endTime: roundTime(startTime + duration),
    };
  }
  return next;
}

/**
 * Function words often land on a single 20ms DTW frame (`of` 4.20–4.22).
 * Grow those stubs into neighboring slack, but do not borrow across a
 * punctuation pause so `world,` still leaves a gap before the next phrase.
 */
function expandWhisperStubWordDurations(
  words: readonly WhisperNativeWord[],
): WhisperNativeWord[] {
  if (words.length === 0) return [];
  const hop = 0.02;
  const next = words.map((word) => ({
    ...word,
    startTime: roundTime(word.startTime),
    endTime: roundTime(word.endTime),
  }));

  for (let index = 0; index < next.length; index++) {
    const word = next[index]!;
    const duration = word.endTime - word.startTime;
    if (duration >= MIN_WORD_DURATION) continue;
    const following = index + 1 < next.length ? next[index + 1] : undefined;
    const limit = following && !hasTrailingPause(word.text)
      ? following.startTime - hop
      : word.startTime + MIN_WORD_DURATION;
    const endTime = roundTime(Math.max(word.endTime, Math.min(word.startTime + MIN_WORD_DURATION, limit)));
    next[index] = { ...word, endTime: Math.max(endTime, word.endTime) };
  }

  for (let index = 1; index < next.length; index++) {
    const word = next[index]!;
    const previous = next[index - 1]!;
    const duration = word.endTime - word.startTime;
    if (duration >= MIN_WORD_DURATION) continue;
    if (hasTrailingPause(previous.text)) continue;
    const previousDuration = previous.endTime - previous.startTime;
    const need = MIN_WORD_DURATION - duration;
    const available = Math.max(0, previousDuration - MIN_WORD_DURATION);
    const borrow = Math.min(need, available);
    if (borrow <= 0) continue;
    const previousEnd = roundTime(previous.endTime - borrow);
    next[index - 1] = { ...previous, endTime: previousEnd };
    next[index] = {
      ...word,
      startTime: previousEnd,
      endTime: roundTime(Math.max(word.endTime, previousEnd + MIN_WORD_DURATION)),
    };
  }

  for (let index = 1; index < next.length; index++) {
    const previous = next[index - 1]!;
    const word = next[index]!;
    if (normalizeAlignableWord(previous.text).length === 0) continue;
    if (normalizeAlignableWord(word.text).length === 0) continue;
    if (word.startTime >= previous.endTime) continue;
    const duration = Math.max(hop, word.endTime - word.startTime);
    const startTime = roundTime(previous.endTime);
    next[index] = {
      ...word,
      startTime,
      endTime: roundTime(startTime + duration),
    };
  }
  return next;
}

function normalizeAlignableWord(text: string): string {
  return text.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '').toLowerCase();
}

/**
 * WhisperX aligns each pause group on its own slice so a mid-clip silence
 * cannot drag the first phrase later. Reject a CTC result whose first word
 * is far later than the Whisper/DTW prior for that group.
 */
export function forcedAlignmentLooksAnchored(
  whisperWords: readonly Pick<WhisperNativeWord, 'startTime'>[],
  alignedWords: readonly Pick<WhisperForcedAlignmentWord, 'startTime'>[],
  maxFirstWordLateDrift: number = MAX_ALIGN_FIRST_WORD_LATE_DRIFT,
): boolean {
  const whisperStart = whisperWords[0]?.startTime;
  const alignedStart = alignedWords[0]?.startTime;
  if (whisperStart === undefined || alignedStart === undefined) return false;
  return alignedStart - whisperStart <= maxFirstWordLateDrift;
}

export function refineWhisperWordsWithForcedAlignment(
  whisperWords: readonly WhisperNativeWord[],
  alignedWords: readonly WhisperForcedAlignmentWord[],
): WhisperNativeWord[] {
  if (whisperWords.length === 0) {
    return [];
  }
  if (alignedWords.length === 0) {
    return packWhisperAlignedWordTimestamps(whisperWords.map((word) => ({ ...word })));
  }

  if (whisperWords.length === alignedWords.length) {
    return packWhisperAlignedWordTimestamps(
      whisperWords.map((word, index) =>
        overlayAlignedWord(word, alignedWords[index]!),
      ),
    );
  }

  const next = whisperWords.map((word) => ({ ...word }));
  let alignedIndex = 0;
  for (let index = 0; index < next.length && alignedIndex < alignedWords.length; index++) {
    const wanted = normalizeAlignableWord(next[index]!.text);
    if (wanted.length === 0) continue;
    while (
      alignedIndex < alignedWords.length &&
      normalizeAlignableWord(alignedWords[alignedIndex]!.text) !== wanted
    ) {
      alignedIndex++;
    }
    if (alignedIndex >= alignedWords.length) break;
    const aligned = alignedWords[alignedIndex]!;
    next[index] = overlayAlignedWord(next[index]!, aligned);
    alignedIndex++;
  }
  return packWhisperAlignedWordTimestamps(next);
}

function overlayAlignedWord(
  whisperWord: WhisperNativeWord,
  aligned: WhisperForcedAlignmentWord,
): WhisperNativeWord {
  return {
    ...whisperWord,
    startTime: aligned.startTime,
    endTime: aligned.endTime,
    ...(aligned.confidence !== undefined ? { confidence: aligned.confidence } : {}),
  };
}

function packWhisperAlignedWordTimestamps(
  words: readonly WhisperNativeWord[],
): WhisperNativeWord[] {
  return expandWhisperStubWordDurations(
    closeWhisperWordTimestampGaps(clipShortWhisperWordDurations(words), MAX_POST_ALIGN_GAP),
  );
}

export function clipShortWhisperWordDurations(
  words: readonly WhisperNativeWord[],
): WhisperNativeWord[] {
  return words.map((word, index) => {
    const letters = normalizeAlignableWord(word.text).length;
    if (letters === 0 || letters > 3) return word;
    const maxDuration = Math.max(MIN_WORD_DURATION, letters * 0.1);
    const duration = word.endTime - word.startTime;
    // Do not erase a verified leading pause from a first short word. The
    // duration constraint has already moved its start toward the DTW end.
    if (index === 0 && word.startTime > 0.4) return word;
    if (duration <= maxDuration) return word;
    return {
      ...word,
      endTime: roundTime(word.startTime + maxDuration),
    };
  });
}
