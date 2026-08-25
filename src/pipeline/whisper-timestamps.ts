const DEFAULT_TIME_PRECISION_SECONDS = 0.02;
const DEFAULT_TIMESTAMP_TOKEN_COUNT = 1500;
const CJK_LANGUAGES = new Set(['chinese', 'japanese', 'thai', 'lao', 'myanmar']);
const PREPENDED_PUNCTUATION = '"\'“¡¿([{-';
const APPENDED_PUNCTUATION = '"\'.。,，!！?？:：”)]}、';
const PUNCTUATION_ONLY_REGEX = /^[\p{P}\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]+$/u;

export interface WhisperTimestampTokenOptions {
  readonly timestampBegin: number;
  readonly timestampEnd?: number;
  readonly timePrecisionSeconds?: number;
}

export interface WhisperTimestampSpan {
  readonly index: number;
  readonly startTime: number;
  readonly endTime: number;
  readonly tokenIds: readonly number[];
  readonly text?: string;
}

export interface DecodeWhisperTimestampSpanOptions extends WhisperTimestampTokenOptions {
  readonly decodeTokens?: (tokenIds: readonly number[]) => string;
}

export interface WhisperTokenTimestamp {
  readonly tokenId: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
}

export interface WhisperWordTimestamp {
  readonly index: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
  readonly tokenIds: readonly number[];
}

export interface CollateWhisperWordTimestampOptions {
  readonly language?: string | null;
  readonly prependedPunctuations?: string;
  readonly appendedPunctuations?: string;
}

interface WordGroup {
  text: string;
  tokenIds: number[];
  startTime: number;
  endTime: number;
}

export function isWhisperTimestampToken(tokenId: number, options: WhisperTimestampTokenOptions): boolean {
  const timestampEnd = options.timestampEnd ?? options.timestampBegin + DEFAULT_TIMESTAMP_TOKEN_COUNT;
  return tokenId >= options.timestampBegin && tokenId <= timestampEnd;
}

export function whisperTimestampTokenToSeconds(
  tokenId: number,
  options: WhisperTimestampTokenOptions,
): number {
  if (!isWhisperTimestampToken(tokenId, options)) {
    throw new Error(`Token ${tokenId} is not a Whisper timestamp token.`);
  }
  return roundTime((tokenId - options.timestampBegin) * (options.timePrecisionSeconds ?? DEFAULT_TIME_PRECISION_SECONDS));
}

export function decodeWhisperTimestampSpans(
  tokenIds: readonly number[],
  options: DecodeWhisperTimestampSpanOptions,
): WhisperTimestampSpan[] {
  const spans: WhisperTimestampSpan[] = [];
  let startTime: number | null = null;
  let currentTokens: number[] = [];

  for (const tokenId of tokenIds) {
    if (!isWhisperTimestampToken(tokenId, options)) {
      if (startTime !== null) {
        currentTokens.push(tokenId);
      }
      continue;
    }

    const timestampSeconds = whisperTimestampTokenToSeconds(tokenId, options);
    if (startTime === null) {
      startTime = timestampSeconds;
      currentTokens = [];
      continue;
    }

    if (timestampSeconds > startTime) {
      spans.push(createSpan(spans.length, startTime, timestampSeconds, currentTokens, options.decodeTokens));
    }
    startTime = timestampSeconds;
    currentTokens = [];
  }

  return spans;
}

export function mergeWhisperTokenSequences(
  sequences: readonly (readonly number[])[],
  tokenTimestampSequences?: readonly (readonly number[])[],
): readonly [number[], number[]] {
  if (sequences.length === 0) {
    return [[], []];
  }

  let leftSequence = [...sequences[0]!];
  let leftTimestamps = tokenTimestampSequences?.[0] ? [...tokenTimestampSequences[0]] : [];
  const mergedTokens: number[] = [];
  const mergedTimestamps: number[] = [];
  const useTimestamps = Array.isArray(tokenTimestampSequences) && tokenTimestampSequences.length > 0;

  for (let i = 1; i < sequences.length; i += 1) {
    const rightSequence = [...sequences[i]!];
    const rightTimestamps = tokenTimestampSequences?.[i] ? [...tokenTimestampSequences[i]!] : [];
    const overlap = findBestSequenceOverlap(leftSequence, rightSequence, leftTimestamps, rightTimestamps, useTimestamps);
    mergedTokens.push(...leftSequence.slice(0, overlap.leftStop));
    leftSequence = rightSequence.slice(overlap.rightStop);
    if (useTimestamps) {
      mergedTimestamps.push(...leftTimestamps.slice(0, overlap.leftStop));
      leftTimestamps = rightTimestamps.slice(overlap.rightStop);
    }
  }

  mergedTokens.push(...leftSequence);
  if (useTimestamps) {
    mergedTimestamps.push(...leftTimestamps);
  }
  return [mergedTokens, mergedTimestamps];
}

export function collateWhisperWordTimestamps(
  tokenTimestamps: readonly WhisperTokenTimestamp[],
  options: CollateWhisperWordTimestampOptions = {},
): WhisperWordTimestamp[] {
  const wordGroups = CJK_LANGUAGES.has(String(options.language ?? '').toLowerCase())
    ? splitWhisperTokensOnUnicode(tokenTimestamps)
    : splitWhisperTokensOnSpaces(tokenTimestamps);
  const merged = mergeWhisperPunctuations(
    wordGroups,
    options.prependedPunctuations ?? PREPENDED_PUNCTUATION,
    options.appendedPunctuations ?? APPENDED_PUNCTUATION,
  );

  return merged.map((word, index) => ({
    index,
    text: word.text.trim(),
    startTime: word.startTime,
    endTime: word.endTime,
    tokenIds: word.tokenIds,
  }));
}

function createSpan(
  index: number,
  startTime: number,
  endTime: number,
  tokenIds: readonly number[],
  decodeTokens: DecodeWhisperTimestampSpanOptions['decodeTokens'],
): WhisperTimestampSpan {
  return {
    index,
    startTime,
    endTime,
    tokenIds: [...tokenIds],
    ...(decodeTokens ? { text: decodeTokens(tokenIds) } : {}),
  };
}

function findBestSequenceOverlap(
  leftSequence: readonly number[],
  rightSequence: readonly number[],
  leftTimestamps: readonly number[],
  rightTimestamps: readonly number[],
  useTimestamps: boolean,
): { leftStart: number; leftStop: number; rightStart: number; rightStop: number } {
  const leftLength = leftSequence.length;
  const rightLength = rightSequence.length;
  let bestScore = 0;
  let best = { leftStart: leftLength, leftStop: leftLength, rightStart: 0, rightStop: 0 };

  for (let j = 1; j < leftLength + rightLength; j += 1) {
    const leftStart = Math.max(0, leftLength - j);
    const leftStop = Math.min(leftLength, leftLength + rightLength - j);
    const rightStart = Math.max(0, j - leftLength);
    const rightStop = Math.min(rightLength, j);
    const overlapLength = leftStop - leftStart;
    if (overlapLength <= 0 || overlapLength !== rightStop - rightStart) {
      continue;
    }

    let matches = 0;
    for (let offset = 0; offset < overlapLength; offset += 1) {
      if (leftSequence[leftStart + offset] !== rightSequence[rightStart + offset]) {
        continue;
      }
      if (useTimestamps) {
        const leftTimestamp = leftTimestamps[leftStart + offset];
        const rightTimestamp = rightTimestamps[rightStart + offset];
        if (leftTimestamp === undefined || rightTimestamp === undefined || leftTimestamp > rightTimestamp) {
          continue;
        }
      }
      matches += 1;
    }

    const score = matches / j + j / 10_000;
    if (matches > 0 && score > bestScore) {
      bestScore = score;
      best = { leftStart, leftStop, rightStart, rightStop };
    }
  }

  return best;
}

function splitWhisperTokensOnUnicode(tokenTimestamps: readonly WhisperTokenTimestamp[]): WordGroup[] {
  return tokenTimestamps
    .filter((token) => token.text.length > 0)
    .map((token) => ({
      text: token.text,
      tokenIds: [token.tokenId],
      startTime: token.startTime,
      endTime: token.endTime,
    }));
}

function splitWhisperTokensOnSpaces(tokenTimestamps: readonly WhisperTokenTimestamp[]): WordGroup[] {
  const words: WordGroup[] = [];
  for (const token of tokenTimestamps) {
    const startsWord = token.text.startsWith(' ') || isPunctuationOnly(token.text.trim()) || words.length === 0;
    if (startsWord) {
      words.push({ text: token.text, tokenIds: [token.tokenId], startTime: token.startTime, endTime: token.endTime });
      continue;
    }

    const previous = words[words.length - 1]!;
    previous.text += token.text;
    previous.tokenIds.push(token.tokenId);
    previous.endTime = token.endTime;
  }
  return words;
}

function mergeWhisperPunctuations(
  words: readonly WordGroup[],
  prepended: string,
  appended: string,
): WordGroup[] {
  const merged = words.map((word) => ({ ...word, tokenIds: [...word.tokenIds] }));

  let i = merged.length - 2;
  let j = merged.length - 1;
  while (i >= 0) {
    const word = merged[i]!;
    if (word.text.startsWith(' ') && prepended.includes(word.text.trim())) {
      mergeWordGroups(word, merged[j]!, 'prepend');
      clearWordGroup(word);
    } else {
      j = i;
    }
    i -= 1;
  }

  i = 0;
  j = 1;
  while (j < merged.length) {
    const right = merged[j]!;
    if (!merged[i]!.text.endsWith(' ') && isAppendedPunctuation(right.text, appended)) {
      mergeWordGroups(merged[i]!, right, 'append');
      clearWordGroup(right);
    } else {
      i = j;
    }
    j += 1;
  }

  return merged.filter((word) => word.text.length > 0);
}

function mergeWordGroups(left: WordGroup, right: WordGroup, mode: 'append' | 'prepend'): void {
  if (mode === 'append') {
    left.text += right.text;
    left.tokenIds.push(...right.tokenIds);
    left.endTime = right.endTime;
    return;
  }
  right.text = left.text + right.text;
  right.tokenIds.unshift(...left.tokenIds);
  right.startTime = left.startTime;
}

function clearWordGroup(word: WordGroup): void {
  word.text = '';
  word.tokenIds = [];
}

function isPunctuationOnly(text: string): boolean {
  PUNCTUATION_ONLY_REGEX.lastIndex = 0;
  return text.length > 0 && PUNCTUATION_ONLY_REGEX.test(text);
}

function isAppendedPunctuation(text: string, appended: string): boolean {
  if (text.length === 0 || text !== text.trim() || !isPunctuationOnly(text)) return false;
  return [...text].every((character) => appended.includes(character));
}

function roundTime(value: number): number {
  return Math.round(value * 100) / 100;
}
