import type { TranscriptSegment, TranscriptSentence, TranscriptWord } from '../types/index.js';

const STRONG_SENTENCE_END_REGEX = /[!?…](?:["')\]]+)?$/u;
const PERIOD_SENTENCE_END_REGEX = /\.(?:["')\]]+)?$/u;
const TRAILING_CLOSERS_REGEX = /["')\]]+$/gu;
const LEADING_OPENERS_REGEX = /^[("'“‘\[{]+/u;
const DOTTED_ACRONYM_REGEX = /^(?:[A-Z]\.){2,}$/;
const SINGLE_LETTER_ENUM_REGEX = /^[A-Z]\.$/;
const ROMAN_ENUM_REGEX = /^(?:[IVXLCDM]+)\.$/i;
const NUMERIC_ENUM_REGEX = /^\d+\.$/;

const DEFAULT_NON_BREAKING_PERIOD_WORDS = new Set([
  'mr.',
  'mrs.',
  'ms.',
  'dr.',
  'prof.',
  'sr.',
  'jr.',
  'vs.',
  'etc.',
  'e.g.',
  'i.e.',
]);

export interface SentenceSegmentationOptions {
  readonly gapThresholdSeconds?: number;
  readonly nonBreakingPeriodWords?: ReadonlySet<string>;
}

export function joinTranscriptWords(words: readonly Pick<TranscriptWord, 'text'>[]): string {
  let text = '';
  for (const word of words) {
    const part = word.text ?? '';
    if (!part) {
      continue;
    }
    if (!text) {
      text = part;
    } else if (/^[,.;:!?)}\]]+$/.test(part)) {
      text += part;
    } else {
      text += ` ${part}`;
    }
  }
  return text;
}

function stripTrailingClosers(text: string): string {
  return text.replace(TRAILING_CLOSERS_REGEX, '');
}

function looksLikeSentenceStart(text: string): boolean {
  const cleaned = text.replace(LEADING_OPENERS_REGEX, '');
  return /^[A-Z]/.test(cleaned);
}

export function shouldEndSentenceAfterWord(
  currentWord: Pick<TranscriptWord, 'text'>,
  nextWord: Pick<TranscriptWord, 'text'> | null,
  gapSeconds = 0,
  options: SentenceSegmentationOptions = {},
): boolean {
  if (!nextWord) {
    return false;
  }

  if (gapSeconds >= (options.gapThresholdSeconds ?? 3)) {
    return true;
  }

  const currentText = String(currentWord.text ?? '');
  if (!currentText) {
    return false;
  }

  if (STRONG_SENTENCE_END_REGEX.test(currentText)) {
    return true;
  }

  if (!PERIOD_SENTENCE_END_REGEX.test(currentText)) {
    return false;
  }

  const stripped = stripTrailingClosers(currentText);
  const lowered = stripped.toLowerCase();
  const nonBreakingWords = options.nonBreakingPeriodWords ?? DEFAULT_NON_BREAKING_PERIOD_WORDS;
  if (
    nonBreakingWords.has(lowered) ||
    DOTTED_ACRONYM_REGEX.test(stripped) ||
    SINGLE_LETTER_ENUM_REGEX.test(stripped) ||
    ROMAN_ENUM_REGEX.test(stripped) ||
    NUMERIC_ENUM_REGEX.test(stripped)
  ) {
    return false;
  }

  return looksLikeSentenceStart(nextWord.text);
}

export function partitionWordsIntoSegments(
  words: readonly TranscriptWord[],
  options: SentenceSegmentationOptions = {},
): TranscriptSegment[] {
  return partitionWordsIntoSentences(words, options);
}

export function partitionWordsIntoSentences(
  words: readonly TranscriptWord[],
  options: SentenceSegmentationOptions = {},
): TranscriptSentence[] {
  if (words.length === 0) {
    return [];
  }

  const sentences: TranscriptSentence[] = [];
  let current: TranscriptWord[] = [];
  for (let i = 0; i < words.length; i += 1) {
    const word = words[i]!;
    current.push(word);

    const nextWord = words[i + 1] ?? null;
    const gapSeconds = nextWord ? Math.max(0, nextWord.startTime - word.endTime) : 0;
    if (shouldEndSentenceAfterWord(word, nextWord, gapSeconds, options)) {
      sentences.push(buildSentence(sentences.length, current));
      current = [];
    }
  }

  if (current.length > 0) {
    sentences.push(buildSentence(sentences.length, current));
  }

  return sentences;
}

function buildSentence(index: number, words: readonly TranscriptWord[]): TranscriptSentence {
  const first = words[0]!;
  const last = words[words.length - 1]!;
  const confidence = words.every((word) => typeof word.confidence === 'number')
    ? words.reduce((sum, word) => sum + (word.confidence ?? 0), 0) / words.length
    : undefined;
  return {
    index,
    text: joinTranscriptWords(words),
    startTime: first.startTime,
    endTime: last.endTime,
    confidence,
    wordIndices: words.map((word) => word.index),
  };
}
