/**
 * Post-Processing Extras — word deduplication, text normalization, sentence boundary.
 *
 * Model-agnostic. Pure functions. No ONNX dependency.
 */

// ---------------------------------------------------------------------------
// Word Deduplication
// ---------------------------------------------------------------------------

export interface DedupWord {
  word: string;
  start: number;
  end: number;
  probability: number;
}

/**
 * Deduplicate words across chunk boundaries.
 * Two words are considered duplicates if they:
 *   - Have the same normalized text
 *   - Overlap in time (start < prevEnd)
 * When duplicate, keep the one with higher probability.
 */
export function deduplicateWords(words: readonly DedupWord[]): DedupWord[] {
  if (words.length <= 1) return [...words];

  const result: DedupWord[] = [words[0]!];

  for (let i = 1; i < words.length; i++) {
    const word = words[i]!;
    const prev = result[result.length - 1]!;

    if (word.word.toLowerCase() === prev.word.toLowerCase() && word.start < prev.end) {
      if (word.probability > prev.probability) {
        result[result.length - 1] = word;
      }
      continue;
    }

    result.push(word);
  }

  return result;
}

// ---------------------------------------------------------------------------
// Text Normalization
// ---------------------------------------------------------------------------

const COLLAPSE_SPACES = /\s+/g;
const STRIP_PUNCTUATION_END = /[.,!?;:]+$/;

/**
 * Normalize transcript text:
 *   - Collapse multiple spaces
 *   - Trim whitespace
 *   - Strip trailing punctuation (for word comparison)
 */
export function normalizeText(text: string, stripTrailingPunctuation = false): string {
  let normalized = text.replace(COLLAPSE_SPACES, ' ').trim();
  if (stripTrailingPunctuation) {
    normalized = normalized.replace(STRIP_PUNCTUATION_END, '');
  }
  return normalized;
}

// ---------------------------------------------------------------------------
// Sentence Boundary Detection
// ---------------------------------------------------------------------------

export interface Sentence {
  text: string;
  words: readonly DedupWord[];
  start: number;
  end: number;
}

const SENTENCE_END = /[.!?]+$/;

/**
 * Split a word sequence into sentences using punctuation heuristics.
 * Sentence boundaries are detected at:
 *   - Words ending with . ! ?
 *   - 3+ second gaps between words
 */
export function buildSentences(
  words: readonly DedupWord[],
  minGapSeconds: number = 3.0,
): Sentence[] {
  if (words.length === 0) return [];

  const sentences: Sentence[] = [];
  let sentenceWords: DedupWord[] = [words[0]!];

  for (let i = 1; i < words.length; i++) {
    const word = words[i]!;
    const prev = words[i - 1]!;
    const gap = word.start - prev.end;

    if (SENTENCE_END.test(prev.word) || gap >= minGapSeconds) {
      sentences.push({
        text: sentenceWords.map((w) => w.word).join(' '),
        words: [...sentenceWords],
        start: sentenceWords[0]!.start,
        end: sentenceWords[sentenceWords.length - 1]!.end,
      });
      sentenceWords = [];
    }

    sentenceWords.push(word);
  }

  // Final sentence
  if (sentenceWords.length > 0) {
    sentences.push({
      text: sentenceWords.map((w) => w.word).join(' '),
      words: [...sentenceWords],
      start: sentenceWords[0]!.start,
      end: sentenceWords[sentenceWords.length - 1]!.end,
    });
  }

  return sentences;
}
