/**
 * Segment Merger — merge ASR chunk results with timestamp adjustment.
 *
 * After the ASR model processes each VAD chunk independently, this module:
 *   1. Adjusts timestamps from chunk-relative to absolute
 *   2. Merges segments across chunks
 *   3. Deduplicates words at overlapping boundaries
 *
 * Model-agnostic. Works with any ASR output format.
 */

interface TimestampedWord {
  readonly start: number;
  readonly end: number;
  readonly word: string;
  readonly probability: number;
}

interface TimestampedSegment {
  id: number;
  start: number;
  end: number;
  text: string;
  words: readonly TimestampedWord[];
}

interface ChunkTranscription {
  readonly segments: readonly TimestampedSegment[];
  readonly words: readonly TimestampedWord[];
  readonly timeOffsetSeconds: number;
}

export interface MergedTranscription {
  readonly segments: TimestampedSegment[];
  readonly words: TimestampedWord[];
}

export function mergeSegments(chunks: readonly ChunkTranscription[]): MergedTranscription {
  if (chunks.length === 0) return { segments: [], words: [] };

  const allSegments: TimestampedSegment[] = [];
  const allWords: TimestampedWord[] = [];
  let nextSegmentId = 0;

  for (const chunk of chunks) {
    const offset = chunk.timeOffsetSeconds;

    for (const seg of chunk.segments) {
      allSegments.push({
        ...seg,
        id: nextSegmentId++,
        start: seg.start + offset,
        end: seg.end + offset,
        words: seg.words.map((w) => ({
          ...w,
          start: w.start + offset,
          end: w.end + offset,
        })),
      });
    }

    for (const word of chunk.words) {
      const adjusted: TimestampedWord = {
        ...word,
        start: word.start + offset,
        end: word.end + offset,
      };

      const lastWord = allWords[allWords.length - 1];
      if (lastWord && lastWord.word === adjusted.word) {
        if (adjusted.start < lastWord.end) {
          if (adjusted.probability > lastWord.probability) {
            allWords[allWords.length - 1] = adjusted;
          }
          continue;
        }
      }

      allWords.push(adjusted);
    }
  }

  return { segments: allSegments, words: allWords };
}
