/**
 * Segment Merger — merge Whisper chunk results with timestamp adjustment.
 *
 * After Whisper processes each VAD chunk independently, this module:
 *   1. Adjusts timestamps from chunk-relative to absolute
 *   2. Merges segments across chunks
 *   3. Deduplicates words at overlapping boundaries
 *
 * Algorithm matches faster-whisper's segment merging approach.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Minimal word shape — matches WhisperNativeWord from types.ts */
interface TimestampedWord {
  readonly start: number;
  readonly end: number;
  readonly word: string;
  readonly probability: number;
}

/** Minimal segment shape — matches WhisperNativeSegment from types.ts */
interface TimestampedSegment {
  id: number;
  start: number;
  end: number;
  text: string;
  words: readonly TimestampedWord[];
}

/** One chunk's transcription result with its time offset. */
interface ChunkTranscription {
  readonly segments: readonly TimestampedSegment[];
  readonly words: readonly TimestampedWord[];
  /** Offset in seconds — the chunk's audio position in the full recording. */
  readonly timeOffsetSeconds: number;
}

/** Merged result for all chunks. */
export interface MergedTranscription {
  readonly segments: TimestampedSegment[];
  readonly words: TimestampedWord[];
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

/**
 * Merge chunk-level transcriptions into a single result.
 *
 * Per chunk:
 *   1. Adjust all timestamps by timeOffsetSeconds
 *   2. Reassign sequential segment IDs
 *
 * For word deduplication:
 *   At chunk boundaries, the same word may appear in both chunks
 *   (previous chunk's last words ≈ next chunk's first words).
 *   Deduplicate by: if a word in the next chunk overlaps (by time)
 *   with the last word in the merged list AND has the same text,
 *   keep the one with higher probability.
 */
export function mergeWhisperSegments(
  chunks: readonly ChunkTranscription[],
): MergedTranscription {
  if (chunks.length === 0) {
    return { segments: [], words: [] };
  }

  const allSegments: TimestampedSegment[] = [];
  const allWords: TimestampedWord[] = [];
  let nextSegmentId = 0;

  for (const chunk of chunks) {
    const offset = chunk.timeOffsetSeconds;

    // Adjust segment timestamps
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

    // Adjust and deduplicate words
    for (const word of chunk.words) {
      const adjusted: TimestampedWord = {
        ...word,
        start: word.start + offset,
        end: word.end + offset,
      };

      // Check if this word overlaps with the last merged word
      const lastWord = allWords[allWords.length - 1];
      if (lastWord && lastWord.word === adjusted.word) {
        // Overlap if the new word's end > last word's start (they overlap in time)
        if (adjusted.start < lastWord.end) {
          // Keep the one with higher probability
          if (adjusted.probability > lastWord.probability) {
            allWords[allWords.length - 1] = adjusted;
          }
          continue; // skip adding duplicate
        }
      }

      allWords.push(adjusted);
    }
  }

  return { segments: allSegments, words: allWords };
}
