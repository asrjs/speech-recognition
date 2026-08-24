/**
 * WAV2VEC2 Forced Aligner — uses CTC Viterbi to align transcript to audio.
 *
 * Takes a WAV2VEC2 logit provider (ONNX session output), tokenizes
 * the transcript, and runs CTC forced alignment for word-level timestamps.
 *
 * @module alignment/wav2vec2-aligner
 */

import { ctcForceAlign, type CtcAlignedFrame } from './ctc-viterbi.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Wav2Vec2AlignerConfig {
  /** Function that produces CTC logits [T*V] Float32Array */
  readonly logitProvider: () => Float32Array;
  /** Tokenizer with encode/decode for the CTC vocabulary */
  readonly tokenizer: {
    encode: (text: string) => number[];
    decode: (ids: readonly number[]) => string;
    decodeTokenPiece?: (id: number) => string;
  };
  readonly vocabSize: number;
  readonly blankId: number;
  readonly wordSeparator: string;
  readonly frameCount: number;
  readonly sampleRate: number;
  /** Optional default duration from a prior Wav2Vec2 logits extraction. */
  readonly audioDurationSeconds?: number;
}

export interface Wav2Vec2AlignedWord {
  readonly text: string;
  readonly start: number;
  readonly end: number;
  readonly confidence: number;
  readonly charFrames: readonly CtcAlignedFrame[];
}

export interface Wav2Vec2AlignmentResult {
  readonly words: readonly Wav2Vec2AlignedWord[];
  readonly totalFrames: number;
  readonly totalChars: number;
}

export interface Wav2Vec2AlignerAlignOptions {
  readonly transcript: string;
  readonly audioDurationSeconds?: number;
}

export interface Wav2Vec2ReusableLogits {
  readonly logits: Float32Array;
  readonly frameCount: number;
  readonly vocabSize: number;
  readonly blankId: number;
  readonly tokenizer: Wav2Vec2AlignerConfig['tokenizer'];
  readonly sampleRate: number;
  readonly audioDurationSeconds: number;
  readonly wordSeparator?: string;
}

export interface Wav2Vec2Aligner {
  align(options: Wav2Vec2AlignerAlignOptions): Wav2Vec2AlignmentResult;
}

// ---------------------------------------------------------------------------
// Helper: groupCharAlignmentToWindows
// ---------------------------------------------------------------------------

/**
 * Group a CTC character-level alignment into word windows using a separator.
 *
 * Walks through characters and aligned frames in parallel.
 * When a separator is encountered in the text, closes the current word.
 *
 * Each word's start = first char frame start,
 * end = last char frame end,
 * confidence = average of char confidences.
 */
export function groupCharAlignmentToWords(
  frames: readonly CtcAlignedFrame[],
  text: string,
  separator: string,
): Wav2Vec2AlignedWord[] {
  if (frames.length === 0) return [];

  // Split text into word-level character groups
  const words: { text: string; charCount: number }[] = [];
  let current = '';
  for (const ch of text) {
    if (ch === separator) {
      if (current.length > 0) {
        words.push({ text: current, charCount: current.length });
        current = '';
      }
    } else {
      current += ch;
    }
  }
  if (current.length > 0) {
    words.push({ text: current, charCount: current.length });
  }

  // Assign frame ranges to each word based on cumulative char position
  const result: Wav2Vec2AlignedWord[] = [];
  let frameIdx = 0;

  for (const word of words) {
    const wordFrames: CtcAlignedFrame[] = [];
    let charConsumed = 0;

    while (frameIdx < frames.length && charConsumed < word.charCount) {
      const frame = frames[frameIdx]!;
      // Skip separator/space frames that may appear in alignment.
      // Wav2Vec2-base-960h uses `|` as the word delimiter; decodeTokenPiece
      // maps it to a space, but raw vocab labels may still be `|`.
      if (
        frame.char === separator ||
        frame.char === ' ' ||
        frame.char === '|' ||
        frame.char.length === 0
      ) {
        frameIdx++;
        continue;
      }
      wordFrames.push(frame);
      frameIdx++;
      charConsumed++;
    }

    if (wordFrames.length > 0) {
      result.push(buildWord(wordFrames, word.text));
    }
  }

  return result;
}

function buildWord(
  charFrames: CtcAlignedFrame[],
  text: string,
): Wav2Vec2AlignedWord {
  const first = charFrames[0]!;
  const last = charFrames[charFrames.length - 1]!;
  const avgConfidence = charFrames.reduce((s, f) => s + f.confidence, 0) / charFrames.length;

  const rawEnd = last.endSeconds ?? last.seconds;
  // q8 Viterbi often parks the last letter of a short word just before the
  // next word. Cap duration by character count so "of" cannot eat 600ms.
  const maxDuration = Math.max(0.2, charFrames.length * 0.1);
  const end = Math.min(Math.max(rawEnd, first.seconds), first.seconds + maxDuration);
  return {
    text,
    start: first.seconds,
    end,
    confidence: Math.min(1.0, avgConfidence),
    charFrames,
  };
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

function tokenLabelFromTokenizer(
  tokenizer: Wav2Vec2AlignerConfig['tokenizer'],
  tokenId: number,
): string {
  if (tokenizer.decodeTokenPiece) {
    return tokenizer.decodeTokenPiece(tokenId);
  }
  const decoded = tokenizer.decode([tokenId]);
  // `decode()` trims, so a `|` / space piece becomes "". Keep it as a separator.
  return decoded.length > 0 ? decoded : ' ';
}

/**
 * Create a WAV2VEC2 forced aligner.
 *
 * The aligner:
 *   1. Gets CTC logits from logitProvider
 *   2. Tokenizes the transcript
 *   3. Runs ctcForceAlign (Viterbi)
 *   4. Groups characters into word windows
 */
export function createWav2Vec2Aligner(
  config: Wav2Vec2AlignerConfig,
): Wav2Vec2Aligner {
  return {
    align(options: Wav2Vec2AlignerAlignOptions): Wav2Vec2AlignmentResult {
      const { transcript } = options;
      const audioDurationSeconds = options.audioDurationSeconds ?? config.audioDurationSeconds;
      if (typeof audioDurationSeconds !== 'number' || !Number.isFinite(audioDurationSeconds)) {
        throw new Error('Wav2Vec2 alignment requires audioDurationSeconds.');
      }

      if (!transcript || transcript.trim().length === 0) {
        return { words: [], totalFrames: config.frameCount, totalChars: 0 };
      }

      // 1. Get logits
      const logits = config.logitProvider();

      // 2. Tokenize transcript
      const targetTokens = config.tokenizer.encode(transcript);

      if (targetTokens.length === 0) {
        return { words: [], totalFrames: config.frameCount, totalChars: 0 };
      }

      // 3. CTC Viterbi alignment
      const alignment = ctcForceAlign(
        logits,
        config.frameCount,
        config.vocabSize,
        targetTokens,
        config.blankId,
        {
          audioDurationSeconds,
          tokenToChar: (tokenId) => tokenLabelFromTokenizer(config.tokenizer, tokenId),
        },
      );

      // 4. Group characters into words
      const words = groupCharAlignmentToWords(
        alignment.alignedFrames,
        transcript,
        config.wordSeparator,
      );

      return {
        words,
        totalFrames: alignment.totalFrames,
        totalChars: alignment.totalTokens,
      };
    },
  };
}

/**
 * Create a Wav2Vec2 aligner from logits that were already extracted by the
 * Wav2Vec2 executor. This avoids a second ONNX run when a caller wants both
 * ASR text and forced alignment over the same audio.
 */
export function createWav2Vec2AlignerFromLogits(
  source: Wav2Vec2ReusableLogits,
): Wav2Vec2Aligner {
  return createWav2Vec2Aligner({
    logitProvider: () => source.logits,
    tokenizer: source.tokenizer,
    vocabSize: source.vocabSize,
    blankId: source.blankId,
    wordSeparator: source.wordSeparator ?? ' ',
    frameCount: source.frameCount,
    sampleRate: source.sampleRate,
    audioDurationSeconds: source.audioDurationSeconds,
  });
}
