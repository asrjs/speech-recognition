import {
  createWav2Vec2Aligner,
  createWav2Vec2AlignerFromLogits,
  groupCharAlignmentToWords,
  type Wav2Vec2AlignerConfig,
} from '@asrjs/speech-recognition/alignment';
import type { CtcAlignmentResult } from '@asrjs/speech-recognition/alignment';
import { describe, expect, it } from 'vitest';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeLogitProvider(
  frameCount: number,
  vocabSize: number,
  charIdToFrames: Map<number, number[]>,
  blankId = 0,
) {
  return (): Float32Array => {
    const logits = new Float32Array(frameCount * vocabSize);
    for (let f = 0; f < frameCount; f++) {
      for (let v = 0; v < vocabSize; v++) {
        logits[f * vocabSize + v] = 0.01;
      }
      logits[f * vocabSize + blankId] = 0.5;
    }
    for (const [charId, frames] of charIdToFrames) {
      for (const f of frames) {
        logits[f * vocabSize + charId] = 10.0;
      }
    }
    return logits;
  };
}

function simpleTokenizer(charToId: Record<string, number>) {
  const idToChar = new Map<number, string>();
  for (const [char, id] of Object.entries(charToId)) {
    idToChar.set(id, char);
  }

  return {
    encode: (text: string): number[] =>
      [...text].map((c) => charToId[c] ?? 0).filter((id) => id !== 0),
    decode: (ids: readonly number[]): string =>
      ids.map((id) => idToChar.get(id) ?? '').join(''),
    decodeTokenPiece: (id: number): string => idToChar.get(id) ?? '',
  };
}

// ---------------------------------------------------------------------------
// Tests: groupCharAlignmentToWords
// ---------------------------------------------------------------------------

describe('groupCharAlignmentToWords', () => {
  it('groups characters into words by spaces', () => {
    const frames: CtcAlignmentResult['alignedFrames'] = [
      { char: 'h', tokenIdx: 1, frame: 0, seconds: 0.0, confidence: 0.9 },
      { char: 'e', tokenIdx: 2, frame: 1, seconds: 0.02, confidence: 0.85 },
      { char: 'l', tokenIdx: 3, frame: 2, seconds: 0.04, confidence: 0.8 },
      { char: 'l', tokenIdx: 3, frame: 3, seconds: 0.06, confidence: 0.75 },
      { char: 'o', tokenIdx: 4, frame: 4, seconds: 0.08, confidence: 0.7 },
      // space
      { char: 'w', tokenIdx: 5, frame: 5, seconds: 0.10, confidence: 0.65 },
      { char: 'o', tokenIdx: 4, frame: 6, seconds: 0.12, confidence: 0.6 },
      { char: 'r', tokenIdx: 6, frame: 7, seconds: 0.14, confidence: 0.55 },
      { char: 'l', tokenIdx: 3, frame: 8, seconds: 0.16, confidence: 0.5 },
      { char: 'd', tokenIdx: 7, frame: 9, seconds: 0.18, confidence: 0.45 },
    ];

    const words = groupCharAlignmentToWords(frames, 'hello world', ' ');

    expect(words).toHaveLength(2);
    expect(words[0]!.text).toBe('hello');
    expect(words[0]!.start).toBe(0.0);
    expect(words[0]!.end).toBe(0.08);
    expect(words[0]!.charFrames).toHaveLength(5);

    expect(words[1]!.text).toBe('world');
    expect(words[1]!.start).toBe(0.10);
    expect(words[1]!.end).toBe(0.18);
    expect(words[1]!.charFrames).toHaveLength(5);
  });

  it('uses token endSeconds so words span until the next token', () => {
    const frames: CtcAlignmentResult['alignedFrames'] = [
      { char: 'h', tokenIdx: 1, frame: 0, seconds: 0.0, endSeconds: 0.02, confidence: 0.9 },
      { char: 'i', tokenIdx: 2, frame: 1, seconds: 0.02, endSeconds: 0.10, confidence: 0.8 },
    ];

    const words = groupCharAlignmentToWords(frames, 'hi', ' ');
    expect(words).toHaveLength(1);
    expect(words[0]!.start).toBe(0.0);
    expect(words[0]!.end).toBe(0.10);
  });

  it('skips wav2vec2 | delimiter frames when grouping space-separated words', () => {
    const frames: CtcAlignmentResult['alignedFrames'] = [
      { char: 'o', tokenIdx: 1, frame: 0, seconds: 1.60, endSeconds: 1.62, confidence: 0.9 },
      { char: 'f', tokenIdx: 2, frame: 1, seconds: 1.62, endSeconds: 1.64, confidence: 0.9 },
      { char: '|', tokenIdx: 4, frame: 30, seconds: 2.20, endSeconds: 2.22, confidence: 0.5 },
      { char: 't', tokenIdx: 3, frame: 31, seconds: 2.22, endSeconds: 2.24, confidence: 0.8 },
    ];

    const words = groupCharAlignmentToWords(frames, 'of t', ' ');
    expect(words).toHaveLength(2);
    expect(words[0]!.text).toBe('of');
    expect(words[0]!.end).toBeCloseTo(1.64, 5);
    expect(words[1]!.text).toBe('t');
    expect(words[1]!.start).toBeCloseTo(2.22, 5);
  });

  it('caps short-word duration when the last letter is parked far from the first', () => {
    const frames: CtcAlignmentResult['alignedFrames'] = [
      { char: 'o', tokenIdx: 1, frame: 80, seconds: 1.60, endSeconds: 1.62, confidence: 0.9 },
      { char: 'f', tokenIdx: 2, frame: 109, seconds: 2.18, endSeconds: 2.20, confidence: 0.9 },
    ];

    const words = groupCharAlignmentToWords(frames, 'of', ' ');
    expect(words).toHaveLength(1);
    expect(words[0]!.start).toBeCloseTo(1.60, 5);
    expect(words[0]!.end).toBeCloseTo(1.80, 5);
  });

  it('handles single word', () => {
    const frames: CtcAlignmentResult['alignedFrames'] = [
      { char: 'a', tokenIdx: 1, frame: 0, seconds: 0.0, confidence: 0.9 },
      { char: 'b', tokenIdx: 2, frame: 1, seconds: 0.02, confidence: 0.8 },
    ];

    const words = groupCharAlignmentToWords(frames, 'ab', ' ');
    expect(words).toHaveLength(1);
    expect(words[0]!.text).toBe('ab');
    expect(words[0]!.start).toBe(0.0);
    expect(words[0]!.end).toBe(0.02);
  });

  it('handles empty text', () => {
    const words = groupCharAlignmentToWords([], '', ' ');
    expect(words).toHaveLength(0);
  });

  it('splits on custom separator', () => {
    const frames: CtcAlignmentResult['alignedFrames'] = [
      { char: 'a', tokenIdx: 1, frame: 0, seconds: 0.0, confidence: 0.9 },
      { char: 'b', tokenIdx: 2, frame: 1, seconds: 0.02, confidence: 0.8 },
      { char: 'c', tokenIdx: 3, frame: 2, seconds: 0.04, confidence: 0.7 },
    ];

    const words = groupCharAlignmentToWords(frames, 'a|bc', '|');
    expect(words).toHaveLength(2);
    expect(words[0]!.text).toBe('a');
    expect(words[1]!.text).toBe('bc');
  });

  it('compute words confidence as avg char confidence', () => {
    const frames: CtcAlignmentResult['alignedFrames'] = [
      { char: 'x', tokenIdx: 1, frame: 0, seconds: 0.0, confidence: 1.0 },
      { char: 'y', tokenIdx: 2, frame: 1, seconds: 0.02, confidence: 0.5 },
    ];

    const words = groupCharAlignmentToWords(frames, 'xy', ' ');
    expect(words[0]!.confidence).toBeCloseTo(0.75, 2);
  });
});

// ---------------------------------------------------------------------------
// Tests: createWav2Vec2Aligner (integration)
// ---------------------------------------------------------------------------

describe('createWav2Vec2Aligner', () => {
  const VOCAB = 10;
  const BLANK = 0;
  const SEPARATOR = ' ';
  const CHARS: Record<string, number> = { ' ': 0, 'h': 1, 'e': 2, 'l': 3, 'o': 4, 'w': 5, 'r': 6, 'd': 7 };

  it('aligns transcript to audio via CTC Viterbi', () => {
    const frameCount = 11; // 2 words * 5 chars + blanks
    const logitProvider = makeLogitProvider(frameCount, VOCAB, new Map([
      [1, [1]], [2, [2]], [3, [3]], [3, [4]], [4, [5]],  // hello
      [5, [6]], [4, [7]], [6, [8]], [3, [9]], [7, [10]], // world
    ]), BLANK);

    const tokenizer = simpleTokenizer(CHARS);
    const config: Wav2Vec2AlignerConfig = {
      logitProvider,
      tokenizer: tokenizer as any,
      vocabSize: VOCAB,
      blankId: BLANK,
      wordSeparator: SEPARATOR,
      frameCount,
      sampleRate: 16000,
    };

    const aligner = createWav2Vec2Aligner(config);
    const result = aligner.align({ transcript: 'hello world', audioDurationSeconds: 2.2 });

    expect(result.words).toHaveLength(2);
    expect(result.words[0]!.text).toBe('hello');
    expect(result.words[1]!.text).toBe('world');
    expect(result.words[0]!.start).toBeLessThan(result.words[1]!.start);
  });

  it('uses decoded separator token labels so spaces do not consume word characters', () => {
    const frameCount = 8;
    const separatorAwareChars: Record<string, number> = {
      ' ': 4,
      h: 1,
      i: 2,
      y: 5,
      o: 6,
      u: 7,
    };
    const logitProvider = makeLogitProvider(frameCount, VOCAB, new Map([
      [1, [1]], // h
      [2, [2]], // i
      [4, [3]], // separator / space
      [5, [4]], // y
      [6, [5]], // o
      [7, [6]], // u
    ]), BLANK);
    const tokenizer = simpleTokenizer(separatorAwareChars);
    const config: Wav2Vec2AlignerConfig = {
      logitProvider,
      tokenizer,
      vocabSize: VOCAB,
      blankId: BLANK,
      wordSeparator: SEPARATOR,
      frameCount,
      sampleRate: 16000,
    };

    const aligner = createWav2Vec2Aligner(config);
    const result = aligner.align({ transcript: 'hi you', audioDurationSeconds: 1.6 });

    expect(result.words).toHaveLength(2);
    expect(result.totalChars).toBe(6);
    expect(result.words[0]!.text).toBe('hi');
    expect(result.words[0]!.charFrames.map((frame) => frame.tokenIdx)).toEqual([1, 2]);
    expect(result.words[1]!.text).toBe('you');
    expect(result.words[1]!.charFrames.map((frame) => frame.tokenIdx)).toEqual([5, 6, 7]);
    expect(result.words[1]!.start).toBeCloseTo(0.8, 1);
  });

  it('creates an aligner directly from reusable Wav2Vec2 logits', () => {
    const frameCount = 8;
    const tokenizer = simpleTokenizer({ ' ': 4, h: 1, i: 2, y: 5, o: 6, u: 7 });
    const logits = makeLogitProvider(frameCount, VOCAB, new Map([
      [1, [1]],
      [2, [2]],
      [4, [3]],
      [5, [4]],
      [6, [5]],
      [7, [6]],
    ]), BLANK)();

    const aligner = createWav2Vec2AlignerFromLogits({
      logits,
      frameCount,
      vocabSize: VOCAB,
      blankId: BLANK,
      tokenizer,
      sampleRate: 16000,
      audioDurationSeconds: 1.6,
      wordSeparator: SEPARATOR,
    });

    const result = aligner.align({ transcript: 'hi you' });

    expect(result.words.map((word) => word.text)).toEqual(['hi', 'you']);
    expect(result.words[1]!.start).toBeCloseTo(0.8, 1);
  });

  it('returns empty words for empty transcript', () => {
    const logitProvider = makeLogitProvider(5, VOCAB, new Map(), BLANK);
    const tokenizer = simpleTokenizer(CHARS);
    const config: Wav2Vec2AlignerConfig = {
      logitProvider,
      tokenizer: tokenizer as any,
      vocabSize: VOCAB,
      blankId: BLANK,
      wordSeparator: SEPARATOR,
      frameCount: 5,
      sampleRate: 16000,
    };

    const aligner = createWav2Vec2Aligner(config);
    const result = aligner.align({ transcript: '', audioDurationSeconds: 1.0 });

    expect(result.words).toHaveLength(0);
    expect(result.totalFrames).toBe(5);
  });

  it('timestamps are monotonic', () => {
    const frameCount = 8;
    const logitProvider = makeLogitProvider(frameCount, VOCAB, new Map([
      [1, [1]],
      [2, [2]],
      [5, [4]],
      [7, [5]],
    ]), BLANK);

    const tokenizer = simpleTokenizer({ ' ': 0, 'a': 1, 'b': 2, 'c': 5, 'd': 7 });
    const config: Wav2Vec2AlignerConfig = {
      logitProvider,
      tokenizer: tokenizer as any,
      vocabSize: VOCAB,
      blankId: BLANK,
      wordSeparator: ' ',
      frameCount,
      sampleRate: 16000,
    };

    const aligner = createWav2Vec2Aligner(config);
    const result = aligner.align({ transcript: 'ab cd', audioDurationSeconds: 1.6 });

    expect(result.words).toHaveLength(2);
    expect(result.words[0]!.start).toBeLessThanOrEqual(result.words[0]!.end);
    expect(result.words[1]!.start).toBeLessThanOrEqual(result.words[1]!.end);
    expect(result.words[0]!.end).toBeLessThanOrEqual(result.words[1]!.start);
  });
});
