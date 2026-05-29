import { describe, expect, test } from 'vitest';

import {
  CtcDecoder,
  addTimesToTokenSpans,
  argmaxAndSelectedLogProbs,
  buildSentenceTimings,
  buildUtteranceTiming,
  buildWordsFromCharSpans,
  ctcCollapseWithSpans,
  estimateSecondsPerOutputFrame,
} from '../src/ctc/decoder.js';
import type {
  CtcDecodeResult,
  CtcRawTokenSpan,
  CtcSentenceTiming,
  CtcTokenSpan,
  CtcUtteranceTiming,
} from '../src/ctc/types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Minimal tokenizer that satisfies CtcTokenizerLike for testing. */
class SimpleCharTokenizer {
  private readonly vocab: string[];
  readonly blankId: number;

  constructor(vocab: string[], blankId = 0) {
    this.vocab = vocab;
    this.blankId = blankId;
  }

  decode(ids: readonly number[]): string {
    return ids
      .map((id) => this.vocab[id] ?? '')
      .join('');
  }

  decodeTokenPiece(tokenId: number): string {
    return this.vocab[tokenId] ?? '';
  }
}

/** WAV2VEC2-style char tokenizer: 26 letters + pad(0), |(4=space). */
const WAV2VEC_VOCAB = [
  '<pad>',  // 0 — blank
  '<s>',    // 1
  '</s>',   // 2
  '<unk>',  // 3
  '|',      // 4 — word separator (maps to space in decode)
  "'",      // 5
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',   // 6-15
  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',   // 16-25
  'U', 'V', 'W', 'X', 'Y', 'Z',                        // 26-31
];

class Wav2VecCharTokenizer {
  readonly blankId = 0;

  decode(ids: readonly number[]): string {
    return ids
      .map((id) => {
        if (id === 4) return ' '; // | → space
        return WAV2VEC_VOCAB[id] ?? '';
      })
      .join('');
  }

  decodeTokenPiece(tokenId: number): string {
    if (tokenId === 4) return ' ';
    return WAV2VEC_VOCAB[tokenId] ?? '';
  }
}

/** MedASR-style BPE tokenizer. */
class MedAsrBpeTokenizer {
  private readonly vocab: string[];
  readonly blankId = 0;

  constructor() {
    this.vocab = ['<epsilon>', '▁hello', '▁world', '.', '▁next', '▁line', '!'];
  }

  decode(ids: readonly number[]): string {
    return ids
      .map((id) => (this.vocab[id] ?? '').replace(/▁/g, ' '))
      .join('')
      .trim();
  }

  decodeTokenPiece(tokenId: number): string {
    return (this.vocab[tokenId] ?? '').replace(/▁/g, ' ');
  }
}

/** Generate fake logits from frame IDs for deterministic testing. */
function logitsFromFrameIds(
  frameIds: readonly number[],
  vocabSize: number,
  peak = 8,
  floor = -8,
): Float32Array {
  const logits = new Float32Array(frameIds.length * vocabSize);
  for (let frameIndex = 0; frameIndex < frameIds.length; frameIndex += 1) {
    const rowOffset = frameIndex * vocabSize;
    for (let vocabIndex = 0; vocabIndex < vocabSize; vocabIndex += 1) {
      logits[rowOffset + vocabIndex] = floor;
    }
    const targetId = frameIds[frameIndex] ?? 0;
    logits[rowOffset + targetId] = peak;
  }
  return logits;
}

// ===========================================================================
// PARITY TESTS — must produce identical results to old lasr-ctc/ctc.ts
// ===========================================================================

describe('CtcDecoder parity with lasr-ctc/ctc.ts', () => {
  test('argmaxAndSelectedLogProbs: correct frame IDs and log probs', () => {
    const frameIds = [0, 1, 1, 0, 2, 2, 3, 0];
    const vocabSize = 4;
    const logits = logitsFromFrameIds(frameIds, vocabSize);

    const result = argmaxAndSelectedLogProbs(logits, frameIds.length, vocabSize);

    expect(result.frameIds).toEqual(frameIds);
    expect(result.selectedLogProbs.length).toBe(frameIds.length);
    // All log probs should be near 0 for peak=8, floor=-8 (strong confidence)
    for (let i = 0; i < result.selectedLogProbs.length; i += 1) {
      expect(result.selectedLogProbs[i]).toBeCloseTo(0, 1);
    }
  });

  test('ctcCollapseWithSpans: collapses blanks and repeats', () => {
    const frameIds = [0, 1, 1, 0, 2, 2, 3, 0];
    const logProbs = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0]);
    const blankId = 0;

    const result = ctcCollapseWithSpans(frameIds, logProbs, blankId);

    expect(result.collapsedIds).toEqual([1, 2, 3]);
    expect(result.tokenSpans).toHaveLength(3);
    expect(result.tokenSpans[0]).toEqual({
      tokenId: 1,
      startFrame: 1,
      endFrame: 2,
      frameCount: 2,
      averageLogProb: 0,
      confidence: 1,
    });
    expect(result.tokenSpans[1]).toEqual({
      tokenId: 2,
      startFrame: 4,
      endFrame: 5,
      frameCount: 2,
      averageLogProb: 0,
      confidence: 1,
    });
    expect(result.tokenSpans[2]).toEqual({
      tokenId: 3,
      startFrame: 6,
      endFrame: 6,
      frameCount: 1,
      averageLogProb: 0,
      confidence: 1,
    });
  });

  test('ctcCollapseWithSpans: empty input returns empty', () => {
    const result = ctcCollapseWithSpans([], new Float32Array(0), 0);
    expect(result.collapsedIds).toEqual([]);
    expect(result.tokenSpans).toEqual([]);
  });

  test('estimateSecondsPerOutputFrame: from audio duration', () => {
    const result = estimateSecondsPerOutputFrame({
      audioDurationSec: 5.0,
      outFrames: 100,
    });
    expect(result).toBeCloseTo(0.05, 6);
  });

  test('estimateSecondsPerOutputFrame: from input frames', () => {
    const result = estimateSecondsPerOutputFrame({
      inputFrames: 80000,
      inputFrameHopSeconds: 0.01,
      outFrames: 100,
    });
    expect(result).toBeCloseTo(8.0, 6);
  });

  test('estimateSecondsPerOutputFrame: zero out frames returns 0', () => {
    const result = estimateSecondsPerOutputFrame({ outFrames: 0 });
    expect(result).toBe(0);
  });

  test('addTimesToTokenSpans: adds correct timing', () => {
    const tokenizer = new SimpleCharTokenizer(['<blank>', 'A', 'B', 'C']);
    const rawSpans: CtcRawTokenSpan[] = [
      {
        tokenId: 1,
        startFrame: 2,
        endFrame: 4,
        frameCount: 3,
        averageLogProb: -0.1,
        confidence: 0.9,
      },
    ];
    const result = addTimesToTokenSpans(tokenizer, rawSpans, 0.02);
    expect(result).toHaveLength(1);
    const span = result[0]!;
    expect(span.tokenId).toBe(1);
    expect(span.text).toBe('A');
    expect(span.startFrame).toBe(2);
    expect(span.endFrame).toBe(4);
    expect(span.frameCount).toBe(3);
    expect(span.startTime).toBeCloseTo(0.04, 6);
    expect(span.endTime).toBeCloseTo(0.1, 6);
    expect(span.duration).toBeCloseTo(0.06, 6);
    expect(span.confidence).toBe(0.9);
    expect(span.averageLogProb).toBe(-0.1);
  });

  test('buildUtteranceTiming: speech detected', () => {
    const frameIds = [0, 1, 1, 0, 2, 0];
    const logProbs = new Float32Array([0, 0, 0, 0, 0, 0]);
    const result = buildUtteranceTiming(frameIds, logProbs, 0, 0.1);
    expect(result.hasSpeech).toBe(true);
    expect(result.startFrame).toBe(1);
    expect(result.endFrame).toBe(4);
    expect(result.startTime).toBeCloseTo(0.1, 6);
    expect(result.endTime).toBeCloseTo(0.5, 6);
  });

  test('buildUtteranceTiming: no speech', () => {
    const frameIds = [0, 0, 0];
    const logProbs = new Float32Array([0, 0, 0]);
    const result = buildUtteranceTiming(frameIds, logProbs, 0, 0.1);
    expect(result.hasSpeech).toBe(false);
    expect(result.startFrame).toBeNull();
    expect(result.endFrame).toBeNull();
  });

  test('buildSentenceTimings: two sentences', () => {
    const tokenizer = new MedAsrBpeTokenizer();
    const collapsedIds = [1, 2, 3, 4, 5, 6]; // hello world. next line!
    const text = tokenizer.decode(collapsedIds);

    const tokenSpans: CtcTokenSpan[] = collapsedIds.map((id, i) => ({
      tokenId: id,
      text: tokenizer.decodeTokenPiece(id),
      startFrame: i * 10,
      endFrame: i * 10 + 9,
      frameCount: 10,
      startTime: i * 0.1,
      endTime: (i + 1) * 0.1,
      duration: 0.1,
      confidence: 0.99,
      averageLogProb: -0.01,
    }));

    const sentences = buildSentenceTimings(text, tokenizer, collapsedIds, tokenSpans);
    expect(sentences).toHaveLength(2);
    expect(sentences[0]?.text).toBe('hello world.');
    expect(sentences[1]?.text).toBe('next line!');
    expect(sentences[0]?.startTime).toBeCloseTo(0, 6);
    expect(sentences[1]?.startTime).toBeCloseTo(0.3, 6);
  });
});

// ===========================================================================
// WORD BUILDING — char-level (WAV2VEC2 style)
// ===========================================================================

describe('buildWordsFromCharSpans', () => {
  const tokenizer = new Wav2VecCharTokenizer();

  test('builds words from char-level spans with space separator', () => {
    // "HE|LLO|WORLD" → "HE LLO WORLD" (3 words)
    // Token IDs: H=13, E=10, space=4, L=17, L=17, O=20, space=4, W=28, O=20, R=23, L=17, D=9
    const spans: CtcTokenSpan[] = [
      { tokenId: 13, text: 'H', startFrame: 0, endFrame: 0, frameCount: 1, startTime: 0.0, endTime: 0.02, duration: 0.02, confidence: 0.99, averageLogProb: -0.01 },
      { tokenId: 10, text: 'E', startFrame: 1, endFrame: 1, frameCount: 1, startTime: 0.02, endTime: 0.04, duration: 0.02, confidence: 0.98, averageLogProb: -0.02 },
      { tokenId: 4, text: ' ', startFrame: 2, endFrame: 2, frameCount: 1, startTime: 0.04, endTime: 0.06, duration: 0.02, confidence: 0.99, averageLogProb: -0.01 },
      { tokenId: 17, text: 'L', startFrame: 3, endFrame: 3, frameCount: 1, startTime: 0.06, endTime: 0.08, duration: 0.02, confidence: 0.97, averageLogProb: -0.03 },
      { tokenId: 17, text: 'L', startFrame: 4, endFrame: 4, frameCount: 1, startTime: 0.08, endTime: 0.10, duration: 0.02, confidence: 0.97, averageLogProb: -0.03 },
      { tokenId: 20, text: 'O', startFrame: 5, endFrame: 5, frameCount: 1, startTime: 0.10, endTime: 0.12, duration: 0.02, confidence: 0.98, averageLogProb: -0.02 },
      { tokenId: 4, text: ' ', startFrame: 6, endFrame: 6, frameCount: 1, startTime: 0.12, endTime: 0.14, duration: 0.02, confidence: 0.99, averageLogProb: -0.01 },
      { tokenId: 28, text: 'W', startFrame: 7, endFrame: 7, frameCount: 1, startTime: 0.14, endTime: 0.16, duration: 0.02, confidence: 0.99, averageLogProb: -0.01 },
      { tokenId: 20, text: 'O', startFrame: 8, endFrame: 8, frameCount: 1, startTime: 0.16, endTime: 0.18, duration: 0.02, confidence: 0.98, averageLogProb: -0.02 },
      { tokenId: 23, text: 'R', startFrame: 9, endFrame: 9, frameCount: 1, startTime: 0.18, endTime: 0.20, duration: 0.02, confidence: 0.97, averageLogProb: -0.03 },
      { tokenId: 17, text: 'L', startFrame: 10, endFrame: 10, frameCount: 1, startTime: 0.20, endTime: 0.22, duration: 0.02, confidence: 0.97, averageLogProb: -0.03 },
      { tokenId: 9, text: 'D', startFrame: 11, endFrame: 11, frameCount: 1, startTime: 0.22, endTime: 0.24, duration: 0.02, confidence: 0.98, averageLogProb: -0.02 },
    ];

    const words = buildWordsFromCharSpans(spans);
    expect(words).toHaveLength(3);
    expect(words[0]?.text).toBe('HE');
    expect(words[0]?.startTime).toBeCloseTo(0.0, 6);
    expect(words[0]?.endTime).toBeCloseTo(0.04, 6);
    expect(words[1]?.text).toBe('LLO');
    expect(words[2]?.text).toBe('WORLD');
    expect(words[2]?.endTime).toBeCloseTo(0.24, 6);
  });

  test('empty spans produce no words', () => {
    const words = buildWordsFromCharSpans([]);
    expect(words).toEqual([]);
  });

  test('no space separator produces single word', () => {
    const spans: CtcTokenSpan[] = [
      { tokenId: 13, text: 'H', startFrame: 0, endFrame: 0, frameCount: 1, startTime: 0.0, endTime: 0.02, duration: 0.02, confidence: 0.99, averageLogProb: -0.01 },
      { tokenId: 10, text: 'E', startFrame: 1, endFrame: 1, frameCount: 1, startTime: 0.02, endTime: 0.04, duration: 0.02, confidence: 0.98, averageLogProb: -0.02 },
    ];
    const words = buildWordsFromCharSpans(spans);
    expect(words).toHaveLength(1);
    expect(words[0]?.text).toBe('HE');
  });

  test('trailing space does not produce empty word', () => {
    const spans: CtcTokenSpan[] = [
      { tokenId: 13, text: 'H', startFrame: 0, endFrame: 0, frameCount: 1, startTime: 0.0, endTime: 0.02, duration: 0.02, confidence: 0.99, averageLogProb: -0.01 },
      { tokenId: 4, text: ' ', startFrame: 1, endFrame: 1, frameCount: 1, startTime: 0.02, endTime: 0.04, duration: 0.02, confidence: 0.99, averageLogProb: -0.01 },
    ];
    const words = buildWordsFromCharSpans(spans);
    expect(words).toHaveLength(1);
    expect(words[0]?.text).toBe('H');
  });

  test('words carry tokenIds and tokenIndices', () => {
    const spans: CtcTokenSpan[] = [
      { tokenId: 13, text: 'H', startFrame: 0, endFrame: 0, frameCount: 1, startTime: 0.0, endTime: 0.02, duration: 0.02, confidence: 0.99, averageLogProb: -0.01 },
      { tokenId: 10, text: 'E', startFrame: 1, endFrame: 1, frameCount: 1, startTime: 0.02, endTime: 0.04, duration: 0.02, confidence: 0.98, averageLogProb: -0.02 },
      { tokenId: 4, text: ' ', startFrame: 2, endFrame: 2, frameCount: 1, startTime: 0.04, endTime: 0.06, duration: 0.02, confidence: 0.99, averageLogProb: -0.01 },
      { tokenId: 20, text: 'O', startFrame: 3, endFrame: 3, frameCount: 1, startTime: 0.06, endTime: 0.08, duration: 0.02, confidence: 0.98, averageLogProb: -0.02 },
    ];
    const words = buildWordsFromCharSpans(spans);
    expect(words).toHaveLength(2);
    expect(words[0]?.tokenIds).toEqual([13, 10]);
    expect(words[0]?.tokenIndices).toEqual([0, 1]);
    expect(words[1]?.tokenIds).toEqual([20]);
    expect(words[1]?.tokenIndices).toEqual([3]);
  });
});

// ===========================================================================
// CtcDecoder CLASS
// ===========================================================================

describe('CtcDecoder', () => {
  test('decodeFromLogits: full pipeline with BPE tokenizer', () => {
    const tokenizer = new MedAsrBpeTokenizer();
    const decoder = new CtcDecoder({
      blankId: 0,
      vocabSize: 7,
      tokenizer,
    });

    // hello world. next line!
    const frameIds = [0, 1, 1, 0, 2, 2, 3, 0, 4, 4, 5, 6];
    const logits = logitsFromFrameIds(frameIds, 7);

    const result = decoder.decodeFromLogits(logits, frameIds.length, {
      audioDurationSec: 1.2,
    });

    expect(result.text).toBe('hello world. next line!');
    expect(result.collapsedIds).toEqual([1, 2, 3, 4, 5, 6]);
    expect(result.frameIds).toEqual(frameIds);
    expect(result.utterance.hasSpeech).toBe(true);
    expect(result.sentences).toHaveLength(2);
    expect(result.secondsPerFrame).toBeCloseTo(0.1, 6);
    // No word building without wordSeparator
    expect(result.words).toEqual([]);
  });

  test('decodeFromLogits: full pipeline with char tokenizer + word building', () => {
    const tokenizer = new Wav2VecCharTokenizer();
    const decoder = new CtcDecoder({
      blankId: 0,
      vocabSize: 32,
      tokenizer,
      wordSeparator: ' ',
    });

    // "H E [space] O" → token IDs: H=13, E=10, space=4, O=20
    const frameIds = [0, 13, 13, 0, 10, 10, 0, 4, 4, 0, 20, 20, 0];
    const logits = logitsFromFrameIds(frameIds, 32);

    const result = decoder.decodeFromLogits(logits, frameIds.length, {
      audioDurationSec: 0.26,
    });

    expect(result.text).toBe('HE O');
    expect(result.words).toHaveLength(2);
    expect(result.words[0]?.text).toBe('HE');
    expect(result.words[1]?.text).toBe('O');
  });

  test('decodeFromLogits: all-blank frames produce empty result', () => {
    const tokenizer = new SimpleCharTokenizer(['<blank>', 'A']);
    const decoder = new CtcDecoder({
      blankId: 0,
      vocabSize: 2,
      tokenizer,
    });

    const frameIds = [0, 0, 0, 0];
    const logits = logitsFromFrameIds(frameIds, 2);

    const result = decoder.decodeFromLogits(logits, frameIds.length, {
      audioDurationSec: 0.08,
    });

    expect(result.text).toBe('');
    expect(result.collapsedIds).toEqual([]);
    expect(result.utterance.hasSpeech).toBe(false);
    expect(result.sentences).toEqual([]);
    expect(result.words).toEqual([]);
  });

  test('individual methods match combined pipeline', () => {
    const tokenizer = new MedAsrBpeTokenizer();
    const decoder = new CtcDecoder({
      blankId: 0,
      vocabSize: 7,
      tokenizer,
    });

    const frameIds = [0, 1, 1, 0, 2, 3];
    const logits = logitsFromFrameIds(frameIds, 7);

    // Step by step
    const argmaxResult = decoder.argmax(logits, frameIds.length);
    expect(argmaxResult.frameIds).toEqual(frameIds);

    const collapseResult = decoder.collapse(argmaxResult.frameIds, argmaxResult.selectedLogProbs);
    expect(collapseResult.collapsedIds).toEqual([1, 2, 3]);

    const text = tokenizer.decode(collapseResult.collapsedIds);
    expect(text).toBe('hello world.');

    const secPerFrame = decoder.estimateSecondsPerFrame({ audioDurationSec: 0.6, outFrames: frameIds.length });
    expect(secPerFrame).toBeCloseTo(0.1, 6);

    const timedSpans = decoder.addTiming(collapseResult.tokenSpans, secPerFrame);
    expect(timedSpans).toHaveLength(3);

    const utterance = decoder.buildUtterance(argmaxResult.frameIds, argmaxResult.selectedLogProbs, secPerFrame);
    expect(utterance.hasSpeech).toBe(true);

    const sentences = decoder.buildSentences(text, collapseResult.collapsedIds, timedSpans);
    expect(sentences).toHaveLength(1);
    expect(sentences[0]?.text).toBe('hello world.');

    // Combined must match
    const combined = decoder.decodeFromLogits(logits, frameIds.length, { audioDurationSec: 0.6 });
    expect(combined.text).toBe(text);
    expect(combined.collapsedIds).toEqual(collapseResult.collapsedIds);
  });
});

// ===========================================================================
// BACKWARD COMPAT — lasr-ctc/ctc.ts function signatures must be preserved
// ===========================================================================

describe('backward compat: function signatures match lasr-ctc/ctc.ts', () => {
  test('argmaxAndSelectedLogProbs signature is compatible', () => {
    const logits = new Float32Array([8, -8, -8, -8, 8]); // 1 frame, vocab=5
    const result = argmaxAndSelectedLogProbs(logits, 1, 5);
    expect(result.frameIds).toEqual([0]);
    expect(result.selectedLogProbs).toBeInstanceOf(Float32Array);
  });

  test('ctcCollapseWithSpans signature is compatible', () => {
    const result = ctcCollapseWithSpans([1, 1, 0, 2], new Float32Array([0, 0, 0, 0]), 0);
    expect(result.collapsedIds).toEqual([1, 2]);
    expect(result.tokenSpans).toHaveLength(2);
  });

  test('addTimesToTokenSpans signature is compatible', () => {
    const tokenizer = new SimpleCharTokenizer(['', 'A']);
    const result = addTimesToTokenSpans(
      tokenizer,
      [{ tokenId: 1, startFrame: 0, endFrame: 1, frameCount: 2, averageLogProb: 0, confidence: 1 }],
      0.02,
    );
    expect(result).toHaveLength(1);
    expect(result[0]?.text).toBe('A');
  });

  test('buildUtteranceTiming signature is compatible', () => {
    const result = buildUtteranceTiming([1, 0, 2], new Float32Array([0, 0, 0]), 0, 0.1);
    expect(result.hasSpeech).toBe(true);
  });

  test('buildSentenceTimings signature is compatible', () => {
    const tokenizer = new MedAsrBpeTokenizer();
    const spans: CtcTokenSpan[] = [
      { tokenId: 1, text: '▁hello', startFrame: 0, endFrame: 0, frameCount: 1, startTime: 0, endTime: 0.1, duration: 0.1, confidence: 0.99, averageLogProb: -0.01 },
      { tokenId: 2, text: '▁world', startFrame: 1, endFrame: 1, frameCount: 1, startTime: 0.1, endTime: 0.2, duration: 0.1, confidence: 0.99, averageLogProb: -0.01 },
      { tokenId: 3, text: '.', startFrame: 2, endFrame: 2, frameCount: 1, startTime: 0.2, endTime: 0.3, duration: 0.1, confidence: 0.99, averageLogProb: -0.01 },
    ];
    const result = buildSentenceTimings('hello world.', tokenizer, [1, 2, 3], spans);
    expect(result).toHaveLength(1);
    expect(result[0]?.text).toBe('hello world.');
  });

  test('estimateSecondsPerOutputFrame signature is compatible', () => {
    const result = estimateSecondsPerOutputFrame({ audioDurationSec: 2.0, outFrames: 100 });
    expect(result).toBeCloseTo(0.02, 6);
  });
});
