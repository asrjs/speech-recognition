import {
  ctcForceAlign,
  ctcViterbiBacktrack,
  ctcLogSoftmax,
  type CtcAlignedFrame,
  type CtcAlignmentResult,
} from '@asrjs/speech-recognition/alignment';
import { describe, expect, it } from 'vitest';

// Helper: build logits where frame f has high probability for charIdx,
// medium for blank, low for everything else.
function makeSimpleLogits(
  frameCount: number,
  vocabSize: number,
  charToFrame: Map<number, number[]>, // charIdx → list of frames where it's dominant
  blankId = 0,
): Float32Array {
  const logits = new Float32Array(frameCount * vocabSize);
  for (let f = 0; f < frameCount; f++) {
    // Default: blank=0.5, everything else=0.01
    for (let v = 0; v < vocabSize; v++) {
      logits[f * vocabSize + v] = 0.01;
    }
    logits[f * vocabSize + blankId] = 0.5;
  }
  // Boost specific char at specific frames
  for (const [charIdx, frames] of charToFrame) {
    for (const f of frames) {
      logits[f * vocabSize + charIdx] = 10.0; // dominant
      logits[f * vocabSize + blankId] = 0.001; // suppress blank
    }
  }
  return logits;
}

describe('CTC Viterbi forced alignment', () => {
  const VOCAB = 30;
  const BLANK = 0;

  it('aligns single character to single frame', () => {
    // Frames: [blank_high, 'a'_high]
    // Target: ['a']
    const logits = new Float32Array(2 * VOCAB);
    logits[0 * VOCAB + BLANK] = 10.0;
    logits[1 * VOCAB + 1 /*'a'*/] = 10.0;

    const result = ctcForceAlign(logits, 2, VOCAB, [1], BLANK);

    expect(result.alignedFrames).toHaveLength(1);
    expect(result.alignedFrames[0]!.char).toBe('1'); // char is tokenIdx string
    expect(result.alignedFrames[0]!.tokenIdx).toBe(1);
    expect(result.alignedFrames[0]!.frame).toBe(1);
    expect(result.alignedFrames[0]!.confidence).toBeGreaterThan(0.5);
    expect(result.totalFrames).toBe(2);
  });

  it('aligns "ab" with blanks', () => {
    // Frames: 0=blank, 1='a', 2=blank, 3='b', 4=blank
    const logits = makeSimpleLogits(5, VOCAB, new Map([
      [1, [1]], // 'a' at frame 1
      [2, [3]], // 'b' at frame 3
    ]), BLANK);

    const result = ctcForceAlign(logits, 5, VOCAB, [1, 2], BLANK);
    expect(result.alignedFrames).toHaveLength(2);
    expect(result.alignedFrames[0]!.tokenIdx).toBe(1);
    expect(result.alignedFrames[1]!.tokenIdx).toBe(2);
    // Frames should be monotonic
    expect(result.alignedFrames[0]!.frame).toBeLessThanOrEqual(result.alignedFrames[1]!.frame);
  });

  it('handles repeated characters with blank insertion', () => {
    // Target "aa" — CTC requires blank between repeats: a, blank, a
    // Frames: 1='a', 2=blank, 3='a'
    const logits = makeSimpleLogits(4, VOCAB, new Map([
      [1, [1, 3]], // 'a' at frames 1 and 3
    ]), BLANK);

    const result = ctcForceAlign(logits, 4, VOCAB, [1, 1], BLANK);
    expect(result.alignedFrames).toHaveLength(2);
    // First 'a' at frame 1, second 'a' at frame 3
    expect(result.alignedFrames[0]!.frame).toBe(1);
    expect(result.alignedFrames[1]!.frame).toBe(3);
  });

  it('returns empty result for empty target', () => {
    const logits = makeSimpleLogits(3, VOCAB, new Map(), BLANK);
    const result = ctcForceAlign(logits, 3, VOCAB, [], BLANK);
    expect(result.alignedFrames).toHaveLength(0);
    expect(result.totalFrames).toBe(3);
  });

  it('produces monotonic frame indices', () => {
    // Longer sequence: "hello" -> h,e,l,l,o
    const charIds = [8, 5, 12, 12, 15];
    const logits = makeSimpleLogits(charIds.length * 2 + 1, VOCAB, new Map(
      charIds.map((c, i) => [c, [i * 2 + 1]]),
    ), BLANK);

    const result = ctcForceAlign(logits, charIds.length * 2 + 1, VOCAB, charIds, BLANK);
    expect(result.alignedFrames).toHaveLength(charIds.length);

    for (let i = 1; i < result.alignedFrames.length; i++) {
      expect(result.alignedFrames[i]!.frame).toBeGreaterThanOrEqual(
        result.alignedFrames[i - 1]!.frame,
      );
    }
  });

  it('confidence scores are in [0,1] range', () => {
    const logits = makeSimpleLogits(10, VOCAB, new Map([
      [1, [2, 4, 6, 8]],
    ]), BLANK);

    const result = ctcForceAlign(logits, 10, VOCAB, [1, 1, 1, 1], BLANK);
    for (const f of result.alignedFrames) {
      expect(f.confidence).toBeGreaterThanOrEqual(0);
      expect(f.confidence).toBeLessThanOrEqual(1);
    }
  });

  it('handles single frame with single character (no room for blanks)', () => {
    const logits = new Float32Array(1 * VOCAB);
    logits[0 * VOCAB + 5] = 10.0;

    const result = ctcForceAlign(logits, 1, VOCAB, [5], BLANK);
    expect(result.alignedFrames).toHaveLength(1);
    expect(result.alignedFrames[0]!.frame).toBe(0);
  });

  it('aligns long sequence (10+ chars)', () => {
    const chars = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const frameCount = chars.length * 2 + 3; // blanks + chars + extras
    const logits = makeSimpleLogits(frameCount, VOCAB, new Map(
      chars.map((c, i) => [c, [i * 2 + 1]]),
    ), BLANK);

    const result = ctcForceAlign(logits, frameCount, VOCAB, chars, BLANK);
    expect(result.alignedFrames).toHaveLength(chars.length);
    // Each char should be at approximately its expected frame
    for (let i = 0; i < chars.length; i++) {
      expect(result.alignedFrames[i]!.tokenIdx).toBe(chars[i]);
    }
  });

  it('exposes seconds correctly when duration provided', () => {
    const logits = makeSimpleLogits(5, VOCAB, new Map([
      [1, [1]],
      [2, [3]],
    ]), BLANK);

    const result = ctcForceAlign(logits, 5, VOCAB, [1, 2], BLANK, { audioDurationSeconds: 10 });
    // 5 frames over 10s = 2s per frame
    // 'a' at frame 1 → 2.0s, 'b' at frame 3 → 6.0s
    expect(result.alignedFrames[0]!.seconds).toBeCloseTo(2.0, 0);
    expect(result.alignedFrames[1]!.seconds).toBeCloseTo(6.0, 0);
  });
});

describe('ctcViterbiBacktrack — low-level', () => {
  it('backtracks from trellis correctly', () => {
    // Build a simple trellis and backtrack
    // S = 2*len+1 = 2*2+1 = 5 states: [B, a, B, b, B]
    const targets = [1, 2];
    const frameCount = 5;
    const S = 5;

    // Manually fill alpha and backS so path is unambiguous
    // Frame:   0   1   2   3   4
    // BackS shows: t=0→stay, t=1→B(s=0)→a(s=1), t=2→a(s=1)→B(s=2), t=3→B(s=2)→b(s=3)
    const alpha = new Float64Array(frameCount * S).fill(-Infinity);
    const backS = new Uint16Array(frameCount * S).fill(0);

    alpha[0 * S + 0] = 2.0;
    alpha[1 * S + 0] = 1.8; backS[1 * S + 0] = 0; // stayed at B
    alpha[1 * S + 1] = 0.5; backS[1 * S + 1] = 0; // advanced from B(s=0) to a(s=1)
    alpha[2 * S + 1] = 0.3; backS[2 * S + 1] = 1; // stayed at a
    alpha[2 * S + 2] = 0.8; backS[2 * S + 2] = 1; // advanced from a(s=1) to B(s=2)
    alpha[3 * S + 2] = 0.6; backS[3 * S + 2] = 2; // stayed at B
    alpha[3 * S + 3] = 0.9; backS[3 * S + 3] = 2; // advanced from B(s=2) to b(s=3)
    alpha[4 * S + 3] = 0.7; backS[4 * S + 3] = 3; // stayed at b
    alpha[4 * S + 4] = 0.2; backS[4 * S + 4] = 3; // advanced from b(s=3) to B(s=4)

    const path = ctcViterbiBacktrack(alpha, backS, frameCount, targets.length);
    // Should give frame for each target char: a at t=1, b at t=3
    expect(path).toHaveLength(2);
    expect(path[0]).toBe(1);
    expect(path[1]).toBe(3);
  });

  it('returns ascending frames for repeated chars', () => {
    // Target "aa" → S = 5: [B, a, B, a, B]
    const targets = [1, 1];
    const frameCount = 4;
    const S = 5;

    const alpha = new Float64Array(frameCount * S).fill(-Infinity);
    const backS = new Uint16Array(frameCount * S).fill(0);

    alpha[0 * S + 0] = 2.0;
    alpha[1 * S + 0] = 1.8; backS[1 * S + 0] = 0;
    alpha[1 * S + 1] = 0.5; backS[1 * S + 1] = 0; // a at t=1
    alpha[2 * S + 1] = 0.3; backS[2 * S + 1] = 1;
    alpha[2 * S + 2] = 0.8; backS[2 * S + 2] = 1; // B at t=2
    alpha[3 * S + 2] = 0.6; backS[3 * S + 2] = 2;
    alpha[3 * S + 3] = 0.9; backS[3 * S + 3] = 2; // a at t=3

    const path = ctcViterbiBacktrack(alpha, backS, frameCount, targets.length);
    expect(path).toHaveLength(2);
    expect(path[0]).toBe(1);
    expect(path[1]).toBe(3);
  });
});

describe('ctcLogSoftmax', () => {
  it('computes log-softmax correctly', () => {
    // Single frame, small vocab
    const logits = new Float32Array([2.0, 1.0, 0.1]);
    const result = ctcLogSoftmax(logits, 1, 3);

    // log_softmax(x_i) = x_i - log(sum(exp(x_j)))
    // sum(exp) = e^2 + e^1 + e^0.1 ≈ 7.389 + 2.718 + 1.105 = 11.212
    // log(11.212) ≈ 2.417
    // log_softmax(2.0) = 2.0 - 2.417 = -0.417
    // log_softmax(1.0) = 1.0 - 2.417 = -1.417
    expect(result[0]).toBeCloseTo(-0.417, 2);
    expect(result[1]).toBeCloseTo(-1.417, 2);
  });

  it('handles all-zeros (uniform distribution)', () => {
    const logits = new Float32Array([0, 0, 0]);
    const result = ctcLogSoftmax(logits, 1, 3);
    // log(1/3) = -1.0986
    for (let i = 0; i < 3; i++) {
      expect(result[i]).toBeCloseTo(-1.099, 2);
    }
  });

  it('handles multi-frame', () => {
    // 2 frames, 2 vocab
    const logits = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const result = ctcLogSoftmax(logits, 2, 2);
    // Frame 0: [1,2] → log_softmax = [-1.313, -0.313]
    expect(result[0]).toBeCloseTo(-1.313, 2);
    expect(result[1]).toBeCloseTo(-0.313, 2);
    // Frame 1: [3,4] → log_softmax = [-1.313, -0.313]  (same differences)
    expect(result[2]).toBeCloseTo(-1.313, 2);
    expect(result[3]).toBeCloseTo(-0.313, 2);
  });
});
