/**
 * CTC Viterbi forced alignment — shared, model-agnostic.
 *
 * Any CTC model (WAV2VEC2, MedASR, future) produces frame-level logits;
 * this module aligns a known transcript token sequence to those frames.
 *
 * Algorithm: standard CTC Viterbi alignment from Graves et al.
 *   - Expanded state space: S = 2*N + 1  (blank between each token)
 *   - Forward pass computes alpha[t][s] with back-pointer tracking
 *   - Backtrack reconstructs optimal frame for each target token
 *
 * @module alignment/ctc-viterbi
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CtcAlignedFrame {
  readonly char: string;
  readonly tokenIdx: number;
  readonly frame: number;
  readonly seconds: number;
  readonly confidence: number;
}

export interface CtcAlignmentResult {
  readonly alignedFrames: readonly CtcAlignedFrame[];
  readonly totalFrames: number;
  readonly totalTokens: number;
  readonly audioDurationSeconds?: number;
}

export interface CtcForceAlignOptions {
  /** Optional audio duration for seconds conversion
   *  seconds = (frame + 0.5) / frameCount * audioDurationSeconds */
  readonly audioDurationSeconds?: number;
}

// ---------------------------------------------------------------------------
// Log-Softmax
// ---------------------------------------------------------------------------

/**
 * Compute log-softmax per frame. Numerically stable.
 */
export function ctcLogSoftmax(
  logits: Float32Array,
  frameCount: number,
  vocabSize: number,
): Float64Array {
  const result = new Float64Array(frameCount * vocabSize);

  for (let t = 0; t < frameCount; t++) {
    const offset = t * vocabSize;

    let maxVal = -Infinity;
    for (let v = 0; v < vocabSize; v++) {
      if (logits[offset + v]! > maxVal) {
        maxVal = logits[offset + v]!;
      }
    }

    let sumExp = 0.0;
    for (let v = 0; v < vocabSize; v++) {
      sumExp += Math.exp(logits[offset + v]! - maxVal);
    }

    const logSum = Math.log(sumExp);
    for (let v = 0; v < vocabSize; v++) {
      result[offset + v] = logits[offset + v]! - maxVal - logSum;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sStateToTokenIdx(
  s: number,
  targets: readonly number[],
  blankId: number,
): number {
  if (s % 2 === 0) return blankId;
  return targets[(s - 1) / 2]!;
}

/**
 * Is state s a character state (odd)?
 */
function isCharState(s: number): boolean {
  return s % 2 === 1;
}

// ---------------------------------------------------------------------------
// Viterbi Forward Pass (with back-pointer register)
// ---------------------------------------------------------------------------

/**
 * Run CTC Viterbi forward pass storing back-pointers.
 *
 * @returns { alpha, backS: Uint16Array, backT: Uint8Array }
 *   alpha: [T*S] log probs
 *   backS: [T*S] previous state that yielded the best path to (t,s)
 *   backT: same width, stores 0=stay-at-t-1, 1=advance-from-s-1, 2=skip-from-s-2
 *   (backT layout matches alpha, in practice we store backS only and recompute transitions)
 */
function ctcViterbiForward(
  logProbs: Float64Array, // [T*V]
  frameCount: number,
  vocabSize: number,
  targets: readonly number[],
  blankId: number,
): { alpha: Float64Array; backS: Uint16Array } {
  const S = 2 * targets.length + 1;
  const total = frameCount * S;
  const alpha = new Float64Array(total);
  const backS = new Uint16Array(total);
  alpha.fill(-Infinity);
  backS.fill(0);

  // Initialize t=0
  const logProbs0 = 0 * vocabSize;
  alpha[0] = logProbs[logProbs0 + blankId]!;
  if (S > 1) {
    alpha[1] = logProbs[logProbs0 + targets[0]!]!;
  }

  // Forward pass t = 1..
  for (let t = 1; t < frameCount; t++) {
    const logProbsT = t * vocabSize;
    const alphaPrev = (t - 1) * S;
    const alphaCurr = t * S;

    for (let s = 0; s < S; s++) {
      const tokenIdx = sStateToTokenIdx(s, targets, blankId);
      const emission = logProbs[logProbsT + tokenIdx]!;

      let bestScore = -Infinity;
      let bestPrevS = 0;

      // (1) Stay: s → s from t-1
      const stay = alpha[alphaPrev + s]!;
      if (stay > bestScore) { bestScore = stay; bestPrevS = s; }

      // (2) Advance from s-1
      if (s > 0) {
        const advance = alpha[alphaPrev + s - 1]!;
        if (advance > bestScore) { bestScore = advance; bestPrevS = s - 1; }
      }

      // (3) Skip-blank: from s-2 (if s>=2 and chars at s and s-2 differ)
      if (s >= 2) {
        const tokenS = tokenIdx;
        const tokenSMinus2 = sStateToTokenIdx(s - 2, targets, blankId);
        if (tokenS !== tokenSMinus2) {
          const skip = alpha[alphaPrev + s - 2]!;
          if (skip > bestScore) { bestScore = skip; bestPrevS = s - 2; }
        }
      }

      alpha[alphaCurr + s] = bestScore + emission;
      backS[alphaCurr + s] = bestPrevS;
    }
  }

  return { alpha, backS };
}

// ---------------------------------------------------------------------------
// Backtrack
// ---------------------------------------------------------------------------

/**
 * Backtrack through the Viterbi trellis.
 *
 * Uses stored back-pointers to reconstruct the best path.
 *
 * @returns number[] where path[i] = frame index of the i-th target token.
 */
export function ctcViterbiBacktrack(
  alpha: Float64Array,
  backS: Uint16Array,
  frameCount: number,
  targetLength: number,
): number[] {
  const S = 2 * targetLength + 1;
  const path = new Array<number>(targetLength).fill(-1);

  if (targetLength === 0) return path;

  // Start from best final state at t = frameCount - 1
  let t = frameCount - 1;
  const alphaLastStart = t * S;
  let s: number;

  const lastBlankScore = alpha[alphaLastStart + S - 1]!;
  const lastCharScore = S >= 2 ? alpha[alphaLastStart + S - 2]! : -Infinity;

  if (lastCharScore >= lastBlankScore) {
    s = S - 2;
  } else {
    s = S - 1;
  }

  // Walk backward. Record character frames as we EXIT them.
  // When at (t, s=char), check if backS[s] != s (we entered at time t).
  // If backS[s] == s, we were already in this char at time t-1 — keep going.
  let charIdx = targetLength - 1;

  while (t >= 0 && charIdx >= 0) {
    const prevS = t > 0 ? backS[t * S + s]! : s;

    if (isCharState(s)) {
      // We are at a character state. Check if we just entered it.
      if (t === 0 || prevS !== s) {
        // Just entered this character at time t (or at boundary)
        path[charIdx] = t;
        charIdx--;
      }
    }

    if (t === 0) break;

    s = prevS;
    t--;
  }

  return path;
}

// ---------------------------------------------------------------------------
// Main: ctcForceAlign
// ---------------------------------------------------------------------------

/**
 * CTC forced alignment: find best frame for each target token.
 *
 * @param logits - Raw CTC logits [T*V] row-major Float32Array
 * @param frameCount - Number of time frames
 * @param vocabSize - CTC vocabulary size
 * @param targets - Target token sequence to align to
 * @param blankId - CTC blank token index (default 0)
 * @param options - Optional audioDurationSeconds for timestamp conversion
 */
export function ctcForceAlign(
  logits: Float32Array,
  frameCount: number,
  vocabSize: number,
  targets: readonly number[],
  blankId: number = 0,
  options?: CtcForceAlignOptions,
): CtcAlignmentResult {
  // Edge case: empty targets
  if (targets.length === 0) {
    return {
      alignedFrames: [],
      totalFrames: frameCount,
      totalTokens: 0,
      audioDurationSeconds: options?.audioDurationSeconds,
    };
  }

  // Edge case: single frame
  if (frameCount === 1) {
    const secondsPerFrame = options?.audioDurationSeconds
      ? options.audioDurationSeconds / frameCount
      : 0;
    return {
      alignedFrames: targets.map((tokenIdx) => ({
        char: String(tokenIdx),
        tokenIdx,
        frame: 0,
        seconds: secondsPerFrame * 0.5,
        confidence: 1.0,
      })),
      totalFrames: frameCount,
      totalTokens: targets.length,
      audioDurationSeconds: options?.audioDurationSeconds,
    };
  }

  // 1. Compute log-softmax
  const logProbs = ctcLogSoftmax(logits, frameCount, vocabSize);

  // 2. Viterbi forward pass with back-pointers
  const { alpha, backS } = ctcViterbiForward(logProbs, frameCount, vocabSize, targets, blankId);

  // 3. Backtrack to get frame per token
  const framePath = ctcViterbiBacktrack(alpha, backS, frameCount, targets.length);

  // 4. Build result with timestamps and confidence
  const secondsPerFrame = options?.audioDurationSeconds
    ? options.audioDurationSeconds / frameCount
    : 0;

  const alignedFrames: CtcAlignedFrame[] = framePath.map((frame, i) => {
    const tokenIdx = targets[i]!;
    const frameOffset = frame * vocabSize;

    // Confidence = softmax probability of this token at this frame
    const logProb = logProbs[frameOffset + tokenIdx]!;
    const confidence = Math.exp(logProb);

    return {
      char: String(tokenIdx),
      tokenIdx,
      frame,
      seconds: frame * secondsPerFrame,
      confidence: Math.min(1.0, Math.max(0.0, confidence)),
    };
  });

  return {
    alignedFrames,
    totalFrames: frameCount,
    totalTokens: targets.length,
    audioDurationSeconds: options?.audioDurationSeconds,
  };
}
