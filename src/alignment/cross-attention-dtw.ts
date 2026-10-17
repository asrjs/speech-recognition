/**
 * Cross-Attention DTW Alignment Module
 *
 * Model-agnostic Dynamic Time Warping over cross-attention matrices.
 * Extracted from Whisper's attention-alignment pipeline but works with
 * any encoder-decoder attention output.
 *
 * Independently importable: @asrjs/speech-recognition/alignment
 */

/**
 * Run DTW with negative attention cost (maximizes attention path).
 *
 * @param matrix Flattened attention matrix [tokens x frames] in row-major order.
 *   Higher values indicate stronger attention (better alignment).
 * @param tokenCount Number of decoder tokens (rows).
 * @param frameCount Number of encoder frames (columns).
 * @returns Token-to-frame alignment indices from the DTW path.
 */
export function dynamicTimeWarpNegative(
  matrix: Float32Array,
  tokenCount: number,
  frameCount: number,
): {
  readonly textIndices: readonly number[];
  readonly timeIndices: readonly number[];
} {
  const rows = tokenCount;
  const cols = frameCount;

  // Match whisper.timing.dtw_cpu: the accumulated cost is float32 even
  // though the normalized attention matrix is later consumed as a negative
  // float64-like score by the reference implementation.
  const cost = Array.from({ length: rows + 1 }, () => new Float32Array(cols + 1).fill(Infinity));
  const trace = Array.from({ length: rows + 1 }, () => new Uint8Array(cols + 1).fill(255));
  cost[0]![0] = 0;

  // Forward pass: fill cost and trace
  for (let row = 1; row <= rows; row++) {
    for (let col = 1; col <= cols; col++) {
      const diagonal = cost[row - 1]![col - 1]!;
      const up = cost[row - 1]![col]!;
      const left = cost[row]![col - 1]!;

      // Match OpenAI Whisper's dtw_cpu tie-breaking. In particular, a tie
      // falls through to the horizontal move, which keeps this path aligned
      // with the reference jump-time extraction on flat attention regions.
      let best: number;
      let direction: number;
      if (diagonal < up && diagonal < left) {
        best = diagonal;
        direction = 0;
      } else if (up < diagonal && up < left) {
        best = up;
        direction = 1;
      } else {
        best = left;
        direction = 2;
      }

      cost[row]![col] = best - (matrix[(row - 1) * cols + (col - 1)] ?? 0);
      trace[row]![col] = direction;
    }
  }

  // The reference backtrace explicitly walks both zero-cost borders. Those
  // dummy row indices are useful for finding the first real row's jump time;
  // computeWhisperDtwTokenTimestamps ignores the negative row itself.
  for (let col = 0; col <= cols; col++) trace[0]![col] = 2;
  for (let row = 0; row <= rows; row++) trace[row]![0] = 1;

  // Backtrace: follow trace directions from bottom-right to top-left
  const textIndices: number[] = [];
  const timeIndices: number[] = [];
  let row = rows;
  let col = cols;

  while (row > 0 || col > 0) {
    textIndices.push(row - 1);
    timeIndices.push(col - 1);

    const direction = trace[row]![col];
    if (direction === 0) {
      row--;
      col--;
    } else if (direction === 1) {
      row--;
    } else {
      col--;
    }
  }

  textIndices.reverse();
  timeIndices.reverse();
  return { textIndices, timeIndices };
}

/**
 * Average multiple attention head matrices into a single matrix.
 * All matrices must have the same dimensions (tokenCount × frameCount).
 */
function averageHeads(
  heads: readonly Float32Array[],
  tokenCount: number,
  frameCount: number,
): Float32Array {
  const length = tokenCount * frameCount;
  const output = new Float32Array(length);
  const invCount = 1 / heads.length;

  for (const head of heads) {
    for (let i = 0; i < length; i++) {
      output[i] = (output[i] ?? 0) + (head[i] ?? 0) * invCount;
    }
  }
  return output;
}

/**
 * Compute per-token timestamps from cross-attention matrices using DTW alignment.
 *
 * This is a pure, model-agnostic function. It takes one or more pre-processed
 * attention matrices (e.g. after softmax + normalization), averages them across
 * heads, finds the optimal monotonic alignment path via dynamic time warping,
 * and returns a timestamp (in seconds) for each token.
 *
 * @param attentionMatrices - One or more flattened attention matrices.
 *   Each matrix has shape [tokens.length × numFrames] in row-major order.
 *   Higher values mean stronger attention / better alignment.
 * @param tokens - Token IDs corresponding to the rows of the attention matrices.
 * @param numFrames - Number of encoder frames (columns of each matrix).
 * @param frameDurationSeconds - Duration of each encoder frame in seconds.
 *   Default: 0.02 (20 ms, standard for Whisper encoders).
 * @returns Array of timestamps with length `tokens.length + 1`.
 *   Index i (0 ≤ i < tokens.length) is the start time for token i.
 *   The last element is the end time of the final token.
 *   Timestamps are guaranteed monotonic non-decreasing.
 */
export function crossAttentionDtwTimestamps(
  attentionMatrices: readonly Float32Array[],
  tokens: readonly number[],
  numFrames: number,
  frameDurationSeconds = 0.02,
): readonly number[] {
  const tokenCount = tokens.length;
  const frameCount = Math.max(0, Math.floor(numFrames));

  if (tokenCount === 0) {
    return [0];
  }
  if (frameCount === 0) {
    return Array.from({ length: tokenCount + 1 }, () => 0);
  }

  if (attentionMatrices.length === 0) {
    throw new Error('At least one attention matrix is required for DTW alignment.');
  }

  // Validate dimensions
  for (const head of attentionMatrices) {
    const expected = tokenCount * frameCount;
    if (head.length < expected) {
      throw new Error(
        `Attention matrix has ${head.length} values; expected at least ${expected} (${tokenCount} tokens × ${frameCount} frames).`,
      );
    }
  }

  // Average heads → single matrix → DTW
  const matrix =
    attentionMatrices.length === 1
      ? attentionMatrices[0]!
      : averageHeads(attentionMatrices, tokenCount, frameCount);

  const { textIndices, timeIndices } = dynamicTimeWarpNegative(matrix, tokenCount, frameCount);

  // Build timestamps: first-seen token gets its frame index * frameDuration
  const timestamps = new Array<number>(tokenCount + 1).fill(0);
  const seen = new Set<number>();

  for (let i = 0; i < textIndices.length; i++) {
    const token = textIndices[i] ?? 0;
    if (!seen.has(token)) {
      timestamps[token] = (timeIndices[i] ?? 0) * frameDurationSeconds;
      seen.add(token);
    }
  }

  // End timestamp = last frame
  timestamps[tokenCount] = (frameCount - 1) * frameDurationSeconds;

  // Enforce monotonicity
  for (let i = 1; i < timestamps.length; i++) {
    if ((timestamps[i] ?? 0) < (timestamps[i - 1] ?? 0)) {
      timestamps[i] = timestamps[i - 1]!;
    }
  }

  return timestamps;
}
