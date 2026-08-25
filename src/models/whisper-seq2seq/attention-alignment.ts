export interface WhisperAttentionHeadMatrix {
  /** Post-softmax cross-attention weights, laid out as token-major rows. */
  readonly values: Float32Array;
  readonly tokenCount: number;
  readonly frameCount: number;
}

export interface WhisperMedianFilterOptions {
  readonly tokenCount: number;
  readonly frameCount: number;
  readonly width: number;
}

export interface WhisperDtwTokenTimestampOptions {
  readonly attentionHeads: readonly WhisperAttentionHeadMatrix[];
  readonly tokenCount: number;
  readonly frameCount: number;
  readonly medianFilterWidth?: number;
  readonly timePrecisionSeconds?: number;
}

function reflectIndex(index: number, length: number): number {
  if (length <= 1) return 0;
  if (index < 0) return -index;
  if (index >= length) return 2 * length - index - 2;
  return index;
}

export function medianFilterWhisperAttention(
  values: Float32Array,
  options: WhisperMedianFilterOptions,
): Float32Array {
  const { tokenCount, frameCount } = options;
  const width = Math.max(1, Math.floor(options.width));
  if (width <= 1) return new Float32Array(values);
  if (values.length !== tokenCount * frameCount) {
    throw new Error(
      `Whisper attention matrix has ${values.length} values; expected ${tokenCount * frameCount}.`,
    );
  }

  const radius = Math.floor(width / 2);
  const output = new Float32Array(values.length);
  const window: number[] = [];
  for (let token = 0; token < tokenCount; token++) {
    const rowOffset = token * frameCount;
    for (let frame = 0; frame < frameCount; frame++) {
      window.length = 0;
      for (let offset = -radius; offset <= radius; offset++) {
        window.push(values[rowOffset + reflectIndex(frame + offset, frameCount)] ?? 0);
      }
      window.sort((a, b) => a - b);
      output[rowOffset + frame] = window[Math.floor(window.length / 2)] ?? 0;
    }
  }
  return output;
}

function normalizeOverTokens(values: Float32Array, tokenCount: number, frameCount: number): Float32Array {
  const output = new Float32Array(values.length);
  for (let frame = 0; frame < frameCount; frame++) {
    let sum = 0;
    for (let token = 0; token < tokenCount; token++) sum += values[token * frameCount + frame] ?? 0;
    const mean = sum / tokenCount;
    let variance = 0;
    for (let token = 0; token < tokenCount; token++) {
      const delta = (values[token * frameCount + frame] ?? 0) - mean;
      variance += delta * delta;
    }
    const std = Math.sqrt(variance / tokenCount) || 1;
    for (let token = 0; token < tokenCount; token++) {
      const index = token * frameCount + frame;
      output[index] = ((values[index] ?? 0) - mean) / std;
    }
  }
  return output;
}

/**
 * Restrict a padded attention row to the real encoder frames.
 *
 * decoder_align exports post-softmax cross-attention weights over the fixed
 * 30-second encoder axis. When a short clip is cropped, the remaining mass
 * must be renormalized; applying softmax again would treat probabilities as
 * logits and flatten the alignment distribution.
 */
function renormalizeOverFrames(values: Float32Array, tokenCount: number, frameCount: number): Float32Array {
  const output = new Float32Array(values.length);
  for (let token = 0; token < tokenCount; token++) {
    const rowOffset = token * frameCount;
    let sum = 0;
    for (let frame = 0; frame < frameCount; frame++) {
      sum += Math.max(0, values[rowOffset + frame] ?? 0);
    }
    if (sum > 0) {
      for (let frame = 0; frame < frameCount; frame++) {
        output[rowOffset + frame] = Math.max(0, values[rowOffset + frame] ?? 0) / sum;
      }
    } else {
      const uniform = 1 / frameCount;
      for (let frame = 0; frame < frameCount; frame++) {
        output[rowOffset + frame] = uniform;
      }
    }
  }
  return output;
}

function averageHeads(heads: readonly Float32Array[], tokenCount: number, frameCount: number): Float32Array {
  const output = new Float32Array(tokenCount * frameCount);
  for (const head of heads) {
    for (let i = 0; i < output.length; i++) output[i] = (output[i] ?? 0) + (head[i] ?? 0) / heads.length;
  }
  return output;
}

function dynamicTimeWarpNegative(matrix: Float32Array, tokenCount: number, frameCount: number): {
  readonly textIndices: readonly number[];
  readonly timeIndices: readonly number[];
} {
  const rows = tokenCount;
  const cols = frameCount;
  const cost = Array.from({ length: rows + 1 }, () => new Float64Array(cols + 1).fill(Infinity));
  const trace = Array.from({ length: rows + 1 }, () => new Uint8Array(cols + 1));
  cost[0]![0] = 0;

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

  const textIndices: number[] = [];
  const timeIndices: number[] = [];
  let row = rows;
  let col = cols;
  while (row > 0 && col > 0) {
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

export function computeWhisperDtwTokenTimestamps(
  options: WhisperDtwTokenTimestampOptions,
): readonly number[] {
  const tokenCount = Math.max(0, Math.floor(options.tokenCount));
  const frameCount = Math.max(0, Math.floor(options.frameCount));
  if (tokenCount === 0) return [0];
  if (frameCount === 0) return Array.from({ length: tokenCount + 1 }, () => 0);
  if (options.attentionHeads.length === 0) {
    throw new Error('At least one Whisper cross-attention head is required for DTW alignment.');
  }

  const processedHeads = options.attentionHeads.map((head) => {
    if (head.tokenCount < tokenCount || head.frameCount < frameCount) {
      throw new Error('Whisper attention head is smaller than the requested DTW crop.');
    }
    const cropped = new Float32Array(tokenCount * frameCount);
    for (let token = 0; token < tokenCount; token++) {
      const sourceOffset = token * head.frameCount;
      const targetOffset = token * frameCount;
      cropped.set(head.values.subarray(sourceOffset, sourceOffset + frameCount), targetOffset);
    }
    const renormalized = renormalizeOverFrames(cropped, tokenCount, frameCount);
    const normalized = normalizeOverTokens(renormalized, tokenCount, frameCount);
    return medianFilterWhisperAttention(normalized, {
      tokenCount,
      frameCount,
      width: options.medianFilterWidth ?? 7,
    });
  });

  const matrix = averageHeads(processedHeads, tokenCount, frameCount);
  const { textIndices, timeIndices } = dynamicTimeWarpNegative(matrix, tokenCount, frameCount);
  const precision = options.timePrecisionSeconds ?? 0.02;
  const timestamps = new Array<number>(tokenCount + 1).fill(0);
  let previousToken = -1;
  for (let i = 0; i < textIndices.length; i++) {
    const token = textIndices[i] ?? 0;
    // OpenAI Whisper and faster-whisper use the first frame of every DTW
    // token-index jump as the token boundary. The path visits each row in
    // order, so this is equivalent to a first-seen map while making the
    // reference semantics explicit.
    if (token !== previousToken && token >= 0 && token < tokenCount) {
      timestamps[token] = (timeIndices[i] ?? 0) * precision;
    }
    previousToken = token;
  }
  timestamps[tokenCount] = (timeIndices[timeIndices.length - 1] ?? 0) * precision;
  for (let i = 1; i < timestamps.length; i++) {
    if (timestamps[i]! < timestamps[i - 1]!) timestamps[i] = timestamps[i - 1]!;
  }
  return spreadWhisperDtwTimestamps(timestamps);
}

/** Linearly interpolate runs of identical jump times so no text token has zero duration. */
export function spreadWhisperDtwTimestamps(timestamps: readonly number[]): number[] {
  const output = [...timestamps];
  let index = 0;
  while (index < output.length - 1) {
    let next = index + 1;
    while (next < output.length && output[next] === output[index]) next++;
    if (next > index + 1 && next < output.length) {
      const start = output[index]!;
      const end = output[next]!;
      const steps = next - index;
      if (end > start) {
        for (let offset = 1; offset < steps; offset++) {
          output[index + offset] = start + ((end - start) * offset) / steps;
        }
      }
    }
    index = next;
  }
  return output;
}
