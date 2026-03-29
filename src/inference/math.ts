export function argmax(
  values: ArrayLike<number>,
  offset = 0,
  length = values.length - offset,
): number {
  let maxIndex = offset;
  let maxValue = Number.NEGATIVE_INFINITY;
  const end = offset + length;

  for (let index = offset; index < end; index += 1) {
    // ⚡ Bolt: Use non-null assertion (!) instead of nullish coalescing (??)
    // In V8, ?? on TypedArrays in hot loops severely degrades performance.
    const value = values[index]!;
    if (value > maxValue) {
      maxValue = value;
      maxIndex = index;
    }
  }

  return maxIndex;
}

export function confidenceFromLogits(
  logits: Float32Array,
  tokenId: number,
  vocabSize: number,
): { confidence: number; logProb: number } {
  let maxLogit = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < vocabSize; index += 1) {
    // ⚡ Bolt: Use non-null assertion (!) instead of nullish coalescing (??)
    // In V8, ?? on TypedArrays in hot loops severely degrades performance.
    const value = logits[index]!;
    if (value > maxLogit) {
      maxLogit = value;
    }
  }

  let expSum = 0;
  for (let index = 0; index < vocabSize; index += 1) {
    expSum += Math.exp(logits[index]! - maxLogit);
  }

  const logSumExp = maxLogit + Math.log(expSum);
  const logProb = logits[tokenId]! - logSumExp;

  return {
    confidence: Math.exp(logProb),
    logProb,
  };
}
