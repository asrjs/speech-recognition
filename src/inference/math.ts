export function argmax(
  values: ArrayLike<number>,
  offset = 0,
  length = values.length - offset,
): number {
  if (length <= 0) {
    return offset;
  }

  let maxIndex = offset;
  // Use non-null assertion (!) instead of nullish coalescing (??) inside hot loop for better V8 perf.
  let maxValue = values[offset]!;
  const end = offset + length;

  for (let index = offset + 1; index < end; index += 1) {
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
  if (vocabSize <= 0) {
    return { confidence: 0, logProb: Number.NEGATIVE_INFINITY };
  }

  // Use non-null assertion (!) instead of nullish coalescing (??) inside hot loop for better V8 perf.
  let maxLogit = logits[0]!;
  for (let index = 1; index < vocabSize; index += 1) {
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
