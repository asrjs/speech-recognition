export function argmax(
  values: ArrayLike<number>,
  offset = 0,
  length = values.length - offset,
): number {
  let maxIndex = offset;
  let maxValue = Number.NEGATIVE_INFINITY;
  const end = offset + length;

  for (let index = offset; index < end; index += 1) {
    const value = values[index] ?? Number.NEGATIVE_INFINITY;
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
    const value = logits[index] ?? Number.NEGATIVE_INFINITY;
    if (value > maxLogit) {
      maxLogit = value;
    }
  }

  let expSum = 0;
  for (let index = 0; index < vocabSize; index += 1) {
    expSum += Math.exp((logits[index] ?? 0) - maxLogit);
  }

  const logSumExp = maxLogit + Math.log(expSum);
  const logProb = (logits[tokenId] ?? Number.NEGATIVE_INFINITY) - logSumExp;

  return {
    confidence: Math.exp(logProb),
    logProb,
  };
}
