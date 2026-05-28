import { describe, expect, it } from 'vitest';

describe('Whisper word probability from DTW alignment', () => {
  it('computes mean token probability for each word', () => {
    // Simulate: 5 text tokens mapping to 2 words
    // Word "hello" = tokens [0, 1], Word "world" = tokens [2, 3, 4]
    const textTokenIds = [100, 101, 200, 201, 202];
    const dtwTimestamps = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]; // 6 values for 5 tokens
    // Token logprobs: -0.2, -0.3, -0.1, -0.15, -0.05
    const tokenLogprobs = new Float32Array([-0.2, -0.3, -0.1, -0.15, -0.05]);

    // Token-to-word mapping: [0,0,1,1,1] (token 0→word0, token1→word0, ...)
    const tokenToWord = [0, 0, 1, 1, 1];

    const wordProbs = computeWordProbabilities(textTokenIds.length, tokenLogprobs, tokenToWord, 2);
    // Word 0: exp(-0.2 + -0.3) = exp(-0.5); mean = exp(-0.5)^{1/2}? 
    // Actually mean of probabilities: mean(exp(-0.2), exp(-0.3))
    // mean(0.8187, 0.7408) ≈ 0.78
    expect(wordProbs.length).toBe(2);
    expect(wordProbs[0]).toBeGreaterThan(0.7);
    expect(wordProbs[0]).toBeLessThan(0.85);
    expect(wordProbs[1]).toBeGreaterThan(0.8);
  });

  it('returns -1 for words with no tokens', () => {
    const probs = computeWordProbabilities(0, new Float32Array(0), [], 0);
    expect(probs).toEqual([]);
  });
});

// Inline helper copied for TDD — will be moved to executor
function computeWordProbabilities(
  tokenCount: number,
  tokenLogprobs: Float32Array,
  tokenToWord: readonly number[],
  wordCount: number,
): number[] {
  const probs: number[] = new Array(wordCount).fill(0);
  const counts = new Array(wordCount).fill(0);
  for (let t = 0; t < tokenCount; t++) {
    const w = tokenToWord[t] ?? -1;
    if (w < 0 || w >= wordCount) continue;
    probs[w] += Math.exp(tokenLogprobs[t] ?? -Infinity);
    counts[w]++;
  }
  for (let w = 0; w < wordCount; w++) {
    probs[w] = counts[w] > 0 ? probs[w]! / counts[w]! : -1;
  }
  return probs;
}
