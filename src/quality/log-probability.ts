/**
 * Log Probability Gate — detects low-confidence decode output.
 *
 * Algorithm (matches faster-whisper):
 *   For each token i at position t:
 *     logits[t] → softmax probabilities
 *     logProb_i = log(prob[chosenToken_i])
 *   avgLogProb = mean(logProbs)
 *   reject if avgLogProb < threshold (default -1.0)
 *
 * Model-agnostic. Pure function. No ONNX dependency.
 */

import type { QualityGate, QualityGateResult } from './types.js';

export function logProbGate(threshold: number = -1.0): QualityGate {
  return (_text: string, tokens: readonly number[], logits: readonly Float32Array[], vocabSize: number): QualityGateResult => {
    if (tokens.length === 0 || logits.length === 0) {
      return { verdict: 'accept', avgLogProb: 0 };
    }

    let sumLogProb = 0;
    let count = 0;

    for (let i = 0; i < tokens.length && i < logits.length; i++) {
      const logitVec = logits[i]!;
      const chosenId = tokens[i]!;
      if (chosenId >= vocabSize) continue;

      const maxLogit = Math.max(...logitVec.subarray(0, vocabSize));
      let sumExp = 0;
      for (let j = 0; j < vocabSize; j++) {
        sumExp += Math.exp(logitVec[j]! - maxLogit);
      }
      const prob = Math.exp(logitVec[chosenId]! - maxLogit) / sumExp;
      if (prob > 0) {
        sumLogProb += Math.log(prob);
        count++;
      }
    }

    if (count === 0) return { verdict: 'accept', avgLogProb: 0 };

    const avgLogProb = sumLogProb / count;

    if (avgLogProb < threshold) {
      return {
        verdict: 'reject',
        avgLogProb,
        reason: `avg_logprob_too_low (${avgLogProb.toFixed(3)} < ${threshold})`,
      };
    }
    return { verdict: 'accept', avgLogProb };
  };
}
