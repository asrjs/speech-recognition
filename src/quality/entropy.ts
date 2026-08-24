/**
 * Entropy Gate — detects uncertain logit distributions.
 *
 * Algorithm (matches whisper.cpp):
 *   H = -sum(p_i * ln(p_i))  where p_i = softmax(logits_i)
 *   Average entropy across all timesteps.
 *   reject if avgEntropy > threshold (default 2.4 nats)
 *
 * Model-agnostic. Pure function. No ONNX dependency.
 */

import type { QualityGate, QualityGateContext, QualityGateResult, TokenQualityTrace } from './types.js';

export function entropyGate(threshold: number = 2.4): QualityGate {
  return (
    _text: string,
    _tokens: readonly number[],
    logits: readonly Float32Array[],
    vocabSize: number,
    context?: QualityGateContext,
  ): QualityGateResult => {
    const traces = context?.tokenTraces;
    if (traces && traces.length > 0) {
      return verdictFromAvgEntropy(averageTraceEntropy(traces), threshold);
    }

    if (logits.length === 0) return { verdict: 'accept', entropy: 0 };

    let sumEntropy = 0;
    let count = 0;

    for (const logitVec of logits) {
      const maxLogit = Math.max(...logitVec.subarray(0, vocabSize));
      let sumExp = 0;
      for (let j = 0; j < vocabSize; j++) {
        sumExp += Math.exp(logitVec[j]! - maxLogit);
      }
      let h = 0;
      for (let j = 0; j < vocabSize; j++) {
        const p = Math.exp(logitVec[j]! - maxLogit) / sumExp;
        if (p > 0) {
          h -= p * Math.log(p);
        }
      }
      sumEntropy += h;
      count++;
    }

    if (count === 0) return { verdict: 'accept', entropy: 0 };
    return verdictFromAvgEntropy(sumEntropy / count, threshold);
  };
}

function averageTraceEntropy(traces: readonly TokenQualityTrace[]): number {
  let sum = 0;
  for (const trace of traces) sum += trace.entropy;
  return sum / traces.length;
}

function verdictFromAvgEntropy(avgEntropy: number, threshold: number): QualityGateResult {
  if (avgEntropy > threshold) {
    return {
      verdict: 'reject',
      entropy: avgEntropy,
      reason: `entropy_too_high (${avgEntropy.toFixed(2)} > ${threshold})`,
    };
  }
  return { verdict: 'accept', entropy: avgEntropy };
}
