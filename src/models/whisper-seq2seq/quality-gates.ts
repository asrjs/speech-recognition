/**
 * Whisper Quality Gates — pure functions for evaluating decode quality.
 *
 * All gates match the algorithms used in faster-whisper and whisper.cpp.
 * No ONNX dependency. No imports from executor.ts.
 *
 * References:
 *   - faster-whisper: transcription.py (compression_ratio, log_prob, no_speech)
 *   - whisper.cpp: whisper-full.cpp (entropy threshold)
 */

import { deflate } from 'pako';
import type { QualityGate, QualityGateResult } from './enhanced-types.js';

// ---------------------------------------------------------------------------
// Gate 1: Compression Ratio
// ---------------------------------------------------------------------------

/**
 * Evaluates whether text is "natural" by checking its zlib compression ratio.
 *
 * Algorithm (matches faster-whisper/whisper.cpp):
 *   textBytes = text.encode('utf-8')
 *   ratio = len(textBytes) / len(deflate(textBytes))
 *   reject if ratio > threshold (default 2.4)
 *
 * Highly repetitive/hallucinated text compresses very well → high ratio → reject.
 */
export function compressionRatioGate(threshold: number = 2.4): QualityGate {
  return (text: string): QualityGateResult => {
    const bytes = new TextEncoder().encode(text);
    const compressed = deflate(bytes, { level: 6 });
    const compressionRatio = bytes.length / Math.max(compressed.length, 1);

    if (compressionRatio > threshold) {
      return {
        verdict: 'reject',
        compressionRatio,
        reason: `compression_ratio_too_high (${compressionRatio.toFixed(2)} > ${threshold})`,
      };
    }
    return { verdict: 'accept', compressionRatio };
  };
}

// ---------------------------------------------------------------------------
// Gate 2: Log Probability
// ---------------------------------------------------------------------------

/**
 * Computes average log probability across generated tokens.
 *
 * Algorithm (matches faster-whisper):
 *   For each token i at position timestep t:
 *     logits[t] → softmax probabilities
 *     logProb_i = log(prob[chosenToken_i])
 *   avgLogProb = mean(logProbs)
 *   reject if avgLogProb < threshold (default -1.0)
 *
 * Lower log prob = less confident decode.
 */
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

      // Softmax: p_i = exp(logits[i] - max_logit) / sum(exp(logits - max_logit))
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

// ---------------------------------------------------------------------------
// Gate 3: No-Speech
// ---------------------------------------------------------------------------

/**
 * Checks whether the first generated token is the no-speech token.
 *
 * Algorithm (matches faster-whisper):
 *   noSpeechProb = softmax(logits[0])[50362]
 *   reject as 'no_speech' if:
 *     noSpeechProb > noSpeechThreshold (default 0.6)
 *     AND avgLogProb < logProbThreshold (default -1.0)
 *
 * Dual condition prevents false positives.
 */
export function noSpeechGate(
  noSpeechThreshold: number = 0.6,
  logProbThreshold: number = -1.0,
): QualityGate {
  const NO_SPEECH_TOKEN = 50362;

  return (text: string, tokens: readonly number[], logits: readonly Float32Array[], vocabSize: number): QualityGateResult => {
    if (logits.length === 0) return { verdict: 'accept', noSpeechProb: 0 };

    // Get first token logits
    const firstLogits = logits[0]!;
    if (vocabSize <= NO_SPEECH_TOKEN) return { verdict: 'accept', noSpeechProb: 0 };

    // Softmax for no-speech token
    const sliceStart = firstLogits.length - vocabSize;
    const maxLogit = Math.max(...firstLogits.subarray(sliceStart, sliceStart + vocabSize));
    let sumExp = 0;
    for (let j = sliceStart; j < sliceStart + vocabSize; j++) {
      sumExp += Math.exp(firstLogits[j]! - maxLogit);
    }
    const noSpeechProb = Math.exp(firstLogits[sliceStart + NO_SPEECH_TOKEN]! - maxLogit) / sumExp;

    // Also compute avgLogProb for the dual check
    const logProbResult = logProbGate(logProbThreshold)(text, tokens, logits, vocabSize);
    const avgLogProb = logProbResult.avgLogProb ?? 0;

    if (noSpeechProb > noSpeechThreshold && avgLogProb < logProbThreshold) {
      return {
        verdict: 'no_speech',
        noSpeechProb,
        avgLogProb,
        reason: `no_speech_detected (prob=${noSpeechProb.toFixed(3)} > ${noSpeechThreshold})`,
      };
    }

    return { verdict: 'accept', noSpeechProb, avgLogProb };
  };
}

// ---------------------------------------------------------------------------
// Gate 4: Entropy
// ---------------------------------------------------------------------------

/**
 * Computes Shannon entropy of the logit distribution.
 *
 * Algorithm (matches whisper.cpp):
 *   H = -sum(p_i * ln(p_i))  where p_i = softmax(logits_i)
 *   Average entropy across all timesteps.
 *   reject if avgEntropy > threshold (default 2.4 nats)
 *
 * High entropy = uncertain model → possible hallucination.
 */
export function entropyGate(threshold: number = 2.4): QualityGate {
  return (_text: string, _tokens: readonly number[], logits: readonly Float32Array[], vocabSize: number): QualityGateResult => {
    if (logits.length === 0) return { verdict: 'accept', entropy: 0 };

    let sumEntropy = 0;
    let count = 0;

    for (const logitVec of logits) {
      const maxLogit = Math.max(...logitVec.subarray(0, vocabSize));
      let sumExp = 0;
      for (let j = 0; j < vocabSize; j++) {
        sumExp += Math.exp(logitVec[j]! - maxLogit);
      }
      // Entropy: H = -sum(p * ln(p))
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

    const avgEntropy = sumEntropy / count;

    if (avgEntropy > threshold) {
      return {
        verdict: 'reject',
        entropy: avgEntropy,
        reason: `entropy_too_high (${avgEntropy.toFixed(2)} > ${threshold})`,
      };
    }
    return { verdict: 'accept', entropy: avgEntropy };
  };
}

// ---------------------------------------------------------------------------
// Composite Runner
// ---------------------------------------------------------------------------

/**
 * Run multiple quality gates. Short-circuits on first non-accept.
 */
export function evaluateGates(
  text: string,
  tokens: readonly number[],
  logits: readonly Float32Array[],
  vocabSize: number,
  gates: readonly QualityGate[],
): QualityGateResult {
  let lastResult: QualityGateResult = { verdict: 'accept' };

  for (const gate of gates) {
    const result = gate(text, tokens, logits, vocabSize);
    if (result.verdict !== 'accept') {
      return result; // short-circuit
    }
    lastResult = result;
  }

  return lastResult;
}
