/**
 * No-Speech Gate — detects silence/background-noise segments.
 *
 * Algorithm (matches faster-whisper):
 *   noSpeechProb = softmax(first_logits)[50362]
 *   reject as 'no_speech' if:
 *     noSpeechProb > noSpeechThreshold (default 0.6)
 *     AND avgLogProb < logProbThreshold (default -1.0)
 *
 * Whisper-specific (token 50362). For non-Whisper models, VAD handles this better.
 */

import type { QualityGate, QualityGateResult } from './types.js';
import { logProbGate } from './log-probability.js';

const NO_SPEECH_TOKEN = 50362;

export function noSpeechGate(
  noSpeechThreshold: number = 0.6,
  logProbThreshold: number = -1.0,
): QualityGate {
  return (text: string, tokens: readonly number[], logits: readonly Float32Array[], vocabSize: number): QualityGateResult => {
    if (logits.length === 0) return { verdict: 'accept', noSpeechProb: 0 };

    const firstLogits = logits[0]!;
    if (vocabSize <= NO_SPEECH_TOKEN) return { verdict: 'accept', noSpeechProb: 0 };

    const sliceStart = firstLogits.length - vocabSize;
    const maxLogit = Math.max(...firstLogits.subarray(sliceStart, sliceStart + vocabSize));
    let sumExp = 0;
    for (let j = sliceStart; j < sliceStart + vocabSize; j++) {
      sumExp += Math.exp(firstLogits[j]! - maxLogit);
    }
    const noSpeechProb = Math.exp(firstLogits[sliceStart + NO_SPEECH_TOKEN]! - maxLogit) / sumExp;

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
