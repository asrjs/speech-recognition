/**
 * No-Speech Gate — detects silence/background-noise segments.
 *
 * Algorithm (matches faster-whisper):
 *   noSpeechProb = softmax(raw decoder-init logits)[no_speech_token_id]
 *   reject as 'no_speech' if:
 *     noSpeechProb > noSpeechThreshold (default 0.6)
 *     AND avgLogProb < logProbThreshold (default -1.0)
 *
 * Whisper-specific (token 50362 is the compatibility fallback). For non-Whisper
 * models, VAD handles this better.
 */

import type { QualityGate, QualityGateContext, QualityGateResult } from './types.js';
import { logProbGate } from './log-probability.js';

const NO_SPEECH_TOKEN = 50362;

export function noSpeechGate(
  noSpeechThreshold: number = 0.6,
  logProbThreshold: number = -1.0,
): QualityGate {
  return (
    text: string,
    tokens: readonly number[],
    logits: readonly Float32Array[],
    vocabSize: number,
    context?: QualityGateContext,
  ): QualityGateResult => {
    const firstLogits = context?.noSpeechLogits ?? logits[0];
    if (!firstLogits) return { verdict: 'accept', noSpeechProb: 0 };

    const noSpeechTokenId = context?.noSpeechTokenId ?? NO_SPEECH_TOKEN;
    if (vocabSize <= noSpeechTokenId || noSpeechTokenId < 0) {
      return { verdict: 'accept', noSpeechProb: 0 };
    }

    const sliceStart = Math.max(0, firstLogits.length - vocabSize);
    const sliceEnd = Math.min(firstLogits.length, sliceStart + vocabSize);
    const noSpeechIndex = sliceStart + noSpeechTokenId;
    if (noSpeechIndex >= sliceEnd) return { verdict: 'accept', noSpeechProb: 0 };

    let maxLogit = Number.NEGATIVE_INFINITY;
    for (let j = sliceStart; j < sliceEnd; j++) {
      if (firstLogits[j]! > maxLogit) maxLogit = firstLogits[j]!;
    }
    let sumExp = 0;
    for (let j = sliceStart; j < sliceEnd; j++) {
      sumExp += Math.exp(firstLogits[j]! - maxLogit);
    }
    const noSpeechProb = Math.exp(firstLogits[noSpeechIndex]! - maxLogit) / sumExp;

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
