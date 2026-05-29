/**
 * Temperature Fallback Retry Loop.
 *
 * Algorithm (matches faster-whisper):
 *   1. Try transcribe at temperature[0] (default 0.0 for greedy)
 *   2. Run quality gates → if 'accept', return
 *   3. If 'no_speech', return immediately with empty text
 *   4. If 'reject', try next temperature
 *   5. After all temperatures exhausted, return last result
 *
 * Generic — no model coupling. Works with any transcribe function.
 */

import type { QualityGate, QualityGateResult } from './types.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const DEFAULT_TEMPERATURES: readonly number[] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface FallbackResult<T> {
  readonly result: T;
  readonly temperature: number;
  readonly attempts: number;
  readonly gateResults: readonly QualityGateResult[];
}

export interface TranscribeAttempt<T> {
  result: T;
  text: string;
  tokens: readonly number[];
  logits: readonly Float32Array[];
  vocabSize: number;
}

// ---------------------------------------------------------------------------
// Retry loop
// ---------------------------------------------------------------------------

export async function withTemperatureFallback<T>(
  transcribeFn: (temperature: number) => Promise<TranscribeAttempt<T>>,
  gates: readonly QualityGate[],
  temperatures: readonly number[] = DEFAULT_TEMPERATURES,
): Promise<FallbackResult<T>> {
  const gateResults: QualityGateResult[] = [];
  let lastAttempt: TranscribeAttempt<T> | null = null;

  for (const temperature of temperatures) {
    const attempt = await transcribeFn(temperature);
    lastAttempt = attempt;

    const verdicts: QualityGateResult[] = [];
    for (const gate of gates) {
      const result = gate(attempt.text, attempt.tokens, attempt.logits, attempt.vocabSize);
      verdicts.push(result);
      if (result.verdict !== 'accept') break;
    }

    const noSpeech = verdicts.find((v) => v.verdict === 'no_speech');
    if (noSpeech) {
      gateResults.push(...verdicts);
      return { result: attempt.result, temperature, attempts: gateResults.length, gateResults };
    }

    const allAccepted = verdicts.every((v) => v.verdict === 'accept');
    if (allAccepted) {
      gateResults.push(...verdicts);
      return { result: attempt.result, temperature, attempts: gateResults.length, gateResults };
    }

    gateResults.push(...verdicts);
  }

  if (!lastAttempt) {
    throw new Error('withTemperatureFallback: no temperatures provided');
  }
  return {
    result: lastAttempt.result,
    temperature: temperatures[temperatures.length - 1]!,
    attempts: gateResults.length,
    gateResults,
  };
}
