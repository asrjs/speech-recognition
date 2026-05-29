/**
 * Temperature Fallback Retry Loop for Whisper.
 *
 * Algorithm (matches faster-whisper):
 *   1. Try transcribe at temperature[0] (default 0.0 for greedy)
 *   2. Run quality gates → if 'accept', return
 *   3. If 'no_speech', return immediately with empty text
 *   4. If 'reject', try next temperature
 *   5. After all temperatures exhausted, return last result
 *
 * The transcribe function is generic — no Whisper coupling.
 * The caller provides a function that runs a single decode at a given temperature
 * and returns the generated text, tokens, and per-token logits.
 */

import type { QualityGate, QualityGateResult } from './enhanced-types.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Default temperature schedule matching faster-whisper. */
export const DEFAULT_TEMPERATURES: readonly number[] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

export interface FallbackResult<T> {
  /** The final decode result. */
  readonly result: T;
  /** Temperature that produced the accepted (or last) result. */
  readonly temperature: number;
  /** Number of transcribe attempts made. */
  readonly attempts: number;
  /** Per-attempt gate results. */
  readonly gateResults: readonly QualityGateResult[];
}

// ---------------------------------------------------------------------------
// Transcribe function shape
// ---------------------------------------------------------------------------

/** Shape of a transcribe function that temperature fallback wraps. */
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

/**
 * Run transcribe with temperature fallback.
 *
 * @param transcribeFn — called with each temperature, returns decode result + quality data
 * @param gates — quality gates to evaluate each result
 * @param temperatures — temperature schedule (default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
 */
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

    // Evaluate all quality gates
    // Note: quality gates receive the attempt context via a temperature-aware
    // evaluation. The standard gates (compression, logprob, entropy, no-speech)
    // don't use temperature, but custom gates might.
    const verdicts: QualityGateResult[] = [];
    for (const gate of gates) {
      const result = gate(
        attempt.text,
        attempt.tokens,
        attempt.logits,
        attempt.vocabSize,
      );
      verdicts.push(result);
      if (result.verdict !== 'accept') break;
    }

    // Check for no_speech first (immediate return)
    const noSpeech = verdicts.find((v) => v.verdict === 'no_speech');
    if (noSpeech) {
      gateResults.push(...verdicts);
      return {
        result: attempt.result, // could be empty-text result
        temperature,
        attempts: gateResults.length,
        gateResults,
      };
    }

    // Check if all gates accepted
    const allAccepted = verdicts.every((v) => v.verdict === 'accept');
    if (allAccepted) {
      gateResults.push(...verdicts);
      return {
        result: attempt.result,
        temperature,
        attempts: gateResults.length,
        gateResults,
      };
    }

    // Rejected — record and try next
    gateResults.push(...verdicts);
  }

  // All temperatures exhausted — return last result
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
