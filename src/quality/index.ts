/**
 * Quality Gates Module — hallucination suppression for ASR output.
 *
 * Model-agnostic. Works with any ASR model that produces text + token logits.
 * Independently importable: @asrjs/speech-recognition/quality
 *
 * Features:
 *   - Compression ratio gate (detects repetitive output)
 *   - Log probability gate (detects low-confidence output)
 *   - Entropy gate (detects uncertain logit distributions)
 *   - No-speech gate (detects silence segments, Whisper-specific)
 *   - Temperature fallback retry loop
 *   - Composite gate evaluator
 */

export type {
  QualityVerdict,
  QualityGate,
  QualityGateContext,
  QualityGateResult,
  SegmentQualityMetrics,
  TokenQualityTrace,
} from './types.js';

export { compressionRatioGate } from './compression-ratio.js';
export { logProbGate } from './log-probability.js';
export { entropyGate } from './entropy.js';
export { noSpeechGate } from './no-speech.js';
export {
  DEFAULT_TEMPERATURES,
  withTemperatureFallback,
  type FallbackResult,
  type TranscribeAttempt,
} from './temperature-fallback.js';

/**
 * Run multiple quality gates. Short-circuits on first non-accept.
 */
export function evaluateGates(
  text: string,
  tokens: readonly number[],
  logits: readonly Float32Array[],
  vocabSize: number,
  gates: readonly import('./types.js').QualityGate[],
  context?: import('./types.js').QualityGateContext,
): import('./types.js').QualityGateResult {
  let lastResult: import('./types.js').QualityGateResult = { verdict: 'accept' };

  for (const gate of gates) {
    const result = gate(text, tokens, logits, vocabSize, context);
    if (result.verdict !== 'accept') {
      return result;
    }
    lastResult = result;
  }

  return lastResult;
}
