import { argmax, confidenceFromLogits, tokenQualityFromLogits } from '@asrjs/speech-recognition/inference';
import { describe, expect, it } from 'vitest';

describe('inference math helpers', () => {
  it('finds the maximum index within an optional slice', () => {
    expect(argmax(new Float32Array([1, 5, 3]))).toBe(1);
    expect(argmax(new Float32Array([1, 5, 9, 3]), 1, 2)).toBe(2);
  });

  it('computes confidence and log-probability from logits', () => {
    const logits = new Float32Array([1, 3, 2]);
    const result = confidenceFromLogits(logits, 1, 3);

    expect(result.confidence).toBeGreaterThan(0.6);
    expect(result.confidence).toBeLessThan(1);
    expect(result.logProb).toBeLessThan(0);
  });

  it('keeps the confidence-only fast path numerically aligned with full quality', () => {
    const logits = new Float32Array([0.25, -1.5, 4.75, 0.5, 2.25]);
    const confidence = confidenceFromLogits(logits, 2, logits.length);
    const quality = tokenQualityFromLogits(logits, 2, logits.length);

    expect(confidence.confidence).toBeCloseTo(quality.confidence, 10);
    expect(confidence.logProb).toBeCloseTo(quality.logProb, 10);
  });

  it('computes chosen-token logprob and distribution entropy together', () => {
    const peaked = new Float32Array([1, 8, 2]);
    const uniform = new Float32Array([1, 1, 1]);
    const peakedQuality = tokenQualityFromLogits(peaked, 1);
    const uniformQuality = tokenQualityFromLogits(uniform, 0);

    expect(peakedQuality.logProb).toBeGreaterThan(uniformQuality.logProb);
    expect(peakedQuality.entropy).toBeLessThan(uniformQuality.entropy);
    expect(uniformQuality.entropy).toBeCloseTo(Math.log(3), 5);
    expect(peakedQuality.confidence).toBeCloseTo(Math.exp(peakedQuality.logProb), 10);
  });

  it('handles edge cases in confidenceFromLogits cleanly', () => {
    // 1. Non-finite / all -Infinity logits where expSum <= 0 or not finite
    const allNegInf = new Float32Array([-Infinity, -Infinity, -Infinity]);
    const negInfResult = confidenceFromLogits(allNegInf, 0, 3);
    expect(negInfResult.confidence).toBe(0);
    expect(negInfResult.logProb).toBe(Number.NEGATIVE_INFINITY);

    const allNaN = new Float32Array([NaN, NaN, NaN]);
    const nanResult = confidenceFromLogits(allNaN, 0, 3);
    expect(nanResult.confidence).toBe(0);
    expect(nanResult.logProb).toBe(Number.NEGATIVE_INFINITY);

    // 2. Out of bounds tokenId
    const normalLogits = new Float32Array([2.0, 1.0, 0.0]);
    const outOfBoundsHigh = confidenceFromLogits(normalLogits, 99, 3);
    expect(outOfBoundsHigh.confidence).toBe(0);
    expect(outOfBoundsHigh.logProb).toBe(Number.NEGATIVE_INFINITY);

    const outOfBoundsNeg = confidenceFromLogits(normalLogits, -1, 3);
    expect(outOfBoundsNeg.confidence).toBe(0);
    expect(outOfBoundsNeg.logProb).toBe(Number.NEGATIVE_INFINITY);

    // 3. Logits shorter than vocabSize
    const shortLogits = new Float32Array([5.0]);
    const shortResult = confidenceFromLogits(shortLogits, 0, 3);
    expect(shortResult.confidence).toBeGreaterThan(0);
    expect(Number.isFinite(shortResult.confidence)).toBe(true);

    // Empty array / vocabSize 0
    const emptyLogits = new Float32Array([]);
    const emptyResult = confidenceFromLogits(emptyLogits, 0, 0);
    expect(emptyResult.confidence).toBe(0);
    expect(emptyResult.logProb).toBe(Number.NEGATIVE_INFINITY);

    // 4. Very large positive logits (numerical stability under softmax max subtraction)
    const largeLogits = new Float32Array([1000, 1000, 1000]);
    const largeResult = confidenceFromLogits(largeLogits, 1, 3);
    expect(largeResult.confidence).toBeCloseTo(1 / 3, 5);
    expect(largeResult.logProb).toBeCloseTo(-Math.log(3), 5);
  });
});
