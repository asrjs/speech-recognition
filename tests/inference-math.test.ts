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
});
