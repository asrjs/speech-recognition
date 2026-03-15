import { argmax, confidenceFromLogits } from '@asrjs/speech-recognition/inference';
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
});
