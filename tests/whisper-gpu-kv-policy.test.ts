import { describe, expect, it } from 'vitest';
import { assertExperimentalGpuKvCacheIsGreedyOnly } from '../src/models/whisper-seq2seq/executor.js';

describe('experimental GPU-KV policy', () => {
  it('allows greedy argmax', () => {
    expect(() =>
      assertExperimentalGpuKvCacheIsGreedyOnly({
        enabled: true,
        numBeams: 1,
        bestOf: 1,
        temperature: 0,
      }),
    ).not.toThrow();
  });

  it('is a no-op when GPU-KV is disabled', () => {
    expect(() =>
      assertExperimentalGpuKvCacheIsGreedyOnly({
        enabled: false,
        numBeams: 5,
        bestOf: 5,
        temperature: 0.2,
      }),
    ).not.toThrow();
  });

  it.each([
    { numBeams: 2, bestOf: 1, temperature: 0 },
    { numBeams: 1, bestOf: 2, temperature: 0 },
    { numBeams: 1, bestOf: 1, temperature: 0.2 },
  ])('rejects beam, best_of, and temperature while GPU-KV is enabled %#', (options) => {
    expect(() =>
      assertExperimentalGpuKvCacheIsGreedyOnly({
        enabled: true,
        ...options,
      }),
    ).toThrow(/greedy argmax/);
  });
});
